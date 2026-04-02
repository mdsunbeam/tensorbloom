/**
 * Shape inference — client-side heuristic fallback + sidecar call.
 * Propagates tensor shapes through the graph without needing Python.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import { isDatasetNode, DATASET_METADATA, resolveDataNodeShape, type ProbeResult } from "./dataset-metadata";

export interface ShapeInfo {
  shape: number[];
  dtype: string;
}

export type ShapeMap = Record<string, ShapeInfo>;

export interface ShapeError {
  message: string;
  type: "root" | "cascaded";
  sourceNodeId: string;
  expectedShape?: number[];
  receivedShape?: number[];
  paramHint?: string;
}

/**
 * Client-side shape inference for common layers.
 * Returns a map of nodeId -> output shape.
 */
export function inferShapesClient(
  nodes: Node<NodeData>[],
  edges: Edge[],
): { shapes: ShapeMap; errors: Record<string, ShapeError> } {
  const shapes: ShapeMap = {};
  const errors: Record<string, ShapeError> = {};

  // Build adjacency (handle-aware for code block support)
  interface IncomingEdge { sourceNodeId: string; sourceHandle: string; targetHandle: string }
  const incoming = new Map<string, string[]>();
  const incomingDetailed = new Map<string, IncomingEdge[]>();
  for (const e of edges) {
    const list = incoming.get(e.target) ?? [];
    list.push(e.source);
    incoming.set(e.target, list);

    const detailed = incomingDetailed.get(e.target) ?? [];
    detailed.push({
      sourceNodeId: e.source,
      sourceHandle: e.sourceHandle ?? "out",
      targetHandle: e.targetHandle ?? "in",
    });
    incomingDetailed.set(e.target, detailed);
  }

  // Topological sort
  const sorted = topoSort(nodes, edges);
  if (!sorted) return { shapes, errors };

  const nodeMap = new Map(nodes.map((n) => [n.id, n]));

  // Helper: resolve input shape considering multi-output handle keys
  const resolveSourceShape = (sourceId: string, sourceHandle: string): number[] | null => {
    // Try handle-specific key first (for multi-output code blocks)
    const handleKey = `${sourceId}:${sourceHandle}`;
    if (shapes[handleKey]) return shapes[handleKey].shape;
    // Fall back to node-level shape
    if (shapes[sourceId]) return shapes[sourceId].shape;
    return null;
  };

  for (const node of sorted) {
    const mt = node.data.moduleType;
    const p = node.data.params;
    const sources = incoming.get(node.id) ?? [];
    const detailedEdges = incomingDetailed.get(node.id) ?? [];
    const inputShape = detailedEdges.length > 0
      ? resolveSourceShape(detailedEdges[0].sourceNodeId, detailedEdges[0].sourceHandle)
      : null;

    // Check if any upstream node has an error → mark as cascaded
    const upstreamError = sources.find((s) => errors[s]);
    if (upstreamError) {
      const rootErr = errors[upstreamError];
      const rootSource = rootErr.type === "cascaded" ? rootErr.sourceNodeId : upstreamError;
      errors[node.id] = {
        message: `Upstream error from ${rootSource}`,
        type: "cascaded",
        sourceNodeId: rootSource,
      };
      // Still try to propagate shape if possible (pass-through)
      if (inputShape) shapes[node.id] = { shape: inputShape, dtype: "float32" };
      continue;
    }

    try {
      if (mt === "Input") {
        const shapeStr = String(p.shape ?? "[1, 3, 224, 224]");
        const parsed = JSON.parse(shapeStr.replace(/\(/g, "[").replace(/\)/g, "]"));
        shapes[node.id] = { shape: parsed, dtype: String(p.dtype ?? "float32") };
      } else if (isDatasetNode(mt)) {
        // Universal Data node: resolve shape from params + per-handle shapes
        if (mt === "Data") {
          const resolved = resolveDataNodeShape(p);
          if (resolved) {
            const probed = p._probed as ProbeResult | undefined;
            const dtype = probed?.sample_dtype ?? "float32";
            shapes[node.id] = { shape: resolved.inputShape, dtype };
            // Store per-handle shapes for multi-output Data nodes
            const dataOutputs = (node.data as Record<string, unknown>).dataOutputs as Array<{id: string; dtype: string}> | undefined;
            if (dataOutputs) {
              // x handle = feature shape
              shapes[`${node.id}:x`] = { shape: resolved.inputShape, dtype };
              // y handle = label shape (scalar class index for classification)
              shapes[`${node.id}:y`] = { shape: [1], dtype: "long" };
              // Additional handles get unknown shape
              for (const o of dataOutputs) {
                if (o.id !== "x" && o.id !== "y" && !shapes[`${node.id}:${o.id}`]) {
                  shapes[`${node.id}:${o.id}`] = { shape: [1], dtype: o.dtype ?? "float32" };
                }
              }
            }
          }
        } else {
          // Legacy dataset node
          const meta = DATASET_METADATA[mt];
          if (meta) {
            shapes[node.id] = { shape: meta.inputShape, dtype: "float32" };
          }
        }
      } else if (mt === "Output") {
        if (inputShape) shapes[node.id] = { shape: inputShape, dtype: "float32" };
      } else if (mt === "Linear") {
        if (inputShape) {
          const inFeatures = Number(p.in_features);
          const lastDim = inputShape[inputShape.length - 1];
          if (lastDim !== inFeatures) {
            errors[node.id] = {
              message: `Dimension mismatch: input last dim is ${lastDim}, but in_features=${inFeatures}`,
              type: "root",
              sourceNodeId: node.id,
              expectedShape: [...inputShape.slice(0, -1), inFeatures],
              receivedShape: inputShape,
              paramHint: "in_features",
            };
          }
          // Warn if input is 3D+ AND in_features doesn't match last dim
          // (likely missing a Flatten). Don't warn when in_features matches —
          // that's intentional for sequence models (Linear on [B, T, D] → [B, T, out]).
          if (inputShape.length > 2 && lastDim !== inFeatures) {
            const flatSize = inputShape.slice(1).reduce((a, b) => a * b, 1);
            if (!errors[node.id]) {
              errors[node.id] = {
                message: `Linear receives ${inputShape.length}D input ${JSON.stringify(inputShape)} — consider adding a Flatten layer. Flattened size would be ${flatSize}.`,
                type: "root",
                sourceNodeId: node.id,
                receivedShape: inputShape,
              };
            }
          }
          const out = [...inputShape];
          out[out.length - 1] = Number(p.out_features);
          shapes[node.id] = { shape: out, dtype: "float32" };
        }
      } else if (mt === "Conv2d" || mt === "Conv1d" || mt === "Conv3d") {
        if (inputShape) {
          const inChannels = Number(p.in_channels);
          const outChannels = Number(p.out_channels);
          const groups = Number(p.groups ?? 1);
          const channelDim = 1; // N, C, ...
          if (inputShape.length >= 2 && inputShape[channelDim] !== inChannels) {
            errors[node.id] = {
              message: `Channel mismatch: input has ${inputShape[channelDim]} channels, but in_channels=${inChannels}`,
              type: "root",
              sourceNodeId: node.id,
              expectedShape: [...inputShape.slice(0, channelDim), inChannels, ...inputShape.slice(channelDim + 1)],
              receivedShape: inputShape,
              paramHint: "in_channels",
            };
          }
          const expectedDims = mt === "Conv1d" ? 3 : mt === "Conv2d" ? 4 : 5;
          if (inputShape.length !== expectedDims) {
            errors[node.id] = {
              message: `${mt} expects ${expectedDims}D input (got ${inputShape.length}D)`,
              type: "root",
              sourceNodeId: node.id,
              receivedShape: inputShape,
            };
          }
          // Grouped convolution divisibility check
          if (groups > 1) {
            if (inChannels % groups !== 0) {
              errors[node.id] = {
                message: `in_channels (${inChannels}) must be divisible by groups (${groups})`,
                type: "root",
                sourceNodeId: node.id,
              };
            }
            if (outChannels % groups !== 0) {
              errors[node.id] = {
                message: `out_channels (${outChannels}) must be divisible by groups (${groups})`,
                type: "root",
                sourceNodeId: node.id,
              };
            }
          }
          const convOut = computeConvShape(mt, inputShape, p);
          // Check for non-positive spatial dims (kernel too large)
          const spatialStart = 2;
          for (let i = spatialStart; i < convOut.length; i++) {
            if (convOut[i] <= 0) {
              errors[node.id] = {
                message: `${mt} produces non-positive size at dim ${i} (${convOut[i]}). Kernel may be too large for input.`,
                type: "root",
                sourceNodeId: node.id,
                receivedShape: inputShape,
                paramHint: "kernel_size",
              };
              break;
            }
          }
          shapes[node.id] = { shape: convOut, dtype: "float32" };
        }
      } else if (mt === "ConvTranspose2d") {
        if (inputShape) {
          const inChannels = Number(p.in_channels);
          const outChannels = Number(p.out_channels);
          const groups = Number(p.groups ?? 1);
          const stride = Number(p.stride ?? 1);
          const dilation = Number(p.dilation ?? 1);
          const outputPadding = Number(p.output_padding ?? 0);
          if (inputShape.length >= 2 && inputShape[1] !== inChannels) {
            errors[node.id] = {
              message: `Channel mismatch: input has ${inputShape[1]} channels, but in_channels=${inChannels}`,
              type: "root",
              sourceNodeId: node.id,
              expectedShape: [inputShape[0], inChannels, ...inputShape.slice(2)],
              receivedShape: inputShape,
              paramHint: "in_channels",
            };
          }
          // output_padding must be < stride and < dilation
          if (outputPadding >= stride || outputPadding >= dilation) {
            errors[node.id] = {
              message: `output_padding (${outputPadding}) must be < stride (${stride}) and < dilation (${dilation})`,
              type: "root",
              sourceNodeId: node.id,
              paramHint: "output_padding",
            };
          }
          if (groups > 1 && (inChannels % groups !== 0 || outChannels % groups !== 0)) {
            errors[node.id] = {
              message: `in_channels (${inChannels}) and out_channels (${outChannels}) must be divisible by groups (${groups})`,
              type: "root",
              sourceNodeId: node.id,
            };
          }
          shapes[node.id] = { shape: computeConvTransposeShape(inputShape, p), dtype: "float32" };
        }
      } else if (mt === "AdaptiveAvgPool1d") {
        if (inputShape && inputShape.length === 3) {
          const os = Number(p.output_size ?? 1);
          shapes[node.id] = { shape: [inputShape[0], inputShape[1], os], dtype: "float32" };
        }
      } else if (mt === "AdaptiveAvgPool2d") {
        if (inputShape && inputShape.length === 4) {
          const os = p.output_size;
          const outH = Array.isArray(os) ? os[0] : (typeof os === "number" ? os : 1);
          const outW = Array.isArray(os) ? os[1] : (typeof os === "number" ? os : 1);
          shapes[node.id] = { shape: [inputShape[0], inputShape[1], outH, outW], dtype: "float32" };
        } else if (inputShape && inputShape.length === 3) {
          // 3D input: treat as adaptive pool on last dim
          const os = p.output_size;
          const outL = Array.isArray(os) ? os[0] : (typeof os === "number" ? os : 1);
          shapes[node.id] = { shape: [inputShape[0], inputShape[1], outL], dtype: "float32" };
        }
      } else if (mt === "Flatten") {
        if (inputShape) {
          const startDim = Number(p.start_dim ?? 1);
          const endDim = Number(p.end_dim ?? -1);
          const ed = endDim < 0 ? inputShape.length + endDim : endDim;
          const before = inputShape.slice(0, startDim);
          const flatDims = inputShape.slice(startDim, ed + 1);
          const after = inputShape.slice(ed + 1);
          const flat = flatDims.reduce((a, b) => a * b, 1);
          shapes[node.id] = { shape: [...before, flat, ...after], dtype: "float32" };
        }
      } else if (mt === "Embedding") {
        if (inputShape) {
          shapes[node.id] = { shape: [...inputShape, Number(p.embedding_dim)], dtype: "float32" };
        }
      } else if (mt === "LSTM" || mt === "GRU") {
        if (inputShape) {
          const inputSize = Number(p.input_size);
          if (inputShape.length >= 2) {
            const lastDim = inputShape[inputShape.length - 1];
            if (lastDim !== inputSize) {
              errors[node.id] = {
                message: `input_size mismatch: input last dim is ${lastDim}, but input_size=${inputSize}`,
                type: "root",
                sourceNodeId: node.id,
                receivedShape: inputShape,
                paramHint: "input_size",
              };
            }
          }
          if (inputShape.length !== 3) {
            errors[node.id] = {
              message: `${mt} expects 3D input (batch, seq, features), got ${inputShape.length}D`,
              type: "root",
              sourceNodeId: node.id,
              receivedShape: inputShape,
            };
          }
          const hs = Number(p.hidden_size);
          const bi = p.bidirectional ? 2 : 1;
          shapes[node.id] = { shape: [inputShape[0], inputShape[1], hs * bi], dtype: "float32" };
        }
      } else if (mt === "Upsample") {
        if (inputShape) {
          if (inputShape.length < 3 || inputShape.length > 5) {
            errors[node.id] = {
              message: `Upsample expects 3D, 4D, or 5D input (got ${inputShape.length}D)`,
              type: "root",
              sourceNodeId: node.id,
              receivedShape: inputShape,
            };
          }
          if (inputShape.length === 4) {
            const sf = Number(p.scale_factor ?? 2);
            shapes[node.id] = {
              shape: [inputShape[0], inputShape[1], Math.round(inputShape[2] * sf), Math.round(inputShape[3] * sf)],
              dtype: "float32",
            };
          } else if (inputShape.length === 3) {
            const sf = Number(p.scale_factor ?? 2);
            shapes[node.id] = {
              shape: [inputShape[0], inputShape[1], Math.round(inputShape[2] * sf)],
              dtype: "float32",
            };
          } else {
            shapes[node.id] = { shape: inputShape, dtype: "float32" };
          }
        }
      } else if (mt === "Add") {
        if (sources.length >= 2 && shapes[sources[0]] && shapes[sources[1]]) {
          const s1 = shapes[sources[0]].shape;
          const s2 = shapes[sources[1]].shape;
          // Check broadcast compatibility
          if (!areBroadcastable(s1, s2)) {
            errors[node.id] = {
              message: `Shapes ${JSON.stringify(s1)} and ${JSON.stringify(s2)} are not broadcastable for element-wise Add`,
              type: "root",
              sourceNodeId: node.id,
              receivedShape: s1,
              expectedShape: s2,
            };
          }
          shapes[node.id] = { shape: broadcastShape(s1, s2), dtype: "float32" };
        } else if (inputShape) {
          shapes[node.id] = { shape: inputShape, dtype: "float32" };
        }
      } else if (mt === "MatMul") {
        if (sources.length >= 2 && shapes[sources[0]] && shapes[sources[1]]) {
          const s1 = shapes[sources[0]].shape;
          const s2 = shapes[sources[1]].shape;
          // For matmul: last dim of s1 must match second-to-last dim of s2
          if (s1.length >= 2 && s2.length >= 2) {
            if (s1[s1.length - 1] !== s2[s2.length - 2]) {
              errors[node.id] = {
                message: `MatMul inner dims mismatch: ${s1[s1.length - 1]} vs ${s2[s2.length - 2]} (shapes ${JSON.stringify(s1)} @ ${JSON.stringify(s2)})`,
                type: "root",
                sourceNodeId: node.id,
                receivedShape: s1,
                expectedShape: s2,
              };
            }
            // Output shape: batch dims broadcast + [s1[-2], s2[-1]]
            const out = [...s1.slice(0, -1), s2[s2.length - 1]];
            shapes[node.id] = { shape: out, dtype: "float32" };
          } else {
            shapes[node.id] = { shape: inputShape ?? s1, dtype: "float32" };
          }
        } else if (inputShape) {
          shapes[node.id] = { shape: inputShape, dtype: "float32" };
        }
      } else if (mt === "BatchNorm1d" || mt === "BatchNorm2d" || mt === "InstanceNorm1d" || mt === "InstanceNorm2d") {
        if (inputShape) {
          const numFeatures = Number(p.num_features);
          if (inputShape.length >= 2 && inputShape[1] !== numFeatures) {
            errors[node.id] = {
              message: `Feature mismatch: input has ${inputShape[1]} features, but num_features=${numFeatures}`,
              type: "root",
              sourceNodeId: node.id,
              expectedShape: [inputShape[0], numFeatures, ...inputShape.slice(2)],
              receivedShape: inputShape,
              paramHint: "num_features",
            };
          }
          if (mt === "BatchNorm1d" && inputShape.length !== 2 && inputShape.length !== 3) {
            errors[node.id] = {
              message: `BatchNorm1d expects 2D or 3D input (got ${inputShape.length}D)`,
              type: "root",
              sourceNodeId: node.id,
              receivedShape: inputShape,
            };
          } else if (mt === "BatchNorm2d" && inputShape.length !== 4) {
            errors[node.id] = {
              message: `BatchNorm2d expects 4D input (got ${inputShape.length}D)`,
              type: "root",
              sourceNodeId: node.id,
              receivedShape: inputShape,
            };
          }
          shapes[node.id] = { shape: inputShape, dtype: "float32" };
        }
      } else if (mt === "LayerNorm" || mt === "RMSNorm") {
        if (inputShape) {
          // LayerNorm normalized_shape should match trailing dims
          const nsStr = String(p.normalized_shape ?? "");
          if (nsStr) {
            try {
              const ns = JSON.parse(nsStr.replace(/\(/g, "[").replace(/\)/g, "]"));
              const expected = Array.isArray(ns) ? ns : [ns];
              const trailing = inputShape.slice(-expected.length);
              const mismatch = expected.some((v: number, i: number) => v !== trailing[i]);
              if (mismatch) {
                errors[node.id] = {
                  message: `normalized_shape ${JSON.stringify(expected)} doesn't match input trailing dims ${JSON.stringify(trailing)}`,
                  type: "root",
                  sourceNodeId: node.id,
                  receivedShape: inputShape,
                  paramHint: "normalized_shape",
                };
              }
            } catch { /* ignore parse errors */ }
          }
          shapes[node.id] = { shape: inputShape, dtype: "float32" };
        }
      } else if (mt === "GroupNorm") {
        if (inputShape) {
          const numGroups = Number(p.num_groups);
          const numChannels = Number(p.num_channels);
          if (inputShape.length >= 2 && inputShape[1] !== numChannels) {
            errors[node.id] = {
              message: `Channel mismatch: input has ${inputShape[1]} channels, but num_channels=${numChannels}`,
              type: "root",
              sourceNodeId: node.id,
              receivedShape: inputShape,
              paramHint: "num_channels",
            };
          } else if (numChannels % numGroups !== 0) {
            errors[node.id] = {
              message: `num_channels (${numChannels}) must be divisible by num_groups (${numGroups})`,
              type: "root",
              sourceNodeId: node.id,
            };
          }
          shapes[node.id] = { shape: inputShape, dtype: "float32" };
        }
      } else if (mt === "MultiheadAttention") {
        if (inputShape) {
          const embedDim = Number(p.embed_dim);
          const numHeads = Number(p.num_heads);
          if (inputShape.length >= 2) {
            const lastDim = inputShape[inputShape.length - 1];
            if (lastDim !== embedDim) {
              errors[node.id] = {
                message: `embed_dim mismatch: input last dim is ${lastDim}, but embed_dim=${embedDim}`,
                type: "root",
                sourceNodeId: node.id,
                receivedShape: inputShape,
                paramHint: "embed_dim",
              };
            }
          }
          if (embedDim % numHeads !== 0) {
            errors[node.id] = {
              message: `embed_dim (${embedDim}) must be divisible by num_heads (${numHeads})`,
              type: "root",
              sourceNodeId: node.id,
            };
          }
          shapes[node.id] = { shape: inputShape, dtype: "float32" };
        }
      } else if (mt === "Dropout" || mt === "Dropout2d") {
        const prob = Number(p.p ?? 0.5);
        if (prob < 0 || prob > 1) {
          errors[node.id] = {
            message: `Dropout probability must be between 0 and 1, got ${prob}`,
            type: "root",
            sourceNodeId: node.id,
            paramHint: "p",
          };
        }
        if (mt === "Dropout2d" && inputShape && inputShape.length < 4) {
          errors[node.id] = {
            message: `Dropout2d expects 4D+ input (N,C,H,W), got ${inputShape.length}D`,
            type: "root",
            sourceNodeId: node.id,
            receivedShape: inputShape,
          };
        }
        if (inputShape) shapes[node.id] = { shape: inputShape, dtype: "float32" };
      } else if (mt === "PReLU") {
        if (inputShape && inputShape.length >= 2) {
          const numParams = Number(p.num_parameters ?? 1);
          if (numParams !== 1 && numParams !== inputShape[1]) {
            errors[node.id] = {
              message: `PReLU num_parameters (${numParams}) must be 1 or match input channels (${inputShape[1]})`,
              type: "root",
              sourceNodeId: node.id,
              paramHint: "num_parameters",
            };
          }
        }
        if (inputShape) shapes[node.id] = { shape: inputShape, dtype: "float32" };
      } else if (mt === "Softmax") {
        if (inputShape) {
          const dim = Number(p.dim ?? -1);
          const ndim = inputShape.length;
          const resolvedDim = dim < 0 ? ndim + dim : dim;
          if (resolvedDim < 0 || resolvedDim >= ndim) {
            errors[node.id] = {
              message: `Softmax dim=${dim} is out of range for ${ndim}D input`,
              type: "root",
              sourceNodeId: node.id,
              paramHint: "dim",
            };
          }
          shapes[node.id] = { shape: inputShape, dtype: "float32" };
        }
      } else if ([
        "ReLU", "LeakyReLU", "GELU", "SiLU", "Mish", "Sigmoid", "Tanh",
        "ELU", "ReLU6", "Identity",
      ].includes(mt)) {
        // Shape-preserving
        if (inputShape) shapes[node.id] = { shape: inputShape, dtype: "float32" };
      } else if (mt === "Permute") {
        if (inputShape) {
          try {
            const rawDims = p.dims;
            const dims: number[] = typeof rawDims === "string" ? JSON.parse(rawDims) : (Array.isArray(rawDims) ? rawDims : []);
            if (dims.length === inputShape.length) {
              shapes[node.id] = { shape: dims.map((d: number) => inputShape[d]), dtype: "float32" };
            } else {
              shapes[node.id] = { shape: inputShape, dtype: "float32" };
            }
          } catch {
            shapes[node.id] = { shape: inputShape, dtype: "float32" };
          }
        }
      } else if (mt === "Unflatten") {
        if (inputShape) {
          try {
            const dim = Number(p.dim ?? 1);
            const rawSizes = p.unflattened_size;
            const sizes: number[] = typeof rawSizes === "string" ? JSON.parse(rawSizes) : (Array.isArray(rawSizes) ? rawSizes : []);
            if (sizes.length > 0) {
              const out = [...inputShape];
              out.splice(dim, 1, ...sizes);
              shapes[node.id] = { shape: out, dtype: "float32" };
            } else {
              shapes[node.id] = { shape: inputShape, dtype: "float32" };
            }
          } catch {
            shapes[node.id] = { shape: inputShape, dtype: "float32" };
          }
        }
      } else if (mt === "Transformer") {
        // Full encoder-decoder transformer: shape-preserving on src input
        if (inputShape) {
          const dModel = Number(p.d_model);
          const nhead = Number(p.nhead);
          if (inputShape.length >= 2) {
            const lastDim = inputShape[inputShape.length - 1];
            if (lastDim !== dModel) {
              errors[node.id] = {
                message: `d_model mismatch: input last dim is ${lastDim}, but d_model=${dModel}`,
                type: "root",
                sourceNodeId: node.id,
                receivedShape: inputShape,
                paramHint: "d_model",
              };
            }
          }
          if (dModel % nhead !== 0) {
            errors[node.id] = {
              message: `d_model (${dModel}) must be divisible by nhead (${nhead})`,
              type: "root",
              sourceNodeId: node.id,
            };
          }
          shapes[node.id] = { shape: inputShape, dtype: "float32" };
        }
      } else if (["TransformerEncoderLayer", "TransformerDecoderLayer"].includes(mt)) {
        if (inputShape) {
          const dModel = Number(p.d_model);
          const nhead = Number(p.nhead);
          if (inputShape.length >= 2) {
            const lastDim = inputShape[inputShape.length - 1];
            if (lastDim !== dModel) {
              errors[node.id] = {
                message: `d_model mismatch: input last dim is ${lastDim}, but d_model=${dModel}`,
                type: "root",
                sourceNodeId: node.id,
                receivedShape: inputShape,
                paramHint: "d_model",
              };
            }
          }
          if (dModel % nhead !== 0) {
            errors[node.id] = {
              message: `d_model (${dModel}) must be divisible by nhead (${nhead})`,
              type: "root",
              sourceNodeId: node.id,
            };
          }
          shapes[node.id] = { shape: inputShape, dtype: "float32" };
        }
      } else if (mt === "MaxPool2d" || mt === "AvgPool2d") {
        if (inputShape && inputShape.length === 4) {
          // already handled above
        } else if (inputShape) {
          errors[node.id] = {
            message: `${mt} expects 4D input (got ${inputShape.length}D)`,
            type: "root",
            sourceNodeId: node.id,
            receivedShape: inputShape,
          };
        }
        // fall through to compute shape if possible
        if (inputShape && inputShape.length === 4) {
          const k = Number(p.kernel_size ?? 2);
          const s = p.stride != null ? Number(p.stride) : k;
          const pad = Number(p.padding ?? 0);
          const h = Math.floor((inputShape[2] + 2 * pad - k) / s + 1);
          const w = Math.floor((inputShape[3] + 2 * pad - k) / s + 1);
          if (h <= 0 || w <= 0) {
            errors[node.id] = {
              message: `${mt} produces non-positive spatial dims (${h}x${w}). Check kernel_size, stride, padding.`,
              type: "root",
              sourceNodeId: node.id,
              receivedShape: inputShape,
              paramHint: "kernel_size",
            };
          }
          shapes[node.id] = { shape: [inputShape[0], inputShape[1], Math.max(h, 0), Math.max(w, 0)], dtype: "float32" };
        }
      } else if (mt === "Concatenate") {
        // Concat along dim — check compatible shapes
        if (sources.length >= 2 && shapes[sources[0]] && shapes[sources[1]]) {
          const s1 = shapes[sources[0]].shape;
          const s2 = shapes[sources[1]].shape;
          const dim = Number(p.dim ?? 1);
          if (s1.length !== s2.length) {
            errors[node.id] = {
              message: `Cannot concatenate tensors with different number of dimensions (${s1.length}D vs ${s2.length}D)`,
              type: "root",
              sourceNodeId: node.id,
              receivedShape: s1,
              expectedShape: s2,
            };
          } else {
            // Check non-concat dims match
            for (let i = 0; i < s1.length; i++) {
              if (i !== dim && s1[i] !== s2[i]) {
                errors[node.id] = {
                  message: `Dimension mismatch at dim ${i}: ${s1[i]} vs ${s2[i]} (concat dim=${dim})`,
                  type: "root",
                  sourceNodeId: node.id,
                  receivedShape: s1,
                  expectedShape: s2,
                };
                break;
              }
            }
          }
          const out = [...s1];
          if (dim < out.length) out[dim] = s1[dim] + s2[dim];
          shapes[node.id] = { shape: out, dtype: "float32" };
        }
      } else if (mt === "CrossEntropyLoss" || mt === "NLLLoss") {
        // Validate: if Softmax feeds directly into CrossEntropyLoss, warn (double softmax)
        if (mt === "CrossEntropyLoss") {
          for (const srcId of sources) {
            const srcNode = nodeMap.get(srcId);
            if (srcNode && srcNode.data.moduleType === "Softmax") {
              errors[node.id] = {
                message: `Softmax before CrossEntropyLoss causes double-softmax. Remove the Softmax — CrossEntropyLoss applies it internally.`,
                type: "root",
                sourceNodeId: node.id,
              };
            }
          }
        }
        if (inputShape) shapes[node.id] = { shape: [1], dtype: "float32" };
      } else if (mt === "BCELoss") {
        // Warn if input doesn't come from Sigmoid
        let hasSigmoidUpstream = false;
        for (const srcId of sources) {
          const srcNode = nodeMap.get(srcId);
          if (srcNode && srcNode.data.moduleType === "Sigmoid") hasSigmoidUpstream = true;
        }
        if (!hasSigmoidUpstream && sources.length > 0) {
          errors[node.id] = {
            message: `BCELoss expects input in [0,1] range. Add Sigmoid before BCELoss, or use BCEWithLogitsLoss instead.`,
            type: "root",
            sourceNodeId: node.id,
          };
        }
        if (inputShape) shapes[node.id] = { shape: [1], dtype: "float32" };
      } else if (node.data.codeBlock) {
        // Code block node: pass through first input shape as default
        if (inputShape) shapes[node.id] = { shape: inputShape, dtype: "float32" };
        // For multi-output, store under each handle key too
        const cb = node.data.codeBlock;
        if (cb.outputHandles.length > 1 && inputShape) {
          for (const h of cb.outputHandles) {
            shapes[`${node.id}:${h.id}`] = { shape: inputShape, dtype: "float32" };
          }
        }
      } else {
        // Unknown — pass through
        if (inputShape) shapes[node.id] = { shape: inputShape, dtype: "float32" };
      }
    } catch (e) {
      errors[node.id] = {
        message: e instanceof Error ? e.message : "Shape error",
        type: "root",
        sourceNodeId: node.id,
      };
    }
  }

  return { shapes, errors };
}

function computeConvShape(type: string, input: number[], p: Record<string, unknown>): number[] {
  const outCh = Number(p.out_channels);
  const k = Number(p.kernel_size ?? 3);
  const s = Number(p.stride ?? 1);
  const pad = Number(p.padding ?? 0);
  const d = Number(p.dilation ?? 1);

  if (type === "Conv1d" && input.length === 3) {
    const l = Math.floor((input[2] + 2 * pad - d * (k - 1) - 1) / s + 1);
    return [input[0], outCh, l];
  }
  if (type === "Conv2d" && input.length === 4) {
    const h = Math.floor((input[2] + 2 * pad - d * (k - 1) - 1) / s + 1);
    const w = Math.floor((input[3] + 2 * pad - d * (k - 1) - 1) / s + 1);
    return [input[0], outCh, h, w];
  }
  if (type === "Conv3d" && input.length === 5) {
    const calc = (dim: number) => Math.floor((input[dim] + 2 * pad - d * (k - 1) - 1) / s + 1);
    return [input[0], outCh, calc(2), calc(3), calc(4)];
  }
  return input;
}

function computeConvTransposeShape(input: number[], p: Record<string, unknown>): number[] {
  const outCh = Number(p.out_channels);
  const k = Number(p.kernel_size ?? 3);
  const s = Number(p.stride ?? 1);
  const pad = Number(p.padding ?? 0);
  const opPad = Number(p.output_padding ?? 0);
  if (input.length === 4) {
    const h = (input[2] - 1) * s - 2 * pad + k + opPad;
    const w = (input[3] - 1) * s - 2 * pad + k + opPad;
    return [input[0], outCh, h, w];
  }
  return input;
}

function topoSort(nodes: Node<NodeData>[], edges: Edge[]): Node<NodeData>[] | null {
  const inDeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const n of nodes) { inDeg.set(n.id, 0); adj.set(n.id, []); }
  for (const e of edges) {
    inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1);
    adj.get(e.source)?.push(e.target);
  }
  const queue: string[] = [];
  for (const [id, deg] of inDeg) { if (deg === 0) queue.push(id); }
  const result: Node<NodeData>[] = [];
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  while (queue.length > 0) {
    const id = queue.shift()!;
    const node = nodeMap.get(id);
    if (node) result.push(node);
    for (const nb of adj.get(id) ?? []) {
      const nd = (inDeg.get(nb) ?? 1) - 1;
      inDeg.set(nb, nd);
      if (nd === 0) queue.push(nb);
    }
  }
  return result.length === nodes.length ? result : null;
}

/** Check if two shapes are broadcast-compatible per NumPy rules */
function areBroadcastable(a: number[], b: number[]): boolean {
  const maxLen = Math.max(a.length, b.length);
  for (let i = 0; i < maxLen; i++) {
    const da = i < a.length ? a[a.length - 1 - i] : 1;
    const db = i < b.length ? b[b.length - 1 - i] : 1;
    if (da !== db && da !== 1 && db !== 1) return false;
  }
  return true;
}

/** Compute the broadcasted output shape */
function broadcastShape(a: number[], b: number[]): number[] {
  const maxLen = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < maxLen; i++) {
    const da = i < a.length ? a[a.length - 1 - i] : 1;
    const db = i < b.length ? b[b.length - 1 - i] : 1;
    out.unshift(Math.max(da, db));
  }
  return out;
}

export function formatShape(shape: number[], batchSize?: number): string {
  if (batchSize != null && shape.length >= 2) {
    const display = [batchSize, ...shape.slice(1)];
    return `[${display.join(", ")}]`;
  }
  return `[${shape.join(", ")}]`;
}

/**
 * Get the output shape for a node, optionally for a specific output handle.
 * For multi-output code blocks, shapes are stored as "nodeId:handleId".
 */
export function getOutputShape(shapes: ShapeMap, nodeId: string, handleId?: string): ShapeInfo | undefined {
  if (handleId) {
    const key = `${nodeId}:${handleId}`;
    if (shapes[key]) return shapes[key];
  }
  return shapes[nodeId];
}
