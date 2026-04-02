/**
 * Pre-flight validation before training.
 * Checks shape compatibility, dataset-model mismatches, and common errors.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { TrainingConfig } from "../store/training-store";
import { inferShapesClient } from "./shape-inference";
import { DATASET_METADATA, isDatasetNode, resolveDataNodeShape } from "./dataset-metadata";
import type { DataOutputDef } from "./types";

interface PreflightResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  autoFixes: Array<{ nodeId: string; param: string; oldValue: unknown; newValue: unknown; reason: string }>;
}

// Derived from DATASET_METADATA for backward-compatible shape/class lookups
const DATASET_INPUT_SHAPES: Record<string, { shape: number[]; numClasses: number }> = Object.fromEntries(
  Object.entries(DATASET_METADATA).map(([k, v]) => [k, { shape: v.inputShape, numClasses: v.numClasses }]),
);

export function preflightCheck(
  nodes: Node<NodeData>[],
  edges: Edge[],
  trainingConfig: TrainingConfig,
): PreflightResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const autoFixes: PreflightResult["autoFixes"] = [];

  if (nodes.length === 0) {
    errors.push("No nodes in graph");
    return { valid: false, errors, warnings, autoFixes };
  }

  // Check Input/Output nodes exist (Dataset nodes count as entry points)
  const inputNodes = nodes.filter((n) => n.data.moduleType === "Input");
  const datasetNodes = nodes.filter((n) => isDatasetNode(n.data.moduleType));
  const outputNodes = nodes.filter((n) => n.data.moduleType === "Output");

  const LOSS_TYPES = new Set([
    "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "BCELoss",
    "L1Loss", "HuberLoss", "NLLLoss", "KLDivLoss",
    "TripletMarginLoss", "CosineEmbeddingLoss", "CTCLoss", "SmoothL1Loss",
  ]);
  const lossNodes = nodes.filter((n) => LOSS_TYPES.has(n.data.moduleType));

  if (inputNodes.length === 0 && datasetNodes.length === 0) errors.push("No Input or Dataset node found");
  if (outputNodes.length === 0 && lossNodes.length === 0) errors.push("No Output or Loss node found");

  // Check dataset vs model input shape — prefer Dataset node over training config
  // Handle universal Data node: resolve shape from params
  let dsInfo: { shape: number[]; numClasses: number } | undefined;
  if (datasetNodes.length > 0) {
    const dsNode = datasetNodes[0];
    if (dsNode.data.moduleType === "Data") {
      const resolved = resolveDataNodeShape(dsNode.data.params);
      if (resolved) {
        dsInfo = { shape: resolved.inputShape, numClasses: resolved.numClasses };
      }
    } else {
      dsInfo = DATASET_INPUT_SHAPES[dsNode.data.moduleType];
    }
  }
  if (!dsInfo) {
    dsInfo = DATASET_INPUT_SHAPES[trainingConfig.dataset.type];
  }
  const effectiveDatasetType = datasetNodes.length > 0
    ? (datasetNodes[0].data.moduleType === "Data"
      ? String(datasetNodes[0].data.params.source ?? "")
      : datasetNodes[0].data.moduleType)
    : trainingConfig.dataset.type;
  if (dsInfo && inputNodes.length > 0) {
    const inputNode = inputNodes[0];
    const currentShapeStr = String(inputNode.data.params.shape ?? "");
    let currentShape: number[];
    try {
      currentShape = JSON.parse(currentShapeStr.replace(/\(/g, "[").replace(/\)/g, "]"));
    } catch {
      currentShape = [];
    }

    // Check if shapes are compatible
    const expectedShape = dsInfo.shape;
    const shapeMatches = JSON.stringify(currentShape) === JSON.stringify(expectedShape);
    const flatExpected = expectedShape.slice(1).reduce((a, b) => a * b, 1);
    const flatCurrent = currentShape.slice(1).reduce((a, b) => a * b, 1);

    if (!shapeMatches && flatCurrent !== flatExpected) {
      autoFixes.push({
        nodeId: inputNode.id,
        param: "shape",
        oldValue: currentShapeStr,
        newValue: JSON.stringify(expectedShape),
        reason: `Dataset ${effectiveDatasetType} expects input shape ${JSON.stringify(expectedShape)}, but Input node has ${currentShapeStr}`,
      });
    }
  }

  // Check last Linear output matches target — works with or without explicit Input node
  if (dsInfo) {
    const lastLinear = findLastLinear(nodes, edges, lossNodes);
    if (lastLinear) {
      const outFeatures = Number(lastLinear.data.params.out_features ?? 0);

      // Determine expected output size from loss type + target shape
      let expectedOut = dsInfo.numClasses;

      // For custom tensor data (scan result), infer from target tensor shape
      if (expectedOut === 0 && datasetNodes.length > 0) {
        const dsNode = datasetNodes[0];
        const scan = dsNode.data.params._scan as { tensors?: { name: string; shape: number[]; num_classes?: number }[] } | undefined;
        const dataOutputs = dsNode.data.dataOutputs as (DataOutputDef & { column?: string })[] | undefined;
        const yDef = dataOutputs?.find((o) => o.id === "y");
        if (scan?.tensors && yDef) {
          // Use column mapping (tensor name), fall back to handle id
          const key = yDef.column || yDef.id;
          const targetTensor = scan.tensors.find((t) => t.name === key);
          if (targetTensor) {
            const targetSampleShape = targetTensor.shape.slice(1);
            const lossType = lossNodes.length > 0 ? lossNodes[0].data.moduleType : trainingConfig.loss;
            if (["MSELoss", "L1Loss", "HuberLoss"].includes(lossType)) {
              expectedOut = targetSampleShape.length === 0 ? 1 : targetSampleShape.reduce((a, b) => a * b, 1);
            } else if (["CrossEntropyLoss", "NLLLoss"].includes(lossType)) {
              // Use num_classes from scan stats if available
              expectedOut = targetTensor.num_classes ?? 0;
            }
          }
        }
      }

      if (expectedOut > 0 && outFeatures !== expectedOut && outFeatures > 0) {
        autoFixes.push({
          nodeId: lastLinear.id,
          param: "out_features",
          oldValue: outFeatures,
          newValue: expectedOut,
          reason: `Final Linear has out_features=${outFeatures} but target expects ${expectedOut}`,
        });
      }
    }
  }

  // Check for sample count mismatch in scanned TensorFile data
  for (const dsNode of datasetNodes) {
    const scan = dsNode.data.params._scan as { tensors?: { name: string; shape: number[] }[] } | undefined;
    if (scan?.tensors && scan.tensors.length >= 2) {
      // Check ALL tensors in the scan for consistent sample counts
      const first = scan.tensors[0];
      const mismatch = scan.tensors.find((t) => t.shape[0] !== first.shape[0]);
      if (mismatch) {
        errors.push(
          `Sample count mismatch in data: "${first.name}" has ${first.shape[0]} samples but "${mismatch.name}" has ${mismatch.shape[0]}. All tensors must have the same number of samples.`
        );
      }
    }
  }

  // Code block validation
  for (const node of nodes) {
    if (!node.data.codeBlock) continue;
    const cb = node.data.codeBlock;
    if (!cb.forwardBody.trim()) {
      warnings.push(`Code block "${node.data.label}" has empty forward body`);
    }
    const pyIdRegex = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
    for (const h of [...cb.inputHandles, ...cb.outputHandles]) {
      if (!pyIdRegex.test(h.id)) {
        warnings.push(`Code block "${node.data.label}": handle "${h.id}" is not a valid Python identifier`);
      }
    }
    // Check return count vs output handles
    const returnCount = (cb.forwardBody.match(/\breturn\b/g) || []).length;
    if (returnCount > 0 && cb.outputHandles.length > 1) {
      // Check if return has a tuple (comma-separated)
      const returnMatch = cb.forwardBody.match(/return\s+(.+)/);
      if (returnMatch) {
        const returnExpr = returnMatch[1];
        const commaCount = returnExpr.split(",").length;
        if (commaCount !== cb.outputHandles.length) {
          warnings.push(`Code block "${node.data.label}": returns ${commaCount} values but has ${cb.outputHandles.length} output handles`);
        }
      }
    }
  }

  // ── Shape-inference-driven auto-fixes ──
  // Apply Input shape + out_features fixes tentatively so shape inference
  // propagates correct dimensions through the rest of the graph.
  const tentativeNodes = applyTentativeFixes(nodes, autoFixes);
  const { shapes: inferredShapes, errors: shapeErrors } = inferShapesClient(tentativeNodes, edges);

  // Auto-fixable param hints from shape inference.
  // Maps paramHint → how to extract the correct value from receivedShape.
  const FIXABLE_PARAMS: Record<string, (received: number[]) => number> = {
    in_features: (s) => s[s.length - 1],      // last dim
    in_channels: (s) => s[1],                   // channel dim (N, C, ...)
    num_features: (s) => s[1],                  // channel/feature dim for BatchNorm
  };
  const alreadyFixed = new Set(autoFixes.map((f) => `${f.nodeId}:${f.param}`));

  for (const [nodeId, shapeErr] of Object.entries(shapeErrors)) {
    if (shapeErr.type === "cascaded") continue;
    if (!shapeErr.paramHint || !shapeErr.receivedShape) continue;

    const extractor = FIXABLE_PARAMS[shapeErr.paramHint];
    if (!extractor) continue;
    if (alreadyFixed.has(`${nodeId}:${shapeErr.paramHint}`)) continue;

    const node = nodes.find((n) => n.id === nodeId);
    if (!node) continue;

    const correctValue = extractor(shapeErr.receivedShape);
    const currentValue = Number(node.data.params[shapeErr.paramHint] ?? 0);
    if (correctValue > 0 && currentValue > 0 && correctValue !== currentValue) {
      autoFixes.push({
        nodeId, param: shapeErr.paramHint,
        oldValue: currentValue, newValue: correctValue,
        reason: `${node.data.moduleType} "${node.data.label}" has ${shapeErr.paramHint}=${currentValue} but receives ${correctValue}`,
      });
      alreadyFixed.add(`${nodeId}:${shapeErr.paramHint}`);
    }
  }

  // Report remaining non-auto-fixable shape errors
  const autoFixedNodeIds = new Set(autoFixes.map((f) => f.nodeId));
  for (const [nodeId, err] of Object.entries(shapeErrors)) {
    if (err.type === "cascaded") continue;
    if (autoFixedNodeIds.has(nodeId)) continue;
    const node = nodes.find((n) => n.id === nodeId);
    errors.push(`Shape error at "${node?.data.label ?? nodeId}": ${err.message}`);
  }

  // ── Output vs target shape check ──
  // For any loss, check if the model output shape is compatible with the target
  for (const lossNode of lossNodes) {
    const predEdge = edges.find((e) => e.target === lossNode.id && (e.targetHandle === "input" || e.targetHandle === "in"));
    if (!predEdge) continue;
    const outputShape = inferredShapes[predEdge.source]?.shape;
    if (!outputShape || outputShape.length < 2) continue;
    const outSample = outputShape.slice(1); // per-sample output shape

    // Determine target shape from the Data node's y-handle tensor (scan result)
    let targetSample: number[] | null = null;
    let targetIsClassLabel = false;
    for (const dsNode of datasetNodes) {
      const scan = dsNode.data.params._scan as { tensors?: { name: string; shape: number[]; num_classes?: number }[] } | undefined;
      const yOutput = (dsNode.data.dataOutputs ?? []).find((o: DataOutputDef) => o.id === "y");
      if (scan?.tensors && yOutput) {
        const key = (yOutput as DataOutputDef & { column?: string }).column || yOutput.id;
        const tgt = scan.tensors.find((t) => t.name === key);
        if (tgt) {
          targetSample = tgt.shape.slice(1); // per-sample shape
          targetIsClassLabel = (tgt.num_classes ?? 0) > 0;
        }
      }
    }

    const lossMt = lossNode.data.moduleType;
    const isRegressionLoss = ["MSELoss", "L1Loss", "HuberLoss", "SmoothL1Loss"].includes(lossMt);
    const isClassificationLoss = ["CrossEntropyLoss", "NLLLoss"].includes(lossMt);

    if (targetIsClassLabel && isRegressionLoss) {
      warnings.push(
        `${lossMt} is a regression loss, but target looks like class labels. Consider using CrossEntropyLoss instead.`
      );
    }
    if (targetSample && isRegressionLoss) {
      const outFlat = outSample.reduce((a: number, b: number) => a * b, 1);
      const tgtFlat = targetSample.length === 0 ? 1 : targetSample.reduce((a: number, b: number) => a * b, 1);
      if (outFlat !== tgtFlat) {
        // Auto-fix: adjust last Linear's out_features to match target
        const lastLinear = findLastLinear(nodes, edges, lossNodes);
        if (lastLinear && !alreadyFixed.has(`${lastLinear.id}:out_features`)) {
          autoFixes.push({
            nodeId: lastLinear.id, param: "out_features",
            oldValue: Number(lastLinear.data.params.out_features ?? 0), newValue: tgtFlat,
            reason: `${lossMt} target has ${tgtFlat} values but model outputs ${outFlat}`,
          });
        } else {
          errors.push(
            `${lossMt}: model outputs [${outSample.join(",")}] (${outFlat} values) but target has [${targetSample.join(",") || "1"}] (${tgtFlat} values). Shapes must match.`
          );
        }
      }
    }
    if (targetIsClassLabel && isClassificationLoss && outSample.length > 0) {
      const numClasses = outSample[outSample.length - 1];
      const dsNode = datasetNodes[0];
      const scan = dsNode?.data.params._scan as { tensors?: { num_classes?: number; name: string }[] } | undefined;
      const yOutput = (dsNode?.data.dataOutputs ?? []).find((o: DataOutputDef) => o.id === "y");
      const key = (yOutput as DataOutputDef & { column?: string })?.column || yOutput?.id;
      const tgt = scan?.tensors?.find((t) => t.name === key);
      const actualClasses = tgt?.num_classes ?? 0;
      if (actualClasses > 0 && numClasses > 0 && numClasses !== actualClasses) {
        warnings.push(
          `${lossMt}: model outputs ${numClasses} classes but target has ${actualClasses} classes. Adjust final layer's out_features.`
        );
      }
    }
  }

  return { valid: errors.length === 0, errors, warnings, autoFixes };
}

/** Apply auto-fixes to a copy of the nodes so shape inference sees corrected values. */
function applyTentativeFixes(
  nodes: Node<NodeData>[],
  fixes: PreflightResult["autoFixes"],
): Node<NodeData>[] {
  if (fixes.length === 0) return nodes;
  const fixesByNode = new Map<string, Record<string, unknown>>();
  for (const fix of fixes) {
    const map = fixesByNode.get(fix.nodeId) ?? {};
    map[fix.param] = fix.newValue;
    fixesByNode.set(fix.nodeId, map);
  }
  return nodes.map((n) => {
    const overrides = fixesByNode.get(n.id);
    if (!overrides) return n;
    return { ...n, data: { ...n.data, params: { ...n.data.params, ...overrides } } };
  });
}

function findLastLinear(nodes: Node<NodeData>[], edges: Edge[], lossNodes: Node<NodeData>[] = []): Node<NodeData> | null {
  // Find the Linear node closest to the Output or Loss node
  const outgoing = new Map<string, string[]>();
  for (const e of edges) {
    const list = outgoing.get(e.source) ?? [];
    list.push(e.target);
    outgoing.set(e.source, list);
  }

  const terminalIds = new Set([
    ...nodes.filter((n) => n.data.moduleType === "Output").map((n) => n.id),
    ...lossNodes.map((n) => n.id),
  ]);
  for (const node of nodes) {
    if (node.data.moduleType !== "Linear") continue;
    const targets = outgoing.get(node.id) ?? [];
    if (targets.some((t) => terminalIds.has(t))) return node;
  }
  return null;
}
