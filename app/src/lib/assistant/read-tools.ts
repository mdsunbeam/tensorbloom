/**
 * Tools that only look: the graph, its shape problems, the layer catalog, the generated code, runs,
 * predictions and dataset statistics. None of them changes anything.
 */

import { useGraphStore } from "../../store/graph-store";
import { useRunStore, runWeights, type RunRecord } from "../../store/run-store";
import { useTrainingStore } from "../../store/training-store";
import { MODULE_REGISTRY, CATEGORY_LABELS } from "../registry";
import { paramHelp, torchName, pytorchDocsUrl } from "../param-help";
import { decompileGraph } from "../decompiler";
import { sidecarCall } from "../sidecar";
import { buildSidecarGraph } from "../sidecar-graph";
import { sidecarTrainConfig } from "../training";
import { isDatasetNode, extractDatasetConfig } from "../dataset-metadata";
import { graphJson, shapeProblems, describeRun, describeTraining, configDiff, graphDiff, round, type GraphView } from "./describe";
import { ToolError, closestNames, type AssistantTool } from "./tool-types";

export function currentView(): GraphView {
  const { nodes, edges, groups, shapeMap, shapeErrors } = useGraphStore.getState();
  return { nodes, edges, groups, shapeMap, shapeErrors };
}

function findRun(number: number): RunRecord {
  const runs = useRunStore.getState().runs;
  const run = runs.find((r) => r.number === number);
  if (!run) {
    throw new ToolError(runs.length
      ? `There is no run ${number}. Runs: ${runs.map((r) => r.number).join(", ")}.`
      : "This project has no runs yet.");
  }
  return run;
}

/** Looks a layer type up in the registry, or explains what exists instead */
export function requireModule(type: string) {
  const def = MODULE_REGISTRY[type];
  if (def) return def;
  const near = closestNames(type, Object.keys(MODULE_REGISTRY));
  throw new ToolError(`"${type}" isn't a layer TensorBloom has. Closest: ${near.join(", ")}. Use get_layer_catalog to see them all.`);
}

const noInput = { type: "object" as const, properties: {}, additionalProperties: false };

const getGraph: AssistantTool = {
  kind: "read",
  definition: {
    name: "get_graph",
    description: "Returns the model on the canvas as JSON: every layer with its id, type, label and parameters, its output shape (batch size 1), connections between layers and their handles, groups, and any custom code. Use it when the summary in <app_state> isn't detailed enough, e.g. for handle names or custom code.",
    input_schema: noInput,
  },
  describe: () => "Reading the graph",
  run: async () => graphJson(currentView()),
};

const getShapeErrors: AssistantTool = {
  kind: "read",
  definition: {
    name: "get_shape_errors",
    description: "Returns the shape problems in the model. Each has the layer, a message, the shape it receives and the shape it expects, and whether it's the cause or follows from an earlier error. When the app knows a one-parameter fix it's included as suggestedFix. Fix causes first: the others usually go away with them.",
    input_schema: noInput,
  },
  describe: () => "Checking shapes",
  run: async () => {
    const problems = shapeProblems(currentView());
    return problems.length ? { problems } : { problems: [], note: "No shape problems." };
  },
};

const getLayerCatalog: AssistantTool<{ module_type?: string; search?: string }> = {
  kind: "read",
  definition: {
    name: "get_layer_catalog",
    description: "Lists the layers TensorBloom has. With module_type, returns that layer's parameters (type, default, allowed values, what each one means), its input and output handles, and its PyTorch name. With search, lists layers whose name or description matches. Use the exact type names it returns.",
    input_schema: {
      type: "object",
      properties: {
        module_type: { type: "string", description: "A layer type, e.g. \"Conv2d\"" },
        search: { type: "string", description: "Words to match against layer names and descriptions" },
      },
      additionalProperties: false,
    },
  },
  describe: (i) => (i.module_type ? `Looking up ${i.module_type}` : "Reading the layer catalog"),
  run: async ({ module_type, search }) => {
    if (module_type) {
      const def = requireModule(module_type);
      return {
        type: def.type,
        label: def.label,
        category: CATEGORY_LABELS[def.category],
        description: def.description,
        pytorch: torchName(def.type),
        docs: pytorchDocsUrl(def.type),
        params: def.params.map((p) => ({
          name: p.name,
          type: p.type,
          default: p.default,
          ...(p.options ? { options: p.options } : {}),
          ...(p.min != null ? { min: p.min } : {}),
          ...(p.max != null ? { max: p.max } : {}),
          ...(paramHelp(def.type, p.name) ? { help: paramHelp(def.type, p.name) } : {}),
        })),
        inputs: def.inputHandles.map((h) => h.id),
        outputs: def.outputHandles.map((h) => h.id),
      };
    }
    const words = (search ?? "").toLowerCase().split(/\s+/).filter(Boolean);
    const defs = Object.values(MODULE_REGISTRY).filter((d) => {
      if (!words.length) return true;
      const text = `${d.type} ${d.label} ${d.description} ${d.category}`.toLowerCase();
      return words.every((w) => text.includes(w));
    });
    const byCategory: Record<string, string[]> = {};
    for (const d of defs) (byCategory[CATEGORY_LABELS[d.category]] ??= []).push(`${d.type}: ${d.description}`);
    return defs.length ? byCategory : { note: `No layer matches "${search}".` };
  },
};

const getCode: AssistantTool = {
  kind: "read",
  definition: {
    name: "get_code",
    description: "Returns the PyTorch nn.Module the app generates from the graph, exactly as shown in the Code tab, plus any problems generating it.",
    input_schema: noInput,
  },
  describe: () => "Reading the generated code",
  run: async () => {
    const { nodes, edges } = useGraphStore.getState();
    const { code, errors } = decompileGraph(nodes, edges);
    return { code: code || "(empty)", ...(errors.length ? { problems: errors } : {}) };
  },
};

const listRuns: AssistantTool = {
  kind: "read",
  definition: {
    name: "list_runs",
    description: "Lists the project's training runs, oldest first, with status, epochs, duration, parameter count and best metrics, and whether each run trained the model that's on the canvas now.",
    input_schema: noInput,
  },
  describe: () => "Looking at runs",
  run: async () => {
    const { runs } = useRunStore.getState();
    if (!runs.length) return { runs: [], note: "No runs yet." };
    const { nodes, edges } = useGraphStore.getState();
    return {
      runs: runs.map((r) => ({
        number: r.number,
        summary: describeRun(r),
        sameModelAsCanvas: graphDiff(r.graph, { nodes, edges }).length === 0,
      })),
    };
  },
};

const getRun: AssistantTool<{ run_number: number; compare_to?: number }> = {
  kind: "read",
  definition: {
    name: "get_run",
    description: "Returns one run in detail: its training settings, loss and accuracy per epoch, results, error if it failed, and how its model differs from the canvas. With compare_to, also lists the settings and layers that differ between the two runs.",
    input_schema: {
      type: "object",
      properties: {
        run_number: { type: "integer", description: "The run's number, as shown in the Runs tab" },
        compare_to: { type: "integer", description: "Another run's number to compare with" },
      },
      required: ["run_number"],
      additionalProperties: false,
    },
  },
  describe: (i) => (i.compare_to != null ? `Comparing runs ${i.run_number} and ${i.compare_to}` : `Reading run ${i.run_number}`),
  run: async ({ run_number, compare_to }) => {
    const run = findRun(run_number);
    const { nodes, edges } = useGraphStore.getState();
    const out: Record<string, unknown> = {
      number: run.number,
      status: run.status,
      startedAt: run.startedAt,
      durationSeconds: Math.round(run.durationMs / 1000),
      ...(run.error ? { error: run.error } : {}),
      settings: describeTraining(run.config),
      results: run.results ? Object.fromEntries(Object.entries(run.results).filter(([k]) => k !== "environment").map(([k, v]) => [k, typeof v === "number" ? round(v) : v])) : null,
      epochs: run.epochs.map((e) => ({
        epoch: e.epoch,
        trainLoss: round(e.trainLoss),
        ...(e.valLoss != null ? { valLoss: round(e.valLoss) } : {}),
        ...(e.trainAcc != null ? { trainAcc: round(e.trainAcc) } : {}),
        ...(e.valAcc != null ? { valAcc: round(e.valAcc) } : {}),
        lr: e.lr,
      })),
      parameters: run.model.params,
      modelChangesSinceRun: graphDiff(run.graph, { nodes, edges }),
    };
    if (compare_to != null) {
      const other = findRun(compare_to);
      out.comparedTo = {
        number: other.number,
        summary: describeRun(other),
        settingChanges: configDiff(run.config, other.config),
        modelChanges: graphDiff(run.graph, other.graph),
      };
    }
    return out;
  },
};

interface PredictSample {
  index: number | null;
  target: { kind: "class"; index: number } | { kind: string; values?: number[]; shape?: number[] } | null;
  output: { kind: "classes"; top: { index: number; prob: number }[] } | { kind: string; values?: number[]; shape?: number[] };
  correct: boolean | null;
}

const getPredictions: AssistantTool<{ run_number: number; mistakes_only?: boolean; count?: number }> = {
  kind: "read",
  definition: {
    name: "get_predictions",
    description: "Runs a trained run's model on validation samples it didn't train on and returns, for each sample, the true class, the model's top guesses with probabilities, and whether it was right. With mistakes_only, returns only wrong predictions, which shows what the model confuses. Inputs (images, text) are not included. Needs Python and the run's weights, and doesn't work while training runs.",
    input_schema: {
      type: "object",
      properties: {
        run_number: { type: "integer" },
        mistakes_only: { type: "boolean" },
        count: { type: "integer", description: "How many samples, 1 to 48. Default 24." },
      },
      required: ["run_number"],
      additionalProperties: false,
    },
  },
  describe: (i) => `Trying run ${i.run_number}'s model`,
  run: async ({ run_number, mistakes_only = false, count = 24 }) => {
    const run = findRun(run_number);
    const status = useTrainingStore.getState().status;
    if (status === "running" || status === "starting") throw new ToolError("Training is running. Predictions can be checked once it finishes.");
    const weights = runWeights(run, useRunStore.getState().liveRunId);
    if (!weights) throw new ToolError(`Run ${run_number}'s weights weren't kept (only the latest run and runs with a checkpoint have them).`);
    const res = await sidecarCall<{ split: string; dataset_size: number; class_names: string[] | null; task: string; samples: PredictSample[]; scanned?: number; scanned_correct?: number }>("model.predict", {
      graph: buildSidecarGraph(run.graph.nodes, run.graph.edges),
      config: sidecarTrainConfig(run.graph.nodes, run.config),
      weights: weights.weights,
      checkpoint_path: weights.weights === "checkpoint" ? weights.path : null,
      count: Math.max(1, Math.min(48, count)),
      mistakes_only,
      sample_seed: 1,
    });
    const name = (i: number) => res.class_names?.[i] ?? String(i);
    const short = (v?: number[]) => (v ? v.slice(0, 8).map(round) : undefined);
    const samples = res.samples.map((s) => ({
      sample: s.index,
      ...(s.target?.kind === "class" ? { truth: name((s.target as { index: number }).index) } : s.target ? { truth: short((s.target as { values?: number[] }).values) } : {}),
      ...(s.output.kind === "classes"
        ? { guesses: (s.output as { top: { index: number; prob: number }[] }).top.slice(0, 3).map((t) => `${name(t.index)} ${(t.prob * 100).toFixed(0)}%`) }
        : { output: short((s.output as { values?: number[] }).values), outputShape: (s.output as { shape?: number[] }).shape }),
      ...(s.correct != null ? { correct: s.correct } : {}),
    }));
    const right = samples.filter((s) => s.correct).length;
    return {
      split: res.split,
      datasetSize: res.dataset_size,
      task: res.task,
      weights: weights.weights === "trained" ? "end of run" : `checkpoint from epoch ${weights.epoch}`,
      ...(res.scanned != null ? { scanned: res.scanned, scannedCorrect: res.scanned_correct } : {}),
      ...(samples.some((s) => s.correct != null) && !mistakes_only ? { correctInSample: `${right} of ${samples.length}` } : {}),
      samples,
    };
  },
};

const previewData: AssistantTool<{ node_id?: string }> = {
  kind: "read",
  definition: {
    name: "preview_data",
    description: "Loads the dataset a Data layer points at and returns its statistics: training and validation sizes, the shape and dtype of one sample, the number of classes and class names, and the mean, std, min and max of the inputs. No samples are returned. Needs Python, and may download the dataset the first time.",
    input_schema: {
      type: "object",
      properties: { node_id: { type: "string", description: "The Data layer's id. Defaults to the first one." } },
      additionalProperties: false,
    },
  },
  describe: () => "Reading dataset statistics",
  run: async ({ node_id }) => {
    const { nodes } = useGraphStore.getState();
    const node = node_id ? nodes.find((n) => n.id === node_id) : nodes.find((n) => isDatasetNode(n.data.moduleType));
    if (!node) throw new ToolError(node_id ? `There is no layer ${node_id}.` : "The model has no Data layer.");
    if (!isDatasetNode(node.data.moduleType)) throw new ToolError(`${node.id} is a ${node.data.moduleType}, not a Data layer.`);
    const { dataset, valSplit, datasetExtra } = extractDatasetConfig(node.data.moduleType, node.data.params);
    const res = await sidecarCall<Record<string, unknown>>("data.preview", { config: { ...dataset, val_split: valSplit, ...datasetExtra }, batch_size: 32 });
    if (res.error) throw new ToolError(String(res.error));
    const introspected = node.data.params._introspected as { label_values?: string[] } | undefined;
    const { train_size, val_size, sample_shape, sample_dtype, num_classes, feature_stats } = res;
    return {
      layer: node.id,
      source: node.data.params.source ?? node.data.moduleType,
      trainSize: train_size,
      validationSize: val_size,
      sampleShape: sample_shape,
      sampleDtype: sample_dtype,
      classes: num_classes,
      ...(introspected?.label_values?.length ? { classNames: introspected.label_values.slice(0, 100) } : {}),
      inputStats: feature_stats,
    };
  },
};

export const READ_TOOLS: AssistantTool[] = [getGraph, getShapeErrors, getLayerCatalog, getCode, listRuns, getRun, getPredictions, previewData] as AssistantTool[];
