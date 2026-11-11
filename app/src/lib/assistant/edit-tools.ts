/**
 * Tools that change the model. Each one works on the graph as it is now (pending changes included),
 * checks everything against the registry first, and either makes the whole change or none of it.
 * Changes show on the canvas as a preview the user applies or discards; nothing is final until then.
 */

import { useGraphStore } from "../../store/graph-store";
import { useTrainingStore, type TrainingConfig } from "../../store/training-store";
import { shapeErrorFix } from "../shape-fixes";
import { autoLayout } from "../auto-layout";
import { OPTIMIZERS, SCHEDULERS, LOSSES, DEVICES } from "../training-options";
import {
  GraphOpError, addLayer, insertAfter, removeLayers, setParams, renameLayer, connect, disconnect,
  groupLayers, freePosition, requireNode, type GraphDoc,
} from "../graph-ops";
import { currentDoc, writeGraph, writeTraining, pendingSummary, discardChanges, hasPendingChanges } from "./change-set";
import { requireModule } from "./read-tools";
import { ToolError, type AssistantTool } from "./tool-types";

/** A layer's name for the chat's activity list, e.g. "fc2" */
function nameOf(id: string): string {
  return useGraphStore.getState().nodes.find((n) => n.id === id)?.data.label ?? id;
}

/** Runs a graph change; a GraphOpError becomes a tool error and leaves the graph as it was */
function edit<T>(change: (doc: GraphDoc) => { doc: GraphDoc; result?: T }): T | undefined {
  let out: { doc: GraphDoc; result?: T };
  try {
    out = change(currentDoc());
  } catch (e) {
    if (e instanceof GraphOpError) throw new ToolError(e.message);
    throw e;
  }
  writeGraph(out.doc);
  return out.result;
}

/** What every edit tool reports back: the pending changes so far and any shape problems they cause */
function report(extra: Record<string, unknown> = {}) {
  const s = pendingSummary();
  return {
    ...extra,
    pendingChanges: s ? [...s.model, ...s.training.map((t) => `training ${t}`)] : [],
    newShapeProblems: s?.newProblems.length ? s.newProblems : "none",
    note: "These changes are on the canvas as a preview. The user applies or discards them.",
  };
}

/** Resolves an id or a name given to a layer earlier in the same call */
function resolver(refs: Map<string, string>, doc: () => GraphDoc) {
  return (key: string) => {
    const id = refs.get(key) ?? key;
    requireNode(doc(), id);
    return id;
  };
}

interface NewLayerInput {
  ref?: string;
  type: string;
  params?: Record<string, unknown>;
  label?: string;
  after?: string;
  after_output?: string;
}

interface ConnectionInput {
  from: string;
  to: string;
  from_output?: string;
  to_input?: string;
  replace?: boolean;
}

const connectionSchema = {
  type: "object" as const,
  properties: {
    from: { type: "string", description: "Layer id, or a ref from this call" },
    to: { type: "string", description: "Layer id, or a ref from this call" },
    from_output: { type: "string", description: "Output handle; defaults to the first" },
    to_input: { type: "string", description: "Input handle; defaults to the first free one" },
    replace: { type: "boolean", description: "Replace what's connected to that input now" },
  },
  required: ["from", "to"],
  additionalProperties: false,
};

const addLayers: AssistantTool<{ layers: NewLayerInput[]; connections?: ConnectionInput[] }> = {
  kind: "edit",
  definition: {
    name: "add_layers",
    description: "Adds layers to the model. With `after`, a layer is inserted right after that one: it takes over that layer's outgoing connections and the layers below move down, which is how to add e.g. a Dropout between two layers. Without `after`, the layer is placed to the side and you connect it yourself with `connections` (or the connect tool). Give a layer a `ref` to use it in `after` or `connections` within the same call; the result maps refs to the new ids. Settings not given keep their defaults; types and settings are checked against the catalog. Check get_layer_catalog for exact type names, settings and handle names.",
    input_schema: {
      type: "object",
      properties: {
        layers: {
          type: "array",
          items: {
            type: "object",
            properties: {
              ref: { type: "string", description: "A name for this layer within this call" },
              type: { type: "string", description: "Layer type, e.g. \"Dropout\"" },
              params: { type: "object", description: "Settings, e.g. {\"p\": 0.3}" },
              label: { type: "string", description: "The name shown on the canvas" },
              after: { type: "string", description: "Insert after this layer (id or ref)" },
              after_output: { type: "string", description: "Which output of `after` to insert on, for layers with several" },
            },
            required: ["type"],
            additionalProperties: false,
          },
        },
        connections: { type: "array", items: connectionSchema },
      },
      required: ["layers"],
      additionalProperties: false,
    },
  },
  describe: (i) => `Adding ${i.layers.length === 1 ? i.layers[0].type : `${i.layers.length} layers`}`,
  run: async ({ layers, connections = [] }) => {
    for (const l of layers) requireModule(l.type);
    const created = edit((doc) => {
      const refs = new Map<string, string>();
      const ids: string[] = [];
      let d = doc;
      const id = resolver(refs, () => d);
      let side = 0;
      for (const l of layers) {
        const spec = { moduleType: l.type, params: l.params, label: l.label };
        const r = l.after ? insertAfter(d, id(l.after), spec, l.after_output) : addLayer(d, { ...spec, position: freePosition(d, side++) });
        d = r.doc;
        if (l.ref) {
          if (refs.has(l.ref)) throw new GraphOpError(`The ref "${l.ref}" is used twice.`);
          refs.set(l.ref, r.id);
        }
        ids.push(r.id);
      }
      for (const c of connections) d = connect(d, { from: id(c.from), to: id(c.to), fromHandle: c.from_output, toHandle: c.to_input, replace: c.replace });
      return { doc: d, result: { ids, refs: Object.fromEntries(refs) } };
    });
    return report({ created });
  },
};

const removeLayersTool: AssistantTool<{ layer_ids: string[]; reconnect?: boolean }> = {
  kind: "edit",
  definition: {
    name: "remove_layers",
    description: "Removes layers and their connections. With reconnect (the default), a removed layer that had one input is bridged: whatever fed it now feeds what it fed, so removing a Dropout from a chain keeps the chain connected.",
    input_schema: {
      type: "object",
      properties: {
        layer_ids: { type: "array", items: { type: "string" } },
        reconnect: { type: "boolean" },
      },
      required: ["layer_ids"],
      additionalProperties: false,
    },
  },
  describe: (i) => `Removing ${i.layer_ids.length === 1 ? nameOf(i.layer_ids[0]) : `${i.layer_ids.length} layers`}`,
  run: async ({ layer_ids, reconnect = true }) => {
    edit((doc) => ({ doc: removeLayers(doc, layer_ids, { bridge: reconnect }) }));
    return report();
  },
};

const connectTool: AssistantTool<{ connections: ConnectionInput[] }> = {
  kind: "edit",
  definition: {
    name: "connect",
    description: "Connects layers. Handles default to the first output and the first free input; multi-input layers like Add, Concatenate or a loss have named inputs (see get_layer_catalog). Fails if the input is taken, unless replace is true.",
    input_schema: { type: "object", properties: { connections: { type: "array", items: connectionSchema } }, required: ["connections"], additionalProperties: false },
  },
  describe: (i) => (i.connections.length === 1 ? `Connecting ${nameOf(i.connections[0].from)} to ${nameOf(i.connections[0].to)}` : `Making ${i.connections.length} connections`),
  run: async ({ connections }) => {
    edit((doc) => {
      let d = doc;
      for (const c of connections) d = connect(d, { from: c.from, to: c.to, fromHandle: c.from_output, toHandle: c.to_input, replace: c.replace });
      return { doc: d };
    });
    return report();
  },
};

const disconnectTool: AssistantTool<{ connections: { from: string; to: string; to_input?: string }[] }> = {
  kind: "edit",
  definition: {
    name: "disconnect",
    description: "Removes connections between layers. Without to_input, removes every connection from `from` to `to`.",
    input_schema: {
      type: "object",
      properties: {
        connections: {
          type: "array",
          items: { type: "object", properties: { from: { type: "string" }, to: { type: "string" }, to_input: { type: "string" } }, required: ["from", "to"], additionalProperties: false },
        },
      },
      required: ["connections"],
      additionalProperties: false,
    },
  },
  describe: () => "Disconnecting layers",
  run: async ({ connections }) => {
    edit((doc) => {
      let d = doc;
      for (const c of connections) d = disconnect(d, { from: c.from, to: c.to, toHandle: c.to_input });
      return { doc: d };
    });
    return report();
  },
};

const setParamsTool: AssistantTool<{ changes: { layer_id: string; params?: Record<string, unknown>; label?: string }[] }> = {
  kind: "edit",
  definition: {
    name: "set_params",
    description: "Changes layers' settings and/or names. Only the settings given change. Values are checked against each layer's setting types and allowed values.",
    input_schema: {
      type: "object",
      properties: {
        changes: {
          type: "array",
          items: {
            type: "object",
            properties: {
              layer_id: { type: "string" },
              params: { type: "object", description: "e.g. {\"out_channels\": 64}" },
              label: { type: "string" },
            },
            required: ["layer_id"],
            additionalProperties: false,
          },
        },
      },
      required: ["changes"],
      additionalProperties: false,
    },
  },
  describe: (i) => (i.changes.length === 1 ? `Changing ${nameOf(i.changes[0].layer_id)}` : `Changing ${i.changes.length} layers`),
  run: async ({ changes }) => {
    edit((doc) => {
      let d = doc;
      for (const c of changes) {
        if (c.params && Object.keys(c.params).length) d = setParams(d, c.layer_id, c.params);
        if (c.label != null) d = renameLayer(d, c.layer_id, c.label);
      }
      return { doc: d };
    });
    return report();
  },
};

const applyShapeFix: AssistantTool<{ layer_id?: string }> = {
  kind: "edit",
  definition: {
    name: "apply_shape_fix",
    description: "Applies the app's one-parameter fix to a layer's shape error (the suggestedFix get_shape_errors lists). Without layer_id, applies every available fix.",
    input_schema: { type: "object", properties: { layer_id: { type: "string" } }, additionalProperties: false },
  },
  describe: (i) => (i.layer_id ? `Fixing ${nameOf(i.layer_id)}'s shape` : "Fixing shapes"),
  run: async ({ layer_id }) => {
    const { shapeErrors, nodes } = useGraphStore.getState();
    const targets = layer_id ? [layer_id] : Object.keys(shapeErrors);
    const fixes: string[] = [];
    edit((doc) => {
      let d = doc;
      for (const id of targets) {
        const err = shapeErrors[id];
        const node = nodes.find((n) => n.id === id);
        if (!err || !node) { if (layer_id) throw new GraphOpError(`${id} has no shape error.`); continue; }
        const fix = shapeErrorFix(err, node.data.params);
        if (!fix) { if (layer_id) throw new GraphOpError(`The app has no one-click fix for ${id}; change its settings or connections with the other tools.`); continue; }
        d = setParams(d, id, { [fix.param]: fix.to });
        fixes.push(`${id}.${fix.param}: ${fix.from} → ${fix.to}`);
      }
      if (!fixes.length) throw new GraphOpError("There are no shape errors with a one-click fix.");
      return { doc: d };
    });
    return report({ fixed: fixes });
  },
};

const groupLayersTool: AssistantTool<{ layer_ids: string[]; name?: string; collapsed?: boolean }> = {
  kind: "edit",
  definition: {
    name: "group_layers",
    description: "Puts layers in a named group, a frame on the canvas that can be collapsed to one card (e.g. a ResNet stage). Groups only organize the canvas; they never change the model. A layer belongs to one group at most.",
    input_schema: {
      type: "object",
      properties: { layer_ids: { type: "array", items: { type: "string" } }, name: { type: "string" }, collapsed: { type: "boolean" } },
      required: ["layer_ids"],
      additionalProperties: false,
    },
  },
  describe: (i) => `Grouping ${i.name ?? `${i.layer_ids.length} layers`}`,
  run: async ({ layer_ids, name, collapsed = false }) => {
    edit((doc) => {
      const r = groupLayers(doc, layer_ids, name);
      return { doc: { ...r.doc, groups: r.doc.groups.map((g) => (g.id === r.id ? { ...g, collapsed } : g)) } };
    });
    return report();
  },
};

const autoLayoutTool: AssistantTool = {
  kind: "edit",
  definition: {
    name: "auto_layout",
    description: "Re-arranges the whole canvas top to bottom in the order data flows. Use it after adding many layers or building an architecture, not after small edits, since it moves the user's layout.",
    input_schema: { type: "object", properties: {}, additionalProperties: false },
  },
  describe: () => "Tidying the layout",
  run: async () => {
    edit((doc) => ({ doc: { ...doc, nodes: autoLayout(doc.nodes, doc.edges) } }));
    return report();
  },
};

interface TrainingInput {
  optimizer?: string;
  learning_rate?: number;
  weight_decay?: number;
  momentum?: number;
  scheduler?: string;
  scheduler_params?: Record<string, unknown>;
  loss?: string;
  epochs?: number;
  batch_size?: number;
  validation_split?: number;
  device?: string;
  mixed_precision?: boolean;
  gradient_clipping?: number | null;
  early_stopping_patience?: number | null;
  seed?: number | null;
}

const setTrainingConfig: AssistantTool<TrainingInput> = {
  kind: "edit",
  definition: {
    name: "set_training_config",
    description: `Changes training settings. Only the fields given change. optimizer: ${OPTIMIZERS.join(", ")}. scheduler: ${SCHEDULERS.join(", ")}. loss: ${LOSSES.join(", ")}. device: ${DEVICES.join(", ")}. gradient_clipping is the max norm, or null to turn it off; early_stopping_patience is in epochs, or null to turn it off.`,
    input_schema: {
      type: "object",
      properties: {
        optimizer: { type: "string", enum: OPTIMIZERS },
        learning_rate: { type: "number" },
        weight_decay: { type: "number" },
        momentum: { type: "number" },
        scheduler: { type: "string", enum: SCHEDULERS },
        scheduler_params: { type: "object" },
        loss: { type: "string", enum: LOSSES },
        epochs: { type: "integer" },
        batch_size: { type: "integer" },
        validation_split: { type: "number" },
        device: { type: "string", enum: DEVICES },
        mixed_precision: { type: "boolean" },
        gradient_clipping: { type: ["number", "null"] },
        early_stopping_patience: { type: ["integer", "null"] },
        seed: { type: ["integer", "null"] },
      },
      additionalProperties: false,
    },
  },
  describe: () => "Changing training settings",
  run: async (i) => {
    const c = useTrainingStore.getState().config;
    const range = (v: number | undefined, name: string, min: number, max = Infinity) => {
      if (v != null && !(v >= min && v <= max)) throw new ToolError(`${name} must be between ${min} and ${max}.`);
    };
    range(i.learning_rate, "learning_rate", 1e-8, 10);
    range(i.weight_decay, "weight_decay", 0, 1);
    range(i.momentum, "momentum", 0, 1);
    range(i.epochs, "epochs", 1, 10000);
    range(i.batch_size, "batch_size", 1, 65536);
    range(i.validation_split, "validation_split", 0, 0.9);
    const patch: Partial<TrainingConfig> = {};
    if (i.optimizer || i.learning_rate != null || i.weight_decay != null || i.momentum != null) {
      patch.optimizer = {
        ...c.optimizer,
        ...(i.optimizer ? { type: i.optimizer } : {}),
        ...(i.learning_rate != null ? { lr: i.learning_rate } : {}),
        ...(i.weight_decay != null ? { weightDecay: i.weight_decay } : {}),
        ...(i.momentum != null ? { momentum: i.momentum } : {}),
      };
    }
    if (i.scheduler) patch.scheduler = i.scheduler === "None" ? null : { type: i.scheduler, params: i.scheduler_params ?? {} };
    else if (i.scheduler_params && c.scheduler) patch.scheduler = { ...c.scheduler, params: i.scheduler_params };
    if (i.loss) patch.loss = i.loss;
    if (i.epochs != null) patch.epochs = i.epochs;
    if (i.batch_size != null) patch.batchSize = i.batch_size;
    if (i.validation_split != null) patch.valSplit = i.validation_split;
    if (i.device) patch.device = i.device;
    if (i.mixed_precision != null) patch.mixedPrecision = i.mixed_precision;
    if (i.gradient_clipping !== undefined) patch.gradientClipping = i.gradient_clipping == null ? { ...c.gradientClipping, enabled: false } : { enabled: true, maxNorm: i.gradient_clipping };
    if (i.early_stopping_patience !== undefined) patch.earlyStopping = i.early_stopping_patience == null ? { ...c.earlyStopping, enabled: false } : { ...c.earlyStopping, enabled: true, patience: i.early_stopping_patience };
    if (i.seed !== undefined) patch.seed = i.seed;
    if (!Object.keys(patch).length) throw new ToolError("No settings were given.");
    writeTraining(patch);
    return report();
  },
};

const discardTool: AssistantTool = {
  kind: "edit",
  definition: {
    name: "discard_changes",
    description: "Throws away all your pending changes and puts the model and training settings back as they were before them. Use it to start over when an approach isn't working.",
    input_schema: { type: "object", properties: {}, additionalProperties: false },
  },
  describe: () => "Starting over",
  run: async () => {
    if (!hasPendingChanges()) return { note: "There were no pending changes." };
    discardChanges();
    return { note: "Pending changes discarded. The model is back as it was." };
  },
};

export const EDIT_TOOLS: AssistantTool[] = [
  addLayers, removeLayersTool, connectTool, disconnectTool, setParamsTool, applyShapeFix,
  groupLayersTool, autoLayoutTool, setTrainingConfig, discardTool,
] as AssistantTool[];
