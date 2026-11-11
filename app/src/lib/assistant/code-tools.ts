/**
 * Tools for whole architectures, custom code and training.
 *
 * replace_graph builds a new model as a preview, like the edit tools. The others run code on the
 * user's computer (Python with full access to their files), so each call shows exactly what will run
 * and waits for the user to approve it.
 */

import { useGraphStore } from "../../store/graph-store";
import { useTrainingStore } from "../../store/training-store";
import { TEMPLATES, templateGraph } from "../templates";
import { autoLayout } from "../auto-layout";
import { sidecarCall } from "../sidecar";
import { startTraining, stopTraining } from "../training";
import { compiledToGraph, type CompiledGraph } from "../compiled-graph";
import { safePythonId, type CodeBlockDef, type HandleDef } from "../types";
import { GraphOpError, addLayer, connect, createNode, insertAfter, freePosition, requireNode, type GraphDoc } from "../graph-ops";
import { currentDoc, writeGraph, writeTraining, pendingSummary, hasPendingChanges } from "./change-set";
import { requireModule } from "./read-tools";
import { refuseDuringLesson } from "./lesson-tools";
import { ToolError, type AssistantTool } from "./tool-types";

const RUNS_CODE = "This runs Python on your computer with the same access to your files as TensorBloom. Approve only code you've read and trust.";

function report(extra: Record<string, unknown> = {}) {
  const s = pendingSummary();
  return {
    ...extra,
    pendingChanges: s ? [...s.model, ...s.training.map((t) => `training ${t}`)] : [],
    newShapeProblems: s?.newProblems.length ? s.newProblems : "none",
    note: "These changes are on the canvas as a preview. The user applies or discards them.",
  };
}

function wrap<T>(fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    if (e instanceof GraphOpError) throw new ToolError(e.message);
    throw e;
  }
}

// ---------------------------------------------------------------------------
// replace_graph
// ---------------------------------------------------------------------------

interface LayerSpec { ref: string; type: string; params?: Record<string, unknown>; label?: string }
interface LinkSpec { from: string; to: string; from_output?: string; to_input?: string }
interface GroupSpec { name: string; layers: string[]; collapsed?: boolean }

/** Builds a graph from a list of layers and connections that refer to each other by ref */
export function buildGraph(layers: LayerSpec[], connections: LinkSpec[], groups: GroupSpec[] = []): { doc: GraphDoc; refs: Record<string, string> } {
  if (!layers.length) throw new GraphOpError("An architecture needs at least one layer.");
  let doc: GraphDoc = { nodes: [], edges: [], groups: [] };
  const ids = new Map<string, string>();
  for (const l of layers) {
    if (ids.has(l.ref)) throw new GraphOpError(`The ref "${l.ref}" is used twice.`);
    const r = addLayer(doc, { moduleType: l.type, params: l.params, label: l.label, position: { x: 0, y: 0 } });
    doc = r.doc;
    ids.set(l.ref, r.id);
  }
  const id = (ref: string) => {
    const v = ids.get(ref);
    if (!v) throw new GraphOpError(`No layer has the ref "${ref}".`);
    return v;
  };
  for (const c of connections) doc = connect(doc, { from: id(c.from), to: id(c.to), fromHandle: c.from_output, toHandle: c.to_input });
  doc = { ...doc, nodes: autoLayout(doc.nodes, doc.edges) };
  for (const [i, g] of groups.entries()) {
    const members = g.layers.map(id);
    if (members.length < 2) throw new GraphOpError(`The group "${g.name}" needs at least two layers.`);
    const taken = new Set(doc.groups.flatMap((x) => x.nodeIds));
    if (members.some((m) => taken.has(m))) throw new GraphOpError(`A layer in "${g.name}" is already in another group.`);
    doc = { ...doc, groups: [...doc.groups, { id: `group_ai_${Date.now().toString(36)}_${i}`, name: g.name, nodeIds: members, collapsed: !!g.collapsed }] };
  }
  return { doc, refs: Object.fromEntries(ids) };
}

const replaceGraph: AssistantTool<{ template?: string; layers?: LayerSpec[]; connections?: LinkSpec[]; groups?: GroupSpec[] }> = {
  kind: "edit",
  definition: {
    name: "replace_graph",
    description: `Replaces the whole model with a built-in template or with an architecture you design, as a preview the user applies or discards. Use it for "build me a ..." requests and to start over; use the other edit tools for changes to the current model.

Templates: ${TEMPLATES.map((t) => t.name).join(", ")}. A template also sets its training defaults.

To design one, list every layer with a unique ref, then the connections between refs. A trainable model needs a Data layer (its outputs are "x" for inputs and "y" for labels), the layers, and a loss layer whose "input" takes the prediction and "target" takes Data's "y"; look at a template with get_graph for the pattern. Use groups to fold repeated blocks (e.g. ResNet stages) into named frames, collapsed if they're long. Every type and setting is checked against the catalog, and the layout is done for you. The result reports shape problems; fix them with the edit tools before finishing.`,
    input_schema: {
      type: "object",
      properties: {
        template: { type: "string", enum: TEMPLATES.map((t) => t.name) },
        layers: {
          type: "array",
          items: {
            type: "object",
            properties: { ref: { type: "string" }, type: { type: "string" }, params: { type: "object" }, label: { type: "string" } },
            required: ["ref", "type"],
            additionalProperties: false,
          },
        },
        connections: {
          type: "array",
          items: {
            type: "object",
            properties: { from: { type: "string" }, to: { type: "string" }, from_output: { type: "string" }, to_input: { type: "string" } },
            required: ["from", "to"],
            additionalProperties: false,
          },
        },
        groups: {
          type: "array",
          items: {
            type: "object",
            properties: { name: { type: "string" }, layers: { type: "array", items: { type: "string" } }, collapsed: { type: "boolean" } },
            required: ["name", "layers"],
            additionalProperties: false,
          },
        },
      },
      additionalProperties: false,
    },
  },
  describe: (i) => (i.template ? `Opening the ${i.template} template` : `Building a ${i.layers?.length ?? 0}-layer model`),
  run: async ({ template, layers, connections = [], groups = [] }) => {
    refuseDuringLesson("replacing the whole model");
    if (template && layers?.length) throw new ToolError("Give either a template or layers, not both.");
    if (template) {
      const tpl = TEMPLATES.find((t) => t.name === template);
      if (!tpl) throw new ToolError(`There's no template "${template}".`);
      const { nodes, edges } = templateGraph(tpl);
      writeGraph({ nodes, edges, groups: [] });
      const d = tpl.trainingDefaults;
      if (d) {
        const c = useTrainingStore.getState().config;
        writeTraining({
          ...(d.optimizer && { optimizer: { ...c.optimizer, ...d.optimizer } }),
          ...(d.loss && { loss: d.loss }),
          ...(d.epochs != null && { epochs: d.epochs }),
          ...(d.batchSize != null && { batchSize: d.batchSize }),
          ...(d.dataset && { dataset: d.dataset }),
        });
      }
      return report({ template: tpl.name, layers: nodes.length });
    }
    if (!layers?.length) throw new ToolError("Give a template or a list of layers.");
    for (const l of layers) requireModule(l.type);
    const built = wrap(() => buildGraph(layers, connections, groups));
    writeGraph(built.doc);
    return report({ refs: built.refs });
  },
};

// ---------------------------------------------------------------------------
// create_custom_block
// ---------------------------------------------------------------------------

interface BlockInput {
  label: string;
  init_body: string;
  forward_body: string;
  inputs?: string[];
  outputs?: string[];
  after?: string;
}

function pythonNames(names: string[] | undefined, fallback: string[], what: string): string[] {
  const list = names?.length ? names : fallback;
  for (const n of list) {
    if (safePythonId(n) !== n) throw new ToolError(`"${n}" can't be a ${what} name; use a Python identifier like x or skip.`);
  }
  if (new Set(list).size !== list.length) throw new ToolError(`The ${what} names must be different.`);
  return list;
}

function blockSource(b: BlockInput, inputs: string[]): string {
  const indent = (s: string) => s.split("\n").map((l) => `        ${l}`).join("\n");
  return `class ${b.label.replace(/\W/g, "") || "CustomBlock"}(nn.Module):\n    def __init__(self):\n        super().__init__()\n${indent(b.init_body.trim() || "pass")}\n\n    def forward(self, ${inputs.join(", ")}):\n${indent(b.forward_body.trim())}`;
}

const createCustomBlock: AssistantTool<BlockInput> = {
  kind: "confirm",
  definition: {
    name: "create_custom_block",
    description: "Adds a layer that runs your own PyTorch code: an __init__ body (create submodules as self.<name>, with torch, nn and F available) and a forward body that returns the output (or a tuple, one per output). Inputs and outputs name its handles; they default to one input \"x\" and one output \"out\". With `after`, it's inserted after that layer like add_layers. The user sees the code and approves it before it's added; it then runs in Python whenever the model is trained, exported or tried. The app can't infer a custom layer's output shape ahead of time and assumes it's the same as its first input, so say what the real shape is.",
    input_schema: {
      type: "object",
      properties: {
        label: { type: "string", description: "Name shown on the canvas, e.g. \"RotaryEmbedding\"" },
        init_body: { type: "string", description: "The body of __init__ after super().__init__(), e.g. \"self.proj = nn.Linear(64, 64)\"" },
        forward_body: { type: "string", description: "The body of forward, ending in return" },
        inputs: { type: "array", items: { type: "string" } },
        outputs: { type: "array", items: { type: "string" } },
        after: { type: "string", description: "Insert after this layer" },
      },
      required: ["label", "init_body", "forward_body"],
      additionalProperties: false,
    },
  },
  describe: (i) => `Adding custom layer ${i.label}`,
  confirm: (i) => {
    const inputs = pythonNames(i.inputs, ["x"], "input");
    pythonNames(i.outputs, ["out"], "output");
    if (!/\breturn\b/.test(i.forward_body)) throw new ToolError("forward_body must return the output.");
    if (i.after) wrap(() => requireNode(currentDoc(), i.after!));
    return {
      title: `Add the custom layer "${i.label}"?`,
      summary: "This adds a layer that runs the code below. It's previewed on the canvas like other changes, and the code runs whenever you train, export or try the model.",
      code: blockSource(i, inputs),
      warning: RUNS_CODE,
      approveLabel: "Add layer",
    };
  },
  run: async (i) => {
    const handle = (id: string): HandleDef => ({ id, label: id, dataType: "tensor" });
    const inputs = pythonNames(i.inputs, ["x"], "input");
    const outputs = pythonNames(i.outputs, ["out"], "output");
    const codeBlock: CodeBlockDef = {
      inputHandles: inputs.map(handle),
      outputHandles: outputs.map(handle),
      initBody: i.init_body.trim() || "pass",
      forwardBody: i.forward_body.trim(),
      originalModuleType: "Identity",
      originalParams: {},
    };
    const doc = currentDoc();
    const result = wrap(() => {
      if (i.after) {
        // Insert as a plain layer first, then give it the code; its handles come from the code
        const base = createNode({ moduleType: "Identity", position: { x: 0, y: 0 } });
        const r = insertAfter(doc, i.after, { moduleType: "Identity", label: i.label, id: base.id });
        const nodes = r.doc.nodes.map((n) => (n.id === r.id ? { ...n, data: { ...n.data, codeBlock } } : n));
        // The inserted connections used Identity's handle names; move them to the code's
        const edges = r.doc.edges.map((e) => (e.target === r.id ? { ...e, targetHandle: inputs[0] } : e.source === r.id ? { ...e, sourceHandle: outputs[0] } : e));
        return { doc: { ...r.doc, nodes, edges }, id: r.id };
      }
      const node = createNode({ moduleType: "Identity", label: i.label, position: freePosition(doc) });
      return { doc: { ...doc, nodes: [...doc.nodes, { ...node, data: { ...node.data, codeBlock } }] }, id: node.id };
    });
    writeGraph(result.doc);
    return report({ created: result.id, inputs, outputs });
  },
};

// ---------------------------------------------------------------------------
// import_code
// ---------------------------------------------------------------------------

const importCode: AssistantTool<{ code: string }> = {
  kind: "confirm",
  definition: {
    name: "import_code",
    description: "Turns PyTorch code that defines an nn.Module (a model the user pasted, or one you write from a paper's description) into a graph that replaces the current one, as a preview. The code runs in Python to trace the model, so the user sees it and approves it first. Layers the app has no block for become Identity layers; say which. Needs Python.",
    input_schema: {
      type: "object",
      properties: { code: { type: "string", description: "Python code defining the model class; it must build with no arguments" } },
      required: ["code"],
      additionalProperties: false,
    },
  },
  describe: () => "Importing code",
  confirm: ({ code }) => {
    refuseDuringLesson("importing code");
    return {
      title: "Run this code to import the model?",
      summary: "TensorBloom runs this code in Python to trace the model, then shows the result on the canvas for you to apply or discard.",
      code,
      warning: RUNS_CODE,
      approveLabel: "Run and import",
    };
  },
  run: async ({ code }) => {
    let compiled: CompiledGraph;
    try {
      compiled = await sidecarCall<CompiledGraph>("compile", { code });
    } catch (e) {
      throw new ToolError(`Importing failed: ${e instanceof Error ? e.message : String(e)}`);
    }
    const { nodes, edges } = compiledToGraph(compiled);
    writeGraph({ nodes, edges, groups: [] });
    const unknown = compiled.nodes.filter((n) => !n.moduleType || n.moduleType === "Identity").map((n) => n.label);
    return report({ layers: nodes.length, ...(unknown.length ? { importedAsIdentity: unknown } : {}) });
  },
};

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

const startTrainingTool: AssistantTool = {
  kind: "confirm",
  definition: {
    name: "start_training",
    description: "Starts training the model on the canvas with the current training settings, after the user approves. Pending changes must be applied first. Returns once the run has started (or failed to start); results arrive in the Runs tab, which list_runs reads later.",
    input_schema: { type: "object", properties: {}, additionalProperties: false },
  },
  describe: () => "Starting training",
  confirm: () => {
    if (hasPendingChanges()) throw new ToolError("There are pending changes. Ask the user to apply or discard them first.");
    const s = useTrainingStore.getState();
    if (s.status === "running" || s.status === "starting") throw new ToolError("Training is already running.");
    const c = s.config;
    const custom = useGraphStore.getState().nodes.filter((n) => n.data.codeBlock || n.data.customCode).map((n) => n.data.label);
    return {
      title: "Start training?",
      summary: `Trains for ${c.epochs} epochs with ${c.optimizer.type} (learning rate ${c.optimizer.lr}), batch size ${c.batchSize}, on device "${c.device}". You can stop it at any time.`,
      warning: custom.length
        ? `Training runs the model in Python on your computer, including the custom code in ${custom.join(", ")}.`
        : "Training runs the model in Python on your computer and may download the dataset first.",
      approveLabel: "Start training",
    };
  },
  run: async () => {
    void startTraining();
    // startTraining marks the run "starting" right away when it goes ahead
    if (useTrainingStore.getState().status !== "starting") {
      return { status: "not started", note: "Training didn't start: Python may not be set up yet (the app asks the user to set it up)." };
    }
    // Wait until the run is going, or until it fails its checks or the user cancels the pre-flight review
    const deadline = Date.now() + 120_000;
    for (;;) {
      await new Promise((r) => setTimeout(r, 250));
      const s = useTrainingStore.getState();
      if (s.status === "running" || s.status === "finished") return { status: "started", note: "Training is running. Results appear in the Runs tab when it ends." };
      if (s.status === "error") throw new ToolError(`Training didn't start: ${s.error ?? "unknown error"}`);
      if (s.status !== "starting") return { status: "not started", note: "The user cancelled at the pre-flight check." };
      if (Date.now() > deadline) return { status: "starting", note: "Still starting (probably downloading the dataset)." };
    }
  },
};

const stopTrainingTool: AssistantTool = {
  kind: "read",
  definition: {
    name: "stop_training",
    description: "Stops the training run in progress after its current batch. The run is kept with the results so far.",
    input_schema: { type: "object", properties: {}, additionalProperties: false },
  },
  describe: () => "Stopping training",
  run: async () => {
    const s = useTrainingStore.getState().status;
    if (s !== "running" && s !== "starting") return { note: "Training isn't running." };
    await stopTraining();
    return { status: "stopping" };
  },
};

export const CODE_TOOLS: AssistantTool[] = [replaceGraph, createCustomBlock, importCode, startTrainingTool, stopTrainingTool] as AssistantTool[];
