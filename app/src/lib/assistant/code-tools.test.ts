import { describe, it, expect, beforeEach, vi } from "vitest";

const compile = vi.fn();
vi.mock("../sidecar", () => ({ sidecarCall: (method: string, params: unknown) => (method === "compile" ? compile(params) : Promise.reject(new Error("no"))) }));
const startTraining = vi.fn();
vi.mock("../training", () => ({ startTraining: () => startTraining(), stopTraining: vi.fn(), sidecarTrainConfig: vi.fn() }));

const { useGraphStore } = await import("../../store/graph-store");
const { useTrainingStore } = await import("../../store/training-store");
const { CODE_TOOLS, buildGraph } = await import("./code-tools");
const { runToolCall } = await import("./agent");
const { applyChanges, discardChanges, hasPendingChanges } = await import("./change-set");
import type { ApprovalRequest } from "./tool-types";

const graph = () => useGraphStore.getState();
let asked: ApprovalRequest[] = [];
const call = async (name: string, input: unknown, approve = true) => {
  const r = await runToolCall(CODE_TOOLS, { id: "t", name, input }, async (req) => { asked.push(req); return approve; });
  let out: unknown = r.result.content;
  try { out = JSON.parse(r.result.content as string); } catch { /* plain text */ }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return { ok: r.ok, out: out as any };
};

beforeEach(() => {
  if (hasPendingChanges()) discardChanges();
  graph().clearGraph();
  asked = [];
  compile.mockReset();
  startTraining.mockReset();
});

const MLP = {
  layers: [
    { ref: "data", type: "Data", params: { source: "MNIST" } },
    { ref: "flat", type: "Flatten" },
    { ref: "fc", type: "Linear", params: { in_features: 784, out_features: 10 } },
    { ref: "loss", type: "CrossEntropyLoss" },
  ],
  connections: [
    { from: "data", to: "flat", from_output: "x" },
    { from: "flat", to: "fc" },
    { from: "fc", to: "loss", to_input: "input" },
    { from: "data", to: "loss", from_output: "y", to_input: "target" },
  ],
};

describe("replace_graph", () => {
  it("builds a designed architecture, laid out, as a preview", async () => {
    const r = await call("replace_graph", { ...MLP, groups: [{ name: "Head", layers: ["fc", "loss"] }] });
    expect(r.ok).toBe(true);
    expect(Object.keys(r.out.refs)).toEqual(["data", "flat", "fc", "loss"]);
    expect(graph().nodes).toHaveLength(4);
    expect(graph().groups[0]).toMatchObject({ name: "Head", nodeIds: [r.out.refs.fc, r.out.refs.loss] });
    expect(new Set(graph().nodes.map((n) => n.position.y)).size).toBeGreaterThan(1);
    expect(asked).toHaveLength(0);
    expect(hasPendingChanges()).toBe(true);
  });

  it("rejects unknown refs and handles without changing anything", async () => {
    const r = await call("replace_graph", { layers: MLP.layers, connections: [{ from: "data", to: "nope" }] });
    expect(r.ok).toBe(false);
    expect(r.out).toBe('No layer has the ref "nope".');
    expect(graph().nodes).toHaveLength(0);
  });

  it("opens a template with its training defaults", async () => {
    const r = await call("replace_graph", { template: "LeNet-5" });
    expect(r.ok).toBe(true);
    expect(graph().nodes.length).toBeGreaterThan(5);
  });

  it("builds every layer of the architecture with the app's layout", () => {
    const { doc } = buildGraph(MLP.layers, MLP.connections);
    expect(doc.edges).toHaveLength(4);
  });
});

describe("create_custom_block", () => {
  const block = { label: "Residual", init_body: "self.fc = nn.Linear(10, 10)", forward_body: "return x + self.fc(x)" };

  it("shows the code and adds nothing when the user says no", async () => {
    await call("replace_graph", MLP);
    applyChanges();
    const r = await call("create_custom_block", { ...block, after: graph().nodes.find((n) => n.data.moduleType === "Linear")!.id }, false);
    expect(asked[0].code).toContain("def forward(self, x):\n        return x + self.fc(x)");
    expect(asked[0].warning).toMatch(/runs Python on your computer/);
    expect(r.ok).toBe(false);
    expect(graph().nodes).toHaveLength(4);
  });

  it("inserts the layer with the code's handles once approved", async () => {
    await call("replace_graph", MLP);
    applyChanges();
    const fc = graph().nodes.find((n) => n.data.moduleType === "Linear")!;
    const r = await call("create_custom_block", { ...block, after: fc.id });
    expect(r.ok).toBe(true);
    const node = graph().nodes.find((n) => n.id === r.out.created)!;
    expect(node.data.codeBlock?.forwardBody).toBe("return x + self.fc(x)");
    const into = graph().edges.find((e) => e.target === node.id)!;
    const out = graph().edges.find((e) => e.source === node.id)!;
    expect(into).toMatchObject({ source: fc.id, targetHandle: "x" });
    expect(out).toMatchObject({ sourceHandle: "out", targetHandle: "input" });
  });

  it("refuses bad handle names and a forward without return before asking", async () => {
    expect((await call("create_custom_block", { ...block, inputs: ["my input"] })).out).toMatch(/Python identifier/);
    expect((await call("create_custom_block", { ...block, forward_body: "x + 1" })).out).toMatch(/must return/);
    expect(asked).toHaveLength(0);
  });
});

describe("import_code", () => {
  it("runs the code only after approval and previews the result", async () => {
    compile.mockResolvedValue({
      nodes: [{ id: "a", type: "x", moduleType: "Linear", label: "fc", position: { x: 0, y: 0 }, params: { in_features: 4, out_features: 2 } }, { id: "b", type: "x", label: "mystery", position: { x: 0, y: 0 }, params: {} }],
      edges: [{ id: "e", source: "a", target: "b", sourceHandle: "out", targetHandle: "in" }],
    });
    const declined = await call("import_code", { code: "class M(nn.Module): ..." }, false);
    expect(declined.ok).toBe(false);
    expect(compile).not.toHaveBeenCalled();
    const r = await call("import_code", { code: "class M(nn.Module): ..." });
    expect(compile).toHaveBeenCalledWith({ code: "class M(nn.Module): ..." });
    expect(r.out.importedAsIdentity).toEqual(["mystery"]);
    expect(graph().nodes).toHaveLength(2);
    expect(hasPendingChanges()).toBe(true);
  });
});

describe("start_training", () => {
  it("won't start with changes pending", async () => {
    await call("replace_graph", MLP);
    const r = await call("start_training", {});
    expect(r.out).toMatch(/apply or discard/);
    expect(startTraining).not.toHaveBeenCalled();
  });

  it("asks first, then starts", async () => {
    startTraining.mockImplementation(() => { useTrainingStore.setState({ status: "starting" }); setTimeout(() => useTrainingStore.setState({ status: "running" }), 10); });
    const r = await call("start_training", {});
    expect(asked[0].title).toBe("Start training?");
    expect(r.out.status).toBe("started");
    useTrainingStore.setState({ status: "idle" });
  });
});
