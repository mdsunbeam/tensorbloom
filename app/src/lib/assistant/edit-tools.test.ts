import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { useGraphStore } from "../../store/graph-store";
import { useHistoryStore, startHistoryTracking, COMMIT_DELAY_MS } from "../../store/history-store";
import { useTrainingStore } from "../../store/training-store";
import { EDIT_TOOLS } from "./edit-tools";
import { applyChanges, discardChanges, hasPendingChanges, takeDecision, useChangeSetStore } from "./change-set";
import { runToolCall } from "./agent";
import { TEMPLATES, templateGraph } from "../templates";

const graph = () => useGraphStore.getState();
const call = async (name: string, input: unknown) => {
  const r = await runToolCall(EDIT_TOOLS, { id: "t", name, input });
  return { ok: r.ok, out: r.ok ? JSON.parse(r.result.content as string) : r.result.content };
};
let stop: () => void;

beforeEach(() => {
  vi.useFakeTimers();
  if (hasPendingChanges()) discardChanges();
  takeDecision();
  graph().clearGraph();
  const tpl = TEMPLATES.find((t) => t.name === "Simple MLP")!;
  const { nodes, edges } = templateGraph(tpl);
  graph().loadGraph(nodes, edges);
  stop = startHistoryTracking();
});
afterEach(() => { stop(); vi.useRealTimers(); });

const linears = () => graph().nodes.filter((n) => n.data.moduleType === "Linear");

describe("edit tools", () => {
  it("inserts layers, reports the change, and leaves it pending", async () => {
    const [fc1] = linears();
    const before = graph().nodes.length;
    const r = await call("add_layers", { layers: [{ type: "BatchNorm1d", params: { num_features: 512 }, after: fc1.id }] });
    expect(r.ok).toBe(true);
    expect(r.out.created.ids).toHaveLength(1);
    expect(r.out.pendingChanges).toContain(`added ${r.out.created.ids[0]} BatchNorm1d`);
    expect(r.out.newShapeProblems).toBe("none");
    expect(graph().nodes).toHaveLength(before + 1);
    expect(hasPendingChanges()).toBe(true);
  });

  it("reports shape problems it causes, so they can be fixed", async () => {
    const [fc1] = linears();
    const r = await call("set_params", { changes: [{ layer_id: fc1.id, params: { out_features: 100 } }] });
    expect(r.out.newShapeProblems[0]).toMatchObject({ kind: "cause", suggestedFix: { param: "in_features", to: 100 } });
    const fixed = await call("apply_shape_fix", {});
    expect(fixed.out.newShapeProblems).toBe("none");
  });

  it("changes nothing when any part of a call is invalid", async () => {
    const [fc1] = linears();
    const before = JSON.stringify(graph().nodes);
    const r = await call("add_layers", { layers: [{ type: "Dropout", after: fc1.id }, { type: "Dropout", after: "nope" }] });
    expect(r.ok).toBe(false);
    expect(r.out).toBe("There's no layer nope.");
    expect(JSON.stringify(graph().nodes)).toBe(before);
  });

  it("rejects made-up layer types with suggestions", async () => {
    const r = await call("add_layers", { layers: [{ type: "Dropuot" }] });
    expect(r.out).toMatch(/isn't a layer TensorBloom has. Closest: Dropout/);
  });

  it("links layers made in the same call by ref", async () => {
    const r = await call("add_layers", {
      layers: [{ ref: "a", type: "Linear", params: { in_features: 10, out_features: 10 } }, { ref: "b", type: "ReLU" }],
      connections: [{ from: "a", to: "b" }],
    });
    const { a, b } = r.out.created.refs;
    expect(graph().edges.some((e) => e.source === a && e.target === b)).toBe(true);
  });

  it("changes training settings within the panel's choices", async () => {
    expect((await call("set_training_config", { optimizer: "Lion" })).ok).toBe(false);
    const r = await call("set_training_config", { optimizer: "SGD", learning_rate: 0.05, epochs: 3 });
    expect(r.out.pendingChanges).toContain("training epochs: 10 → 3");
    expect(useTrainingStore.getState().config.optimizer).toMatchObject({ type: "SGD", lr: 0.05 });
  });
});

describe("applying and discarding", () => {
  it("applies the whole change set as one undo step", async () => {
    const before = graph().nodes.length;
    for (const fc of linears()) await call("add_layers", { layers: [{ type: "Dropout", after: fc.id }] });
    vi.advanceTimersByTime(COMMIT_DELAY_MS + 50);
    expect(useHistoryStore.getState().canUndo).toBe(false);
    applyChanges();
    expect(graph().nodes).toHaveLength(before + 3);
    expect(takeDecision()).toBe("applied");
    useHistoryStore.getState().undo();
    expect(graph().nodes).toHaveLength(before);
    useHistoryStore.getState().redo();
    expect(graph().nodes).toHaveLength(before + 3);
  });

  it("discards back to exactly where it started, leaving no undo step", async () => {
    const before = JSON.stringify(graph().nodes);
    const epochs = useTrainingStore.getState().config.epochs;
    await call("remove_layers", { layer_ids: [linears()[1].id] });
    await call("set_training_config", { epochs: epochs + 5 });
    discardChanges();
    expect(JSON.stringify(graph().nodes)).toBe(before);
    expect(useTrainingStore.getState().config.epochs).toBe(epochs);
    expect(useChangeSetStore.getState().lastDecision).toBe("discarded");
    vi.advanceTimersByTime(COMMIT_DELAY_MS + 50);
    expect(useHistoryStore.getState().canUndo).toBe(false);
  });

  it("turns undo off while changes are previewed", async () => {
    graph().addNode("ReLU", { x: 0, y: 0 });
    vi.advanceTimersByTime(COMMIT_DELAY_MS + 50);
    await call("remove_layers", { layer_ids: [linears()[0].id] });
    const n = graph().nodes.length;
    useHistoryStore.getState().undo();
    expect(graph().nodes.length).toBe(n);
  });
});
