import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { useGraphStore } from "./graph-store";
import { useHistoryStore, startHistoryTracking, COMMIT_DELAY_MS } from "./history-store";

let stop: () => void;
const graph = () => useGraphStore.getState();
const history = () => useHistoryStore.getState();
const settle = () => vi.advanceTimersByTime(COMMIT_DELAY_MS + 50);
const nodeCount = () => graph().nodes.length;

function addLinear(x = 0) {
  graph().addNode("Linear", { x, y: 0 });
  return graph().nodes[graph().nodes.length - 1].id;
}

beforeEach(() => {
  vi.useFakeTimers();
  graph().clearGraph();
  stop = startHistoryTracking();
});

afterEach(() => {
  stop();
  vi.useRealTimers();
});

describe("undo and redo", () => {
  it("undoes an added node with one press", () => {
    addLinear();
    settle();
    history().undo();
    expect(nodeCount()).toBe(0);
  });

  it("undoes an edit made a moment ago, before it has settled", () => {
    addLinear();
    history().undo();
    expect(nodeCount()).toBe(0);
  });

  it("redoes what was undone", () => {
    addLinear();
    settle();
    history().undo();
    settle();
    expect(history().canRedo).toBe(true);
    history().redo();
    expect(nodeCount()).toBe(1);
  });

  it("walks back and forward through several steps", () => {
    addLinear(0);
    settle();
    addLinear(100);
    settle();
    addLinear(200);
    settle();
    history().undo();
    history().undo();
    expect(nodeCount()).toBe(1);
    history().redo();
    expect(nodeCount()).toBe(2);
    history().redo();
    expect(nodeCount()).toBe(3);
    expect(history().canRedo).toBe(false);
  });

  it("drops the redo steps once a new edit is made", () => {
    addLinear();
    settle();
    history().undo();
    addLinear(50);
    settle();
    expect(history().canRedo).toBe(false);
  });

  it("treats a burst of typing in one field as one step", () => {
    const id = addLinear();
    settle();
    for (const v of [5, 51, 512]) graph().updateNodeParam(id, "out_features", v);
    settle();
    history().undo();
    const node = graph().nodes.find((n) => n.id === id)!;
    expect(node.data.params.out_features).not.toBe(512);
    expect(nodeCount()).toBe(1);
  });

  it("undoes separate parameter edits one at a time", () => {
    const id = addLinear();
    settle();
    graph().updateNodeParam(id, "in_features", 10);
    settle();
    graph().updateNodeParam(id, "out_features", 20);
    settle();
    history().undo();
    const node = graph().nodes.find((n) => n.id === id)!;
    expect(node.data.params.in_features).toBe(10);
    expect(node.data.params.out_features).not.toBe(20);
  });

  it("doesn't record selecting a node as a step", () => {
    const id = addLinear();
    settle();
    const steps = history().past.length;
    graph().onNodesChange([{ type: "select", id, selected: true }]);
    settle();
    expect(history().past.length).toBe(steps);
  });

  it("records a whole drag as one step", () => {
    const id = addLinear();
    settle();
    const start = { ...graph().nodes[0].position };
    for (const x of [10, 20, 30]) {
      graph().onNodesChange([{ type: "position", id, position: { x, y: 0 }, dragging: true }]);
      settle();
    }
    graph().onNodesChange([{ type: "position", id, position: { x: 40, y: 0 }, dragging: false }]);
    settle();
    history().undo();
    expect(graph().nodes[0].position).toEqual(start);
  });

  it("keeps undo working after the last step is undone and a new one is made", () => {
    addLinear();
    settle();
    history().undo();
    settle();
    addLinear(30);
    settle();
    history().undo();
    expect(nodeCount()).toBe(0);
  });
});
