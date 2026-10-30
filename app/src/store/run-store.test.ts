import { describe, it, expect, beforeEach } from "vitest";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "./graph-store";
import type { TrainingMetrics } from "./training-store";
import { useRunStore, epochSummary, architectureKey, MAX_RUNS, type RunRecord } from "./run-store";

const node = (id: string, moduleType: string, params: Record<string, unknown>, x = 0): Node<NodeData> =>
  ({ id, type: "tensorbloom", position: { x, y: 0 }, data: { moduleType, params } as NodeData });

const run = (n: number): RunRecord => ({ id: `r${n}`, number: n } as RunRecord);

describe("epochSummary", () => {
  it("keeps each epoch's last report and the validation numbers that came with it", () => {
    const metrics = [
      { epoch: 1, step: 1, trainLoss: 2.0, lr: 0.01 },
      { epoch: 1, step: 2, trainLoss: 1.5, lr: 0.01, valLoss: 1.6, valAcc: 0.4 },
      { epoch: 2, step: 3, trainLoss: 1.2, lr: 0.005 },
      { epoch: 2, step: 4, trainLoss: 1.0, lr: 0.005, valLoss: 1.1 },
    ] as TrainingMetrics[];
    expect(epochSummary(metrics)).toEqual([
      { epoch: 1, trainLoss: 1.5, lr: 0.01, valLoss: 1.6, valAcc: 0.4 },
      { epoch: 2, trainLoss: 1.0, lr: 0.005, valLoss: 1.1 },
    ]);
  });
});

describe("architectureKey", () => {
  const edges: Edge[] = [{ id: "e1", source: "a", target: "b" }];

  it("ignores where layers sit and the order they're listed in", () => {
    const one = [node("a", "Linear", { in_features: 4, out_features: 8 }), node("b", "ReLU", {})];
    const moved = [node("b", "ReLU", {}, 300), node("a", "Linear", { in_features: 4, out_features: 8 }, 100)];
    expect(architectureKey(moved, edges)).toBe(architectureKey(one, edges));
  });

  it("changes with a layer's settings or a connection", () => {
    const one = [node("a", "Linear", { in_features: 4, out_features: 8 }), node("b", "ReLU", {})];
    const wider = [node("a", "Linear", { in_features: 4, out_features: 16 }), node("b", "ReLU", {})];
    expect(architectureKey(wider, edges)).not.toBe(architectureKey(one, edges));
    expect(architectureKey(one, [])).not.toBe(architectureKey(one, edges));
  });
});

describe("useRunStore", () => {
  beforeEach(() => useRunStore.setState({ runs: [], compareIds: [] }));

  it("compares a new run with the one before it", () => {
    useRunStore.getState().addRun(run(1));
    expect(useRunStore.getState().compareIds).toEqual(["r1"]);
    useRunStore.getState().addRun(run(2));
    expect(useRunStore.getState().compareIds).toEqual(["r1", "r2"]);
  });

  it(`keeps the last ${MAX_RUNS} runs`, () => {
    for (let i = 1; i <= MAX_RUNS + 2; i++) useRunStore.getState().addRun(run(i));
    const { runs } = useRunStore.getState();
    expect(runs).toHaveLength(MAX_RUNS);
    expect(runs[0].number).toBe(3);
  });

  it("drops a deleted run from the comparison", () => {
    useRunStore.getState().addRun(run(1));
    useRunStore.getState().addRun(run(2));
    useRunStore.getState().removeRun("r1");
    expect(useRunStore.getState().compareIds).toEqual(["r2"]);
  });
});
