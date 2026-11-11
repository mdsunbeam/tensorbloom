import { describe, it, expect } from "vitest";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { TrainingConfig } from "../store/training-store";
import { graphDiff, settingsDiff, setupDiff, changeCount, changeSummary, describeChanges, nodeMarks } from "./experiment-diff";

const node = (id: string, moduleType: string, params: Record<string, unknown> = {}, label = id): Node<NodeData> => ({
  id, position: { x: 0, y: 0 }, type: "tensorLabNode", data: { moduleType, label, category: "linear", params },
});
const edge = (s: string, t: string): Edge => ({ id: `${s}-${t}`, source: s, target: t, sourceHandle: "out", targetHandle: "in" });

const base = {
  nodes: [node("a", "Linear", { in_features: 4, out_features: 8 }), node("b", "ReLU"), node("c", "Linear", { in_features: 8, out_features: 2 })],
  edges: [edge("a", "b"), edge("b", "c")],
};

const config: TrainingConfig = {
  optimizer: { type: "Adam", lr: 0.001, weightDecay: 0 },
  scheduler: null, loss: "CrossEntropyLoss", epochs: 5, batchSize: 32, valSplit: 0.2,
  dataset: { type: "MNIST", root: "./data" }, device: "auto", mixedPrecision: false,
  gradientClipping: { enabled: false, maxNorm: 1 }, earlyStopping: { enabled: false, patience: 5, minDelta: 0.001 },
  resumeCheckpoint: "", augmentation: { horizontalFlip: false, verticalFlip: false, randomCrop: false, colorJitter: false, rotation: 0 },
  weightInit: "default", seed: 42, dataloaderSeed: 7, deterministicMode: false,
};

describe("graph diff", () => {
  it("finds nothing when only positions and selection differ", () => {
    const moved = { nodes: base.nodes.map((n) => ({ ...n, position: { x: 50, y: 90 }, selected: true })), edges: base.edges };
    const d = graphDiff(base, moved);
    expect(d.layers).toEqual([]);
    expect(d.connections).toEqual([]);
  });

  it("names a changed setting", () => {
    const b = { ...base, nodes: base.nodes.map((n) => (n.id === "a" ? node("a", "Linear", { in_features: 4, out_features: 16 }) : n)) };
    const d = setupDiff({ ...base, config }, { ...b, config });
    expect(d.layers).toEqual([{ id: "a", label: "a", moduleType: "Linear", kind: "changed", params: [{ name: "out_features", from: "8", to: "16" }] }]);
    expect(changeCount(d)).toBe(1);
    expect(changeSummary(d)).toBe("a out_features 8 → 16");
    expect(nodeMarks(d).get("a")).toBe("changed");
  });

  it("finds added and removed layers without counting their own connections", () => {
    const b = {
      nodes: [...base.nodes.filter((n) => n.id !== "b"), node("bn", "BatchNorm1d", { num_features: 8 }), node("b2", "GELU")],
      edges: [edge("a", "bn"), edge("bn", "b2"), edge("b2", "c")],
    };
    const d = graphDiff(base, b);
    expect(d.layers.map((l) => `${l.kind} ${l.id}`).sort()).toEqual(["added b2", "added bn", "removed b"]);
    expect(d.connections).toEqual([]);
  });

  it("finds rewired connections and a replaced layer type", () => {
    const b = { nodes: base.nodes.map((n) => (n.id === "b" ? node("b", "GELU") : n)), edges: [edge("a", "c"), edge("b", "c")] };
    const d = graphDiff(base, b);
    expect(d.layers[0]).toMatchObject({ id: "b", replaced: { from: "ReLU", to: "GELU" } });
    expect(d.connections.map((c) => `${c.kind} ${c.from}>${c.to}`).sort()).toEqual(["added a>c", "removed a>b"]);
    expect(nodeMarks({ ...d, settings: [] }).get("c")).toBe("changed");
  });

  it("doesn't count a rename as a change to the model", () => {
    const b = { ...base, nodes: base.nodes.map((n) => (n.id === "a" ? node("a", "Linear", { in_features: 4, out_features: 8 }, "encoder") : n)) };
    const d = setupDiff({ ...base, config }, { ...b, config });
    expect(d.layers[0].renamed).toEqual({ from: "a", to: "encoder" });
    expect(changeCount(d)).toBe(0);
  });
});

describe("settings diff", () => {
  it("lists changed training settings and ignores the device", () => {
    const d = settingsDiff(config, { ...config, optimizer: { ...config.optimizer, lr: 0.01 }, device: "cuda:0" });
    expect(d).toEqual([{ label: "Learning rate", from: "0.001", to: "0.01" }]);
  });

  it("counts several changes and says so", () => {
    const d = setupDiff({ ...base, config }, { ...base, config: { ...config, epochs: 10, batchSize: 64, mixedPrecision: true } });
    expect(changeCount(d)).toBe(3);
    expect(describeChanges(d)).toEqual(["Epochs 5 → 10", "Batch size 32 → 64", "Mixed precision off → on"]);
    expect(changeSummary(d)).toBe("3 changes: Epochs 5 → 10, Batch size 32 → 64, …");
  });
});
