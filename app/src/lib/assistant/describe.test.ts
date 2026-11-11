import { describe, it, expect } from "vitest";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../../store/graph-store";
import { inferShapesClient } from "../shape-inference";
import { describeGraph, flowOrder, graphDiff, configDiff, shapeProblems, visibleParams, type GraphView } from "./describe";

function node(id: string, moduleType: string, params: Record<string, unknown>, y: number): Node<NodeData> {
  return { id, type: "tensorLabNode", position: { x: 0, y }, data: { moduleType, label: moduleType, category: "linear", params } };
}
const edge = (source: string, target: string): Edge => ({ id: `e_${source}_${target}`, source, target, sourceHandle: "out", targetHandle: "in" });

function view(nodes: Node<NodeData>[], edges: Edge[]): GraphView {
  const { shapes, errors } = inferShapesClient(nodes, edges);
  return { nodes, edges, groups: [], shapeMap: shapes, shapeErrors: errors };
}

// Listed out of order on purpose: the description follows the data, not the array
const nodes = [
  node("n3", "Linear", { in_features: 500, out_features: 10, bias: true }, 200),
  node("n1", "Input", { shape: "[1, 784]", dtype: "float32" }, 0),
  node("n2", "Linear", { in_features: 784, out_features: 512, bias: true, _scan: { big: true } }, 100),
  node("n4", "Output", {}, 300),
];
const edges = [edge("n1", "n2"), edge("n2", "n3"), edge("n3", "n4")];

describe("describeGraph", () => {
  it("lists layers in the order data flows, with inputs and shapes", () => {
    const text = describeGraph(view(nodes, edges));
    const lines = text.split("\n");
    expect(lines[1]).toMatch(/^n1 Input/);
    expect(lines[2]).toMatch(/^n2 Linear \(in_features=784, out_features=512, bias=true\) ← n1 → \[1, 512\]/);
    expect(lines[3]).toContain("← n2");
  });

  it("marks the layer whose shape is wrong", () => {
    const text = describeGraph(view(nodes, edges));
    expect(text.split("\n").find((l) => l.startsWith("n3"))).toContain("⚠ shape error");
  });

  it("leaves out the app's cached scan results", () => {
    expect(describeGraph(view(nodes, edges))).not.toContain("_scan");
    expect(visibleParams({ a: 1, _b: 2, c: undefined })).toEqual({ a: 1 });
  });

  it("says so when the canvas is empty", () => {
    expect(describeGraph(view([], []))).toBe("The canvas is empty.");
  });
});

describe("shapeProblems", () => {
  it("gives the cause first, with the app's one-click fix", () => {
    const [first] = shapeProblems(view(nodes, edges));
    expect(first.layer).toBe("n3");
    expect(first.kind).toBe("cause");
    expect(first.suggestedFix).toEqual({ param: "in_features", from: 500, to: 512 });
  });
});

describe("flowOrder", () => {
  it("still lists every layer when there's a cycle", () => {
    const order = flowOrder(nodes, [...edges, edge("n3", "n2")]);
    expect(order.map((n) => n.id).sort()).toEqual(["n1", "n2", "n3", "n4"]);
  });
});

describe("graphDiff", () => {
  it("names added, removed and changed layers and connections", () => {
    const after = [nodes[1], { ...nodes[2], data: { ...nodes[2].data, params: { ...nodes[2].data.params, out_features: 500 } } }, nodes[0], node("n5", "ReLU", {}, 150)];
    const diff = graphDiff({ nodes, edges }, { nodes: after, edges: [edge("n1", "n2"), edge("n2", "n5"), edge("n5", "n3")] });
    expect(diff).toContain("n2 Linear.out_features: 512 → 500");
    expect(diff).toContain("added n5 ReLU");
    expect(diff).toContain("removed n4 Output");
    expect(diff).toContain("connected n2:out>n5:in");
    expect(diff).toContain("disconnected n3:out>n4:in");
  });
});

describe("configDiff", () => {
  it("lists nested settings that differ", () => {
    expect(configDiff({ a: 1, b: { c: 2, d: 3 } } as never, { a: 1, b: { c: 5, d: 3 } } as never)).toEqual(["b.c: 2 → 5"]);
  });
});
