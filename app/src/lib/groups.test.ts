import { describe, it, expect } from "vitest";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import { addGroup, canvasView, groupRect, nextGroupName, normalizeGroups, pruneGroups, CARD_WIDTH, FRAME_PAD, FRAME_HEADER, GROUP_NODE_TYPE, type LayerGroup } from "./groups";

const layer = (id: string, y: number): Node<NodeData> =>
  ({ id, type: "tensorLabNode", position: { x: 0, y }, measured: { width: 200, height: 80 }, data: { moduleType: "Linear", label: id, category: "linear", params: {} } });
const edge = (source: string, target: string): Edge => ({ id: `${source}-${target}`, source, target, sourceHandle: "out", targetHandle: "in" });

// in → a → b → out, with a and b grouped
const nodes = [layer("in", 0), layer("a", 120), layer("b", 240), layer("out", 360)];
const edges = [edge("in", "a"), edge("a", "b"), edge("b", "out")];
const group: LayerGroup = { id: "g", name: "Block 1", nodeIds: ["a", "b"], collapsed: false };

describe("groupRect", () => {
  it("frames an open group's layers with room for its title", () => {
    expect(groupRect(group, nodes)).toEqual({ x: -FRAME_PAD, y: 120 - FRAME_PAD - FRAME_HEADER, w: 200 + FRAME_PAD * 2, h: 200 + FRAME_PAD * 2 + FRAME_HEADER });
  });

  it("puts a collapsed group's card at the top of its layers", () => {
    const r = groupRect({ ...group, collapsed: true }, nodes)!;
    expect(r.y).toBe(120);
    expect(r.w).toBe(CARD_WIDTH);
  });
});

describe("canvasView", () => {
  it("adds a frame behind an open group and leaves the graph alone", () => {
    const view = canvasView(nodes, edges, [group]);
    const frame = view.nodes.find((n) => n.id === "g")!;
    expect(frame.type).toBe(GROUP_NODE_TYPE);
    expect(frame.zIndex).toBe(-1);
    expect(view.nodes.filter((n) => n.hidden)).toHaveLength(0);
    expect(view.edges).toEqual(edges);
  });

  it("hides a collapsed group's layers and attaches the edges crossing it to its card", () => {
    const view = canvasView(nodes, edges, [{ ...group, collapsed: true }]);
    expect(view.nodes.filter((n) => n.hidden).map((n) => n.id).sort()).toEqual(["a", "b"]);
    const byId = Object.fromEntries(view.edges.map((e) => [e.id, e]));
    expect(byId["in-a"]).toMatchObject({ source: "in", target: "g", targetHandle: "in", data: { origSource: "in", origTarget: "a" } });
    expect(byId["a-b"].hidden).toBe(true);
    expect(byId["b-out"]).toMatchObject({ source: "g", sourceHandle: "out", target: "out" });
  });
});

describe("group bookkeeping", () => {
  it("takes layers out of the group they were in", () => {
    const groups = addGroup([group], ["b", "out"], "Block 2", "g2");
    expect(groups.find((g) => g.id === "g")!.nodeIds).toEqual(["a"]);
    expect(groups.find((g) => g.id === "g2")!.nodeIds).toEqual(["b", "out"]);
  });

  it("forgets deleted layers, and groups left empty", () => {
    const unchanged = [group];
    expect(pruneGroups(unchanged, nodes)).toBe(unchanged);
    expect(pruneGroups([group], nodes.filter((n) => n.id !== "a"))[0].nodeIds).toEqual(["b"]);
    expect(pruneGroups([group], nodes.filter((n) => n.id !== "a" && n.id !== "b"))).toEqual([]);
  });

  it("names new groups after the ones that exist", () => {
    expect(nextGroupName([group])).toBe("Block 2");
  });

  it("keeps only well-formed groups from a file", () => {
    const raw = [
      { id: "g1", name: "Encoder", nodeIds: ["a", "b"], collapsed: true },
      { id: "g2", name: "", nodeIds: ["b", "c", 7] },
      { id: "g3", nodeIds: [] },
      { name: "no id", nodeIds: ["d"] },
      null,
    ];
    expect(normalizeGroups(raw)).toEqual([
      { id: "g1", name: "Encoder", nodeIds: ["a", "b"], collapsed: true },
      { id: "g2", name: "Group", nodeIds: ["c"], collapsed: false },
    ]);
    expect(normalizeGroups(undefined)).toEqual([]);
  });
});
