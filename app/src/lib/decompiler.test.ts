import { describe, it, expect } from "vitest";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import { decompileGraph } from "./decompiler";

function node(id: string, moduleType: string, params: Record<string, unknown> = {}): Node<NodeData> {
  return { id, type: "tensorLabNode", position: { x: 0, y: 0 }, data: { moduleType, label: id, category: "transformer", params } } as Node<NodeData>;
}
const edge = (source: string, target: string, targetHandle = "in"): Edge =>
  ({ id: `${source}-${target}-${targetHandle}`, source, target, sourceHandle: "out", targetHandle }) as Edge;

describe("decompileGraph: TransformerDecoderLayer", () => {
  const decoder = { d_model: 64, nhead: 4, batch_first: true };

  it("masks future positions when the layer is a GPT-style block", () => {
    const { code } = decompileGraph(
      [node("inp", "Input"), node("block", "TransformerDecoderLayer", decoder), node("out", "Output")],
      [edge("inp", "block"), edge("block", "out")],
    );
    expect(code).toContain("block_mask = torch.ones(x.size(-2), x.size(-2), dtype=torch.bool, device=x.device).triu(1)");
    expect(code).toContain("block = self.block(x, x, tgt_mask=block_mask, memory_mask=block_mask)");
  });

  it("attends to the whole encoder output when memory is connected", () => {
    const { code } = decompileGraph(
      [node("inp", "Input"), node("enc", "Linear"), node("block", "TransformerDecoderLayer", decoder), node("out", "Output")],
      [edge("enc", "block", "memory"), edge("inp", "enc"), edge("inp", "block"), edge("block", "out")],
    );
    expect(code).toContain("block = self.block(x, enc, tgt_mask=block_mask)");
  });
});
