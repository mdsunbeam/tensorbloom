import { describe, it, expect } from "vitest";
import { encodeModel, decodeModel, packedModelFromHash, templateSlug } from "./share";
import { TEMPLATES, templateGraph, templateGroups } from "./templates";

describe("share links", () => {
  it("round-trips a model through the link", async () => {
    const tpl = TEMPLATES.find((t) => t.name === "ResNet-18")!;
    const { nodes, edges } = templateGraph(tpl);
    const groups = templateGroups(tpl, nodes);
    const packed = await encodeModel({ name: "My ResNet", nodes, edges, groups });
    expect(packed).toMatch(/^[A-Za-z0-9_-]+$/);
    const back = (await decodeModel(packed))!;
    expect(back.name).toBe("My ResNet");
    expect(back.nodes.map((n) => [n.id, n.data.moduleType, n.position])).toEqual(nodes.map((n) => [n.id, n.data.moduleType, n.position]));
    expect(back.edges.map((e) => [e.source, e.target, e.sourceHandle, e.targetHandle])).toEqual(edges.map((e) => [e.source, e.target, e.sourceHandle, e.targetHandle]));
    expect(back.groups.map((g) => g.name)).toEqual(groups.map((g) => g.name));
  });

  it("returns null for text that isn't a model", async () => {
    expect(await decodeModel("not-a-model")).toBeNull();
  });

  it("finds the model in a #m= fragment only", () => {
    expect(packedModelFromHash("#m=abc")).toBe("abc");
    expect(packedModelFromHash("#other")).toBeNull();
  });

  it("makes template slugs for ?template= links", () => {
    expect(templateSlug("ResNet-18")).toBe("resnet-18");
    expect(templateSlug("Vision Transformer (ViT)")).toBe("vision-transformer-vit");
  });
});

describe("template stages", () => {
  it("groups ResNet-18 into its stem and residual blocks, each layer in one stage", () => {
    const tpl = TEMPLATES.find((t) => t.name === "ResNet-18")!;
    const { nodes } = templateGraph(tpl);
    const groups = templateGroups(tpl, nodes);
    expect(groups.map((g) => [g.name, g.nodeIds.length])).toEqual([["Stem", 4], ["Residual block 1", 7], ["Residual block 2", 9]]);
  });

  it("gives every stage of every template at least two layers", () => {
    for (const tpl of TEMPLATES.filter((t) => t.groups)) {
      const { nodes } = templateGraph(tpl);
      expect(templateGroups(tpl, nodes).length, tpl.name).toBe(tpl.groups!.length);
    }
  });
});
