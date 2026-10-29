import { describe, it, expect } from "vitest";
import { MODULE_REGISTRY } from "./registry";
import { paramHelp, pytorchDocsUrl, torchName } from "./param-help";

describe("paramHelp", () => {
  it("explains every parameter of every layer", () => {
    const missing: string[] = [];
    for (const [type, def] of Object.entries(MODULE_REGISTRY)) {
      for (const p of def.params) if (!paramHelp(type, p.name)) missing.push(`${type}.${p.name}`);
    }
    expect(missing).toEqual([]);
  });

  it("uses a layer's own meaning where a name means something else", () => {
    expect(paramHelp("Dropout", "p")).toMatch(/zeroed/);
    expect(paramHelp("PairwiseDistance", "p")).toMatch(/norm/);
  });
});

describe("PyTorch links", () => {
  it("points layers at torch.nn and functions at torch", () => {
    expect(torchName("Conv2d")).toBe("torch.nn.Conv2d");
    expect(pytorchDocsUrl("Concatenate")).toBe("https://pytorch.org/docs/stable/generated/torch.cat.html");
    expect(pytorchDocsUrl("Data")).toBeNull();
  });
});
