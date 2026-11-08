import { describe, it, expect } from "vitest";
import { TEMPLATES, templateGraph } from "./templates";

describe("template quick runs", () => {
  const builtIn = TEMPLATES.filter((t) => t.trainingDefaults?.dataset && t.trainingDefaults.dataset.type !== "ImageFolder");

  it("every template on a built-in dataset starts on a subset", () => {
    for (const t of builtIn) expect(t.trainingDefaults?.dataset?.maxSamples, t.name).toBeGreaterThan(0);
  });

  it("puts the subset size and other data settings on the Data node", () => {
    const tpl = TEMPLATES.find((t) => t.name === "nanoGPT")!;
    const data = templateGraph(tpl).nodes.find((n) => n.data.moduleType === "Data")!;
    expect(data.data.params.max_samples).toBe(512);
    expect(data.data.params.max_length).toBe(64);
    expect(data.data.params.source).toBe("TinyShakespeare");
  });
});
