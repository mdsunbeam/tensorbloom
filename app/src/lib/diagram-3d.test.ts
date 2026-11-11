import { describe, it, expect } from "vitest";
import { TEMPLATES, templateGraph } from "./templates";
import { inferShapesClient } from "./shape-inference";
import { buildDiagramScene, sceneToSvg, sceneToPdf, shortLayerLabel, blockSize, shade } from "./diagram-3d";
import type { NodeData } from "../store/graph-store";

function sceneFor(name: string, opts = {}) {
  const tpl = TEMPLATES.find((t) => t.name === name)!;
  const { nodes, edges } = templateGraph(tpl);
  const { shapes } = inferShapesClient(nodes, edges);
  return { scene: buildDiagramScene(nodes, edges, shapes, [], opts), nodes };
}

const texts = (s: ReturnType<typeof sceneFor>["scene"]) =>
  s.prims.flatMap((p) => (p.kind === "text" ? [p.text] : []));

describe("diagram-3d", () => {
  it("draws nothing for an empty graph", () => {
    const scene = buildDiagramScene([], [], {}, [], {});
    expect(scene.blockCount).toBe(0);
  });

  it("folds BN, ReLU and pooling onto the conv before them", () => {
    const { scene, nodes } = sceneFor("ResNet-18");
    expect(scene.blockCount).toBeLessThan(nodes.length * 0.6);
    const t = texts(scene);
    expect(t).toContain("Conv 7×7 /2");
    expect(t).toContain("+ BN · ReLU · MaxPool");
    expect(t).toContain("3×32×32");
  });

  it("keeps every layer as its own block when folding is off", () => {
    const folded = sceneFor("ResNet-18").scene.blockCount;
    const unfolded = sceneFor("ResNet-18", { foldLayers: false }).scene.blockCount;
    expect(unfolded).toBeGreaterThan(folded);
  });

  it("stacks identical layers into one block marked ×N", () => {
    expect(texts(sceneFor("BERT Base").scene)).toContain("Encoder ×12");
    expect(texts(sceneFor("BERT Base", { stackRepeats: false }).scene)).not.toContain("Encoder ×12");
  });

  it("leaves out loss layers", () => {
    expect(texts(sceneFor("LeNet-5").scene).some((t) => /Loss/.test(t))).toBe(false);
  });

  it("renders every template in every view without throwing", () => {
    for (const tpl of TEMPLATES) {
      const { nodes, edges } = templateGraph(tpl);
      const { shapes } = inferShapesClient(nodes, edges);
      for (const projection of ["oblique", "isometric", "flat"] as const) {
        const scene = buildDiagramScene(nodes, edges, shapes, [], { projection });
        expect(scene.blockCount).toBeGreaterThan(0);
        expect(scene.width).toBeGreaterThan(0);
        for (const p of scene.prims) {
          const pts = p.kind === "text" ? [[p.x, p.y]] : p.pts;
          for (const [x, y] of pts) expect(Number.isFinite(x) && Number.isFinite(y)).toBe(true);
        }
      }
    }
  });

  it("writes an SVG with the figure's size", () => {
    const { scene } = sceneFor("LeNet-5");
    const svg = sceneToSvg(scene);
    expect(svg).toMatch(new RegExp(`^<svg [^>]*width="${scene.width}" height="${scene.height}"`));
    expect(svg).toContain("Conv 5×5");
  });

  it("writes a PDF whose cross-reference table points at its objects", () => {
    const pdf = sceneToPdf(sceneFor("ResNet-18").scene);
    expect(pdf.startsWith("%PDF-1.4")).toBe(true);
    expect(/[^\x00-\x7f]/.test(pdf)).toBe(false);
    const xrefAt = Number(pdf.match(/startxref\n(\d+)/)![1]);
    expect(pdf.slice(xrefAt, xrefAt + 4)).toBe("xref");
    const offsets = [...pdf.slice(xrefAt).matchAll(/(\d{10}) 00000 n/g)].map((m) => Number(m[1]));
    offsets.forEach((off, i) => expect(pdf.slice(off, off + `${i + 1} 0 obj`.length)).toBe(`${i + 1} 0 obj`));
    const len = Number(pdf.match(/\/Length (\d+)/)![1]);
    const start = pdf.indexOf("stream\n") + "stream\n".length;
    expect(pdf.slice(start + len, start + len + 10)).toBe("\nendstream");
    expect(pdf).toContain("(Conv 7\\3277 /2) Tj");
  });

  it("names layers the way papers do", () => {
    const d = (moduleType: string, params: Record<string, unknown> = {}, label = moduleType) =>
      ({ moduleType, params, label, category: "conv" }) as NodeData;
    expect(shortLayerLabel(d("Conv2d", { kernel_size: 3, stride: 2 }))).toBe("Conv 3×3 /2");
    expect(shortLayerLabel(d("Linear"))).toBe("FC");
    expect(shortLayerLabel(d("Input", {}, "RGB frames"))).toBe("RGB frames");
  });

  it("sizes blocks by channels and spatial size", () => {
    const small = blockSize([1, 16, 8, 8], "conv");
    const big = blockSize([1, 256, 64, 64], "conv");
    expect(big.dx).toBeGreaterThan(small.dx);
    expect(big.h).toBeGreaterThan(small.h);
    expect(blockSize([1, 10], "linear").dx).toBeLessThan(blockSize([1, 64, 8, 8], "conv").dx + 1);
  });

  it("shades colors toward white and black", () => {
    expect(shade("#808080", 1)).toBe("#FFFFFF");
    expect(shade("#808080", -1)).toBe("#000000");
  });
});
