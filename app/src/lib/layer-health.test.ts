import { describe, it, expect } from "vitest";
import {
  appendHistory,
  formatStat,
  layerFlags,
  overlayLabel,
  overlayPosition,
  overlayRange,
  snapshotTime,
  typicalGradient,
  MAX_HISTORY,
  type LayerSnapshot,
} from "./layer-health";

const snap = (layers: LayerSnapshot["layers"], epoch = 1, step = 0, totalSteps = 100): LayerSnapshot => ({ epoch, step, totalSteps, layers });

describe("layer flags", () => {
  it("flags NaN and infinity first", () => {
    const flags = layerFlags({ nonfinite: true, dead: 0.3 }, null);
    expect(flags.map((f) => f.kind)).toEqual(["nonfinite", "dead"]);
    expect(flags[0].severity).toBe("bad");
  });

  it("flags dead units by how many there are", () => {
    expect(layerFlags({ dead: 0.1 }, null)).toEqual([]);
    expect(layerFlags({ dead: 0.3 }, null)[0]).toMatchObject({ kind: "dead", severity: "warn", title: "30% dead" });
    expect(layerFlags({ dead: 0.7 }, null)[0]).toMatchObject({ kind: "dead", severity: "bad" });
    // A Dropout after a dead ReLU only echoes it
    expect(layerFlags({ dead: 0.7 }, null, false)).toEqual([]);
  });

  it("never rounds a share to 0% or 100% unless it is", () => {
    expect(layerFlags({ dead: 0.998 }, null)[0].title).toBe(">99% dead");
    expect(overlayLabel({ std: 1, zero: 0.06, dead: 0.004 }, "activations")).toBe("σ 1.00 · <1% dead");
    expect(overlayLabel({ std: 1, zero: 0.06, dead: 0.5 }, "activations", false)).toBe("σ 1.00 · 6% zero");
  });

  it("judges gradients against the typical layer's", () => {
    expect(layerFlags({ grad: 1e-3 }, 1e-3)).toEqual([]);
    expect(layerFlags({ grad: 5e-6 }, 1e-3)[0]).toMatchObject({ kind: "vanishing", severity: "warn" });
    expect(layerFlags({ grad: 1e-7 }, 1e-3)[0]).toMatchObject({ kind: "vanishing", severity: "bad" });
    expect(layerFlags({ grad: 0 }, 1e-3)[0]).toMatchObject({ kind: "vanishing", severity: "bad" });
    expect(layerFlags({ grad: 0.5 }, 1e-3)[0]).toMatchObject({ kind: "exploding", severity: "warn" });
    expect(layerFlags({ grad: 5 }, 1e-3)[0]).toMatchObject({ kind: "exploding", severity: "bad" });
  });

  it("doesn't judge gradients without enough layers to compare", () => {
    expect(typicalGradient(snap({ a: { grad: 1 }, b: { grad: 1e-9 } }))).toBeNull();
    expect(typicalGradient(snap({ a: { grad: 1 }, b: { grad: 2 }, c: { grad: 3 }, d: {} }))).toBe(2);
    expect(layerFlags({ grad: 1e-9 }, null)).toEqual([]);
  });
});

describe("overlay scale", () => {
  const s = snap({ a: { grad: 1e-4, std: 1 }, b: { grad: 1e-2, std: 0.5 }, c: { grad: null, std: 2 } });

  it("spans the layers that have a value", () => {
    expect(overlayRange(s, "gradients")).toEqual({ min: 1e-4, max: 1e-2 });
    expect(overlayRange(s, "activations")).toEqual({ min: 0.5, max: 2 });
    expect(overlayRange(s, "gradients", new Set(["a"]))).toEqual({ min: 1e-4, max: 1e-4 });
  });

  it("places values on a log scale", () => {
    const range = { min: 1e-4, max: 1e-2 };
    expect(overlayPosition(1e-4, range)).toBe(0);
    expect(overlayPosition(1e-3, range)).toBeCloseTo(0.5);
    expect(overlayPosition(1e-2, range)).toBe(1);
    expect(overlayPosition(1, { min: 1, max: 1 })).toBe(0.5);
  });

  it("labels a layer for the overlay", () => {
    expect(overlayLabel({ grad: 3.1e-4 }, "gradients")).toBe("grad 3.1e-4");
    expect(overlayLabel({ std: 0.82, dead: 0.12 }, "activations")).toBe("σ 0.82 · 12% dead");
    expect(overlayLabel({ std: 0.82, zero: 0.5 }, "activations")).toBe("σ 0.82 · 50% zero");
    expect(overlayLabel({ std: 0.82 }, "off")).toBeNull();
  });
});

describe("history", () => {
  it("records time in epochs", () => {
    expect(snapshotTime({ epoch: 3, step: 50, totalSteps: 100 })).toBe(2.5);
  });

  it("thins long histories but keeps the latest point", () => {
    let h = {};
    for (let i = 0; i <= MAX_HISTORY; i++) h = appendHistory(h, snap({ a: { grad: i + 1 } }, 1, i, 1000));
    const points = (h as Record<string, { t: number; grad?: number }[]>).a;
    expect(points.length).toBeLessThanOrEqual(MAX_HISTORY);
    expect(points[points.length - 1].grad).toBe(MAX_HISTORY + 1);
  });
});

describe("formatStat", () => {
  it("keeps numbers short", () => {
    expect(formatStat(0)).toBe("0");
    expect(formatStat(0.8234)).toBe("0.82");
    expect(formatStat(12.34)).toBe("12.3");
    expect(formatStat(0.00031)).toBe("3.1e-4");
    expect(formatStat(12345)).toBe("1.2e4");
  });
});
