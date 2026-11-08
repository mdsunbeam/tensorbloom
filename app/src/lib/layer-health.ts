/**
 * What training does to each layer: the stats the sidecar reports while a model trains,
 * the problems they point to, and how they're drawn on the graph.
 */

/** One layer's numbers from one captured batch, as the sidecar reports them */
export interface LayerStat {
  /** Mean and spread of the layer's outputs */
  mean?: number;
  std?: number;
  /** Share of outputs that are exactly zero */
  zero?: number;
  /** Share of units (channels or features) that output zero for every sample, for layers whose outputs can't be negative */
  dead?: number;
  /** Root-mean-square of the gradient flowing back into the layer's output */
  grad?: number | null;
  /** Size (L2 norm) of the gradient on the layer's own weights, and of the weights */
  grad_norm?: number | null;
  weight_norm?: number;
  /** NaN or infinity in the outputs or gradients */
  nonfinite?: boolean;
  hist?: { min: number; max: number; counts: number[] };
}

/** Every layer's stats from one captured batch */
export interface LayerSnapshot {
  epoch: number;
  step: number;
  totalSteps: number;
  layers: Record<string, LayerStat>;
}

/** A layer's numbers over a run; t counts epochs, so 2.5 is halfway through epoch 3 */
export interface LayerPoint {
  t: number;
  grad?: number;
  std?: number;
  dead?: number;
}

/** What each image-like layer made of one fixed sample, at the end of an epoch */
export interface FeatureMaps {
  epoch: number;
  input: { png: string; width: number; height: number } | null;
  layers: Record<string, { channels: number; size: [number, number]; png: string[] }>;
}

/** What a run kept about its layers, saved with the run */
export interface RunLayerStats {
  snapshot: LayerSnapshot;
  history: Record<string, LayerPoint[]>;
}

export type LayerOverlay = "off" | "gradients" | "activations";

export interface LayerFlag {
  kind: "nonfinite" | "dead" | "vanishing" | "exploding";
  severity: "warn" | "bad";
  /** A few words for the node */
  title: string;
  /** What it means and what to try, for the hover card */
  detail: string;
}

/** Keeps a layer's history to about this many points, thinning older ones evenly */
export const MAX_HISTORY = 240;

/** Snapshot time in epochs: the batch's place in its epoch */
export function snapshotTime(s: Pick<LayerSnapshot, "epoch" | "step" | "totalSteps">): number {
  return s.epoch - 1 + (s.totalSteps > 0 ? s.step / s.totalSteps : 0);
}

/** Adds a snapshot to each layer's history, thinning it to every other point when it grows past MAX_HISTORY */
export function appendHistory(history: Record<string, LayerPoint[]>, s: LayerSnapshot): Record<string, LayerPoint[]> {
  const t = snapshotTime(s);
  const next = { ...history };
  for (const [id, st] of Object.entries(s.layers)) {
    const point: LayerPoint = { t };
    if (st.grad != null) point.grad = st.grad;
    if (st.std != null) point.std = st.std;
    if (st.dead != null) point.dead = st.dead;
    let list = [...(next[id] ?? []), point];
    if (list.length > MAX_HISTORY) list = list.filter((_, i) => i % 2 === 1 || i === list.length - 1);
    next[id] = list;
  }
  return next;
}

function median(values: number[]): number {
  const v = [...values].sort((a, b) => a - b);
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

/** The middle gradient size across the snapshot's layers, which vanishing and exploding are judged against */
export function typicalGradient(snapshot: LayerSnapshot): number | null {
  const grads = Object.values(snapshot.layers).map((s) => s.grad).filter((g): g is number => g != null && g > 0);
  return grads.length >= 3 ? median(grads) : null;
}

/** A share as a percentage that only reads 0% or 100% when it is exactly that */
export const pct = (v: number) => {
  const p = Math.round(v * 100);
  return p === 100 && v < 1 ? ">99%" : p === 0 && v > 0 ? "<1%" : `${p}%`;
};

/**
 * Problems a layer's stats point to, worst first. Dead units are only flagged on activation layers
 * (ReLU and the like), where units die; a Dropout or pooling layer after them would only echo it.
 */
export function layerFlags(stat: LayerStat | undefined, typicalGrad: number | null, unitsCanDie = true): LayerFlag[] {
  if (!stat) return [];
  const flags: LayerFlag[] = [];
  if (stat.nonfinite) {
    flags.push({
      kind: "nonfinite", severity: "bad", title: "NaN or infinity",
      detail: "This layer's outputs or gradients contain NaN or infinite values, so training has broken down. Try a lower learning rate or turn on gradient clipping.",
    });
  }
  if (unitsCanDie && stat.dead != null && stat.dead >= 0.25) {
    flags.push({
      kind: "dead", severity: stat.dead >= 0.5 ? "bad" : "warn", title: `${pct(stat.dead)} dead`,
      detail: `${pct(stat.dead)} of this layer's units output zero for every sample in the batch. A unit stuck at zero passes no gradient back, so it stops learning. A lower learning rate, LeakyReLU, or BatchNorm before this layer can bring units back.`,
    });
  }
  if (stat.grad != null && typicalGrad != null) {
    const ratio = stat.grad / typicalGrad;
    if (stat.grad === 0 || ratio < 1e-2) {
      flags.push({
        kind: "vanishing", severity: stat.grad === 0 || ratio < 1e-3 ? "bad" : "warn", title: "Vanishing gradient",
        detail: stat.grad === 0
          ? "No gradient reaches this layer, so the layers before it can't learn through it."
          : `The gradient reaching this layer is ${formatRatio(1 / ratio)} smaller than the typical layer's, so the layers before it barely learn. Skip connections, BatchNorm, or ReLU in place of Sigmoid or Tanh usually help.`,
      });
    } else if (ratio > 1e2) {
      flags.push({
        kind: "exploding", severity: ratio > 1e3 ? "bad" : "warn", title: "Exploding gradient",
        detail: `The gradient reaching this layer is ${formatRatio(ratio)} larger than the typical layer's. Large gradients make training jumpy and can turn into NaN. Try a lower learning rate, gradient clipping, or normalization layers.`,
      });
    }
  }
  return flags.sort((a, b) => (a.severity === b.severity ? 0 : a.severity === "bad" ? -1 : 1));
}

function formatRatio(r: number): string {
  if (r >= 1000) return `${Math.round(r).toLocaleString("en-US")} times`;
  return `${r >= 10 ? Math.round(r) : r.toFixed(1)} times`;
}

/** The value the overlay colors a layer by */
export function overlayValue(stat: LayerStat | undefined, overlay: LayerOverlay): number | null {
  if (!stat || overlay === "off") return null;
  const v = overlay === "gradients" ? stat.grad : stat.std;
  return v != null && v > 0 && Number.isFinite(v) ? v : null;
}

/** The range of the overlay's values across the snapshot's layers, for the color scale */
export function overlayRange(snapshot: LayerSnapshot | null, overlay: LayerOverlay, nodeIds?: Set<string>): { min: number; max: number } | null {
  if (!snapshot) return null;
  const values = Object.entries(snapshot.layers)
    .filter(([id]) => !nodeIds || nodeIds.has(id))
    .map(([, s]) => overlayValue(s, overlay))
    .filter((v): v is number => v != null);
  if (values.length === 0) return null;
  return { min: Math.min(...values), max: Math.max(...values) };
}

/** Where a value sits in the range on a log scale, from 0 (smallest) to 1 (largest) */
export function overlayPosition(value: number, range: { min: number; max: number }): number {
  const lo = Math.log10(range.min);
  const hi = Math.log10(range.max);
  if (!(hi > lo)) return 0.5;
  return Math.min(1, Math.max(0, (Math.log10(value) - lo) / (hi - lo)));
}

/** The overlay color for a position on the scale: from a faint tint of the accent up to the full accent */
export function overlayColor(position: number): string {
  return `color-mix(in oklab, var(--accent) ${Math.round(20 + position * 80)}%, var(--bg-tertiary))`;
}

/** Text on a light or dark overlay color */
export function overlayTextColor(position: number): string {
  return position > 0.55 ? "#fff" : "var(--text-primary)";
}

/** A stat value in a few characters: 0.82, 3.1e-4, 1.2e+3 */
export function formatStat(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 0.01 && a < 1000) return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
  return v.toExponential(1).replace("e+", "e");
}

/** The overlay's short reading for a node, such as "grad 3.1e-4" or "σ 0.82 · 12% zero" */
export function overlayLabel(stat: LayerStat | undefined, overlay: LayerOverlay, unitsCanDie = true): string | null {
  if (!stat || overlay === "off") return null;
  if (overlay === "gradients") return stat.grad != null ? `grad ${formatStat(stat.grad)}` : null;
  if (stat.std == null) return null;
  const parts = [`σ ${formatStat(stat.std)}`];
  if (unitsCanDie && stat.dead != null && stat.dead > 0) parts.push(`${pct(stat.dead)} dead`);
  else if (stat.zero != null && stat.zero >= 0.05) parts.push(`${pct(stat.zero)} zero`);
  return parts.join(" · ");
}

/** Keeps only the layers that are still in the graph, e.g. after a run's model was edited */
export function presentLayers(snapshot: LayerSnapshot | null, nodeIds: Set<string>): LayerSnapshot | null {
  if (!snapshot) return null;
  const layers = Object.fromEntries(Object.entries(snapshot.layers).filter(([id]) => nodeIds.has(id)));
  return { ...snapshot, layers };
}
