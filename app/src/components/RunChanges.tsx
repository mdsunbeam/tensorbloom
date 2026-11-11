import { useMemo } from "react";
import type { RunRecord } from "../store/run-store";
import { runDiff } from "../store/experiment-store";
import { describeChanges, type SetupDiff } from "../lib/experiment-diff";
import { useGraphStore } from "../store/graph-store";
import { focusNode } from "../lib/canvas-focus";
import { formatLoss } from "./RunControls";
import { runColor } from "./RunsPanel";

/** Two runs side by side: how the result moved, a map of the graph diff, and each change */
export default function RunChanges({ from, to }: { from: RunRecord; to: RunRecord }) {
  const diff = useMemo(() => runDiff(from, to), [from, to]);
  const changes = describeChanges(diff);
  const onCanvas = useGraphStore((s) => s.nodes);
  const present = useMemo(() => new Set(onCanvas.map((n) => n.id)), [onCanvas]);
  const result = resultChange(from, to);

  return (
    <aside className="w-[300px] flex-shrink-0 border-l border-line overflow-y-auto p-3 space-y-3 text-[11px]" aria-label="What changed">
      <div className="text-xs font-semibold text-fg">
        What changed from <RunName run={from} /> to <RunName run={to} />
      </div>

      {result && (
        <div className="rounded-md bg-surface-raised px-2.5 py-1.5">
          <div className="text-fg-muted">{result.label}</div>
          <div className="font-mono tabular-nums text-fg">
            {result.from} → {result.to}{" "}
            <span className={result.better === true ? "text-success" : result.better === false ? "text-danger" : "text-fg-muted"}>
              ({result.delta})
            </span>
          </div>
        </div>
      )}

      {changes.length === 0 ? (
        <p className="text-fg-muted">Same model and training settings.</p>
      ) : (
        <>
          <ul className="space-y-1">
            {diff.settings.map((s) => (
              <li key={s.label}>
                <span className="text-fg-secondary">{s.label}</span>{" "}
                <span className="font-mono">{s.from} → {s.to}</span>
              </li>
            ))}
            {diff.layers.map((l) => {
              const text = l.kind === "added" ? "added" : l.kind === "removed" ? "removed"
                : [l.replaced && `${l.replaced.from} → ${l.replaced.to}`, ...(l.params ?? []).map((p) => `${p.name} ${p.from} → ${p.to}`), l.renamed && `renamed from ${l.renamed.from}`].filter(Boolean).join(", ");
              const canFocus = present.has(l.id);
              return (
                <li key={l.id} className="flex gap-1.5">
                  <Dot kind={l.kind} />
                  <span className="min-w-0">
                    {canFocus ? (
                      <button type="button" className="font-medium text-fg hover:underline" onClick={() => focusNode(l.id)} title="Show on the canvas">{l.label}</button>
                    ) : <span className="font-medium text-fg">{l.label}</span>}{" "}
                    <span className="font-mono text-fg-secondary">{text}</span>
                  </span>
                </li>
              );
            })}
            {diff.connections.map((c, i) => (
              <li key={i} className="flex gap-1.5">
                <Dot kind={c.kind} />
                <span className="text-fg-secondary">{c.kind === "added" ? "Connected" : "Disconnected"} <span className="text-fg">{c.from}</span> → <span className="text-fg">{c.to}</span></span>
              </li>
            ))}
          </ul>
          {(diff.layers.length > 0 || diff.connections.length > 0) && <GraphDiffMap from={from} to={to} diff={diff} />}
          {changes.length > 1 && (
            <p className="text-fg-muted">
              {changes.length} things changed, so this can't say which one made the difference.
            </p>
          )}
        </>
      )}
    </aside>
  );
}

function RunName({ run }: { run: RunRecord }) {
  return (
    <span className="whitespace-nowrap">
      <span className="inline-block w-2 h-2 rounded-full mr-1 align-middle" style={{ background: runColor(run) }} aria-hidden="true" />
      Run {run.number}
    </span>
  );
}

const KIND_COLOR = { added: "var(--success)", removed: "var(--danger)", changed: "var(--warning)" } as const;

function Dot({ kind }: { kind: keyof typeof KIND_COLOR }) {
  return <span className="mt-[5px] w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: KIND_COLOR[kind] }} aria-hidden="true" />;
}

/** The headline result of each run and how it moved, by best validation accuracy, else best validation loss, else final loss */
function resultChange(a: RunRecord, b: RunRecord): { label: string; from: string; to: string; delta: string; better: boolean | null } | null {
  const ra = a.results;
  const rb = b.results;
  if (!ra || !rb) return null;
  if (ra.bestValAcc != null && rb.bestValAcc != null) {
    // From the numbers as shown, so 97.1% → 97.2% reads +0.1
    const d = Number((rb.bestValAcc * 100).toFixed(1)) - Number((ra.bestValAcc * 100).toFixed(1));
    return {
      label: "Best val accuracy",
      from: `${(ra.bestValAcc * 100).toFixed(1)}%`,
      to: `${(rb.bestValAcc * 100).toFixed(1)}%`,
      delta: `${d >= 0 ? "+" : ""}${d.toFixed(1)} points`,
      better: Math.abs(d) < 0.05 ? null : d > 0,
    };
  }
  const pick = (r: NonNullable<RunRecord["results"]>) => r.bestValLoss ?? r.finalTrainLoss;
  const la = pick(ra);
  const lb = pick(rb);
  const label = ra.bestValLoss != null && rb.bestValLoss != null ? "Best val loss" : "Final train loss";
  const rel = la !== 0 ? ((lb - la) / Math.abs(la)) * 100 : 0;
  return {
    label,
    from: formatLoss(la),
    to: formatLoss(lb),
    delta: `${rel >= 0 ? "+" : ""}${rel.toFixed(0)}%`,
    better: Math.abs(rel) < 0.5 ? null : lb < la,
  };
}

const NODE_W = 200;
const NODE_H = 72;

/**
 * The second run's graph in miniature, with new layers in green, changed ones in amber, and
 * layers it no longer has drawn dashed in red where they used to be.
 */
function GraphDiffMap({ from, to, diff }: { from: RunRecord; to: RunRecord; diff: SetupDiff }) {
  const layout = useMemo(() => {
    const status = new Map(diff.layers.map((l) => [l.id, l.kind]));
    for (const c of diff.connections) if (!status.has(c.target)) status.set(c.target, "changed");
    const boxes = [
      ...to.graph.nodes.filter((n) => !n.parentId).map((n) => ({ id: n.id, x: n.position.x, y: n.position.y, kind: status.get(n.id) })),
      ...from.graph.nodes.filter((n) => status.get(n.id) === "removed" && !n.parentId).map((n) => ({ id: n.id, x: n.position.x, y: n.position.y, kind: "removed" as const })),
    ];
    if (boxes.length === 0) return null;
    const centers = new Map(boxes.map((b) => [b.id, { x: b.x + NODE_W / 2, y: b.y + NODE_H / 2 }]));
    const removedEdges = new Set(diff.connections.filter((c) => c.kind === "removed").map((c) => `${c.source}>${c.target}`));
    const addedEdges = new Set(diff.connections.filter((c) => c.kind === "added").map((c) => `${c.source}>${c.target}`));
    const lines = [
      ...to.graph.edges.map((e) => ({ s: e.source, t: e.target, kind: addedEdges.has(`${e.source}>${e.target}`) || status.get(e.source) === "added" || status.get(e.target) === "added" ? "added" : undefined })),
      ...from.graph.edges
        .filter((e) => removedEdges.has(`${e.source}>${e.target}`) || status.get(e.source) === "removed" || status.get(e.target) === "removed")
        .map((e) => ({ s: e.source, t: e.target, kind: "removed" })),
    ].filter((l) => centers.has(l.s) && centers.has(l.t));
    const minX = Math.min(...boxes.map((b) => b.x));
    const minY = Math.min(...boxes.map((b) => b.y));
    const maxX = Math.max(...boxes.map((b) => b.x + NODE_W));
    const maxY = Math.max(...boxes.map((b) => b.y + NODE_H));
    return { boxes, centers, lines, minX, minY, w: maxX - minX, h: maxY - minY };
  }, [from, to, diff]);

  if (!layout) return null;
  const pad = 20;
  const viewW = 276;
  const viewH = Math.min(170, Math.max(80, (layout.h / layout.w) * viewW));
  return (
    <svg
      width={viewW}
      height={viewH}
      viewBox={`${layout.minX - pad} ${layout.minY - pad} ${layout.w + pad * 2} ${layout.h + pad * 2}`}
      preserveAspectRatio="xMidYMid meet"
      className="block rounded-md bg-surface-sunken"
      role="img"
      aria-label={`Graph of Run ${to.number}: ${diff.layers.filter((l) => l.kind === "added").length} layers added, ${diff.layers.filter((l) => l.kind === "changed").length} changed, ${diff.layers.filter((l) => l.kind === "removed").length} removed`}
    >
      {layout.lines.map((l, i) => {
        const a = layout.centers.get(l.s)!;
        const b = layout.centers.get(l.t)!;
        return (
          <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke={l.kind ? KIND_COLOR[l.kind as "added" | "removed"] : "var(--text-muted)"}
            strokeWidth={l.kind ? 10 : 6} strokeDasharray={l.kind === "removed" ? "24 16" : undefined} opacity={l.kind ? 0.9 : 0.5} />
        );
      })}
      {layout.boxes.map((b) => (
        <rect key={b.id} x={b.x} y={b.y} width={NODE_W} height={NODE_H} rx={14}
          fill={b.kind ? `color-mix(in srgb, ${KIND_COLOR[b.kind]} 30%, var(--bg-secondary))` : "var(--bg-tertiary)"}
          stroke={b.kind ? KIND_COLOR[b.kind] : "var(--border)"} strokeWidth={b.kind ? 10 : 4}
          strokeDasharray={b.kind === "removed" ? "24 16" : undefined} />
      ))}
    </svg>
  );
}
