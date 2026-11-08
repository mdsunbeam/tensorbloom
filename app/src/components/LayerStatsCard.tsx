import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useLayerStatsStore } from "../store/layer-stats-store";
import { formatStat, pct, type LayerFlag, type LayerPoint, type LayerStat } from "../lib/layer-health";
import { overlayScale } from "../lib/use-layer-overlay";

const WIDTH = 248;

/** What training did to one layer: its problems, gradient over the run, output histogram, and feature maps */
export default function LayerStatsCard({ nodeId, label, stat, flags, unitsCanDie }: { nodeId: string; label: string; stat: LayerStat; flags: LayerFlag[]; unitsCanDie: boolean }) {
  const snapshot = useLayerStatsStore((s) => s.snapshot);
  const source = useLayerStatsStore((s) => s.source);
  const history = useLayerStatsStore((s) => s.history[nodeId]);
  const maps = useLayerStatsStore((s) => s.maps);
  const layerMaps = maps?.layers[nodeId];
  const typical = overlayScale(snapshot, "gradients").typicalGrad;

  const when = snapshot ? `epoch ${snapshot.epoch}` : "";
  const from = source?.kind === "run" ? `Run ${source.runNumber}, ${when}` : `Training now, ${when}`;

  return (
    <div
      className="rounded-lg border border-line bg-surface text-fg shadow-xl p-3 space-y-3 text-[11px] leading-snug"
      style={{ width: WIDTH }}
      role="tooltip"
    >
      <div>
        <div className="text-xs font-semibold truncate">{label}</div>
        <div className="text-fg-muted">{from}</div>
      </div>

      {flags.length > 0 && (
        <ul className="space-y-1.5">
          {flags.map((f) => (
            <li key={f.kind} className={`rounded-md px-2 py-1.5 ${f.severity === "bad" ? "bg-danger-subtle" : "bg-warning-subtle"}`}>
              <div className={`font-semibold ${f.severity === "bad" ? "text-danger" : "text-warning"}`}>{f.title}</div>
              <div className="text-fg-secondary">{f.detail}</div>
            </li>
          ))}
        </ul>
      )}

      {stat.grad != null && (
        <section>
          <div className="flex items-baseline justify-between">
            <span className="font-medium text-fg-secondary">Gradient</span>
            <span className="font-mono tabular-nums">{formatStat(stat.grad)}</span>
          </div>
          <p className="text-fg-muted">
            How hard training pushes on this layer's output.
            {typical != null && <> The typical layer gets {formatStat(typical)}.</>}
          </p>
          {history && <Sparkline points={history} pick={(p) => p.grad} log label="Gradient over the run" />}
        </section>
      )}

      {stat.std != null && (
        <section>
          <div className="flex items-baseline justify-between">
            <span className="font-medium text-fg-secondary">Outputs</span>
            <span className="font-mono tabular-nums text-fg-secondary">
              mean {formatStat(stat.mean ?? 0)} · σ {formatStat(stat.std)}
            </span>
          </div>
          {stat.hist && <Histogram hist={stat.hist} />}
          <p className="text-fg-muted">
            {unitsCanDie && stat.dead != null
              ? `${pct(stat.dead)} of units gave zero for every sample in the batch.`
              : stat.zero != null && stat.zero > 0.01 ? `${pct(stat.zero)} of outputs were exactly zero, for one batch.` : "How this layer's outputs were spread, for one batch."}
          </p>
        </section>
      )}

      {layerMaps && (
        <section>
          <div className="flex items-baseline justify-between">
            <span className="font-medium text-fg-secondary">Feature maps</span>
            <span className="text-fg-muted tabular-nums">{layerMaps.size[0]}×{layerMaps.size[1]}</span>
          </div>
          <div className="flex gap-2 mt-1 items-start">
            {maps?.input && (
              <figure className="flex-shrink-0 text-center">
                <img src={`data:image/png;base64,${maps.input.png}`} alt="The sample image" className="w-12 h-12 rounded object-contain bg-surface-raised" style={{ imageRendering: "pixelated" }} />
                <figcaption className="text-fg-muted mt-0.5">Input</figcaption>
              </figure>
            )}
            <div className="grid grid-cols-4 gap-1 flex-1">
              {layerMaps.png.map((png, i) => (
                <img key={i} src={`data:image/png;base64,${png}`} alt={`Channel ${i + 1}`} title={`Channel ${i + 1}`}
                  className="w-full aspect-square rounded-sm object-cover bg-surface-raised" style={{ imageRendering: "pixelated" }} />
              ))}
            </div>
          </div>
          <p className="text-fg-muted mt-1">
            {layerMaps.png.length < layerMaps.channels ? `The first ${layerMaps.png.length} of ${layerMaps.channels} channels` : `All ${layerMaps.channels} channels`} for one validation sample, after epoch {maps?.epoch}. Bright is where the channel responds most.
          </p>
        </section>
      )}
    </div>
  );
}

/** A line over the run, one point per captured batch */
function Sparkline({ points, pick, log, label }: { points: LayerPoint[]; pick: (p: LayerPoint) => number | undefined; log?: boolean; label: string }) {
  const data = points
    .map((p) => ({ x: p.t, y: pick(p) }))
    .filter((p): p is { x: number; y: number } => p.y != null && (!log || p.y > 0));
  if (data.length < 2) return null;
  const w = WIDTH - 24;
  const h = 34;
  const ys = data.map((d) => (log ? Math.log10(d.y) : d.y));
  const xMin = data[0].x;
  const xMax = data[data.length - 1].x;
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const px = (x: number) => (xMax > xMin ? ((x - xMin) / (xMax - xMin)) * (w - 4) + 2 : w / 2);
  const py = (y: number) => (yMax > yMin ? h - 3 - ((y - yMin) / (yMax - yMin)) * (h - 6) : h / 2);
  const path = data.map((d, i) => `${i ? "L" : "M"}${px(d.x).toFixed(1)},${py(ys[i]).toFixed(1)}`).join("");
  const last = data[data.length - 1];
  return (
    <svg width={w} height={h} className="mt-1 block" role="img" aria-label={`${label}: from ${formatStat(data[0].y)} to ${formatStat(last.y)}`}>
      <line x1={0} x2={w} y1={h - 0.5} y2={h - 0.5} stroke="var(--border)" />
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth={1.5} strokeLinejoin="round" />
      <circle cx={px(last.x)} cy={py(ys[ys.length - 1])} r={2.5} fill="var(--accent)" />
    </svg>
  );
}

/** How the layer's outputs were spread over one batch */
function Histogram({ hist }: { hist: NonNullable<LayerStat["hist"]> }) {
  const w = WIDTH - 24;
  const h = 36;
  const max = Math.max(...hist.counts, 1);
  const bw = w / hist.counts.length;
  // Where zero falls, when it's inside the range
  const zeroX = hist.min < 0 && hist.max > 0 ? ((0 - hist.min) / (hist.max - hist.min)) * w : null;
  return (
    <div className="mt-1">
      <svg width={w} height={h} className="block" role="img" aria-label={`Histogram of outputs from ${formatStat(hist.min)} to ${formatStat(hist.max)}`}>
        {hist.counts.map((c, i) => {
          const bh = c > 0 ? Math.max(1, (c / max) * (h - 2)) : 0;
          return <rect key={i} x={i * bw + 0.5} y={h - bh} width={Math.max(1, bw - 1)} height={bh} fill="var(--accent)" opacity={0.75} />;
        })}
        {zeroX != null && <line x1={zeroX} x2={zeroX} y1={0} y2={h} stroke="var(--text-muted)" strokeDasharray="2 2" />}
      </svg>
      <div className="flex justify-between font-mono tabular-nums text-fg-muted">
        <span>{formatStat(hist.min)}</span>
        {zeroX != null && <span>0</span>}
        <span>{formatStat(hist.max)}</span>
      </div>
    </div>
  );
}

/** Places the card beside a layer, above everything on the canvas, kept inside the window */
export function FloatingLayerCard({ anchor, children }: { anchor: HTMLElement; children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  useLayoutEffect(() => {
    // Follows the layer while the canvas pans or zooms under the pointer
    let frame = 0;
    const place = () => {
      const a = anchor.getBoundingClientRect();
      const card = ref.current?.getBoundingClientRect();
      const w = card?.width ?? WIDTH;
      const h = card?.height ?? 0;
      const gap = 12;
      // Right of the layer, or left when there's no room; level with its top, moved up to stay in the window
      const left = a.right + gap + w <= window.innerWidth - 8 ? a.right + gap : Math.max(8, a.left - gap - w);
      const top = Math.max(8, Math.min(a.top, window.innerHeight - 8 - h));
      setPos((p) => (p && p.left === left && p.top === top ? p : { left, top }));
      frame = requestAnimationFrame(place);
    };
    place();
    return () => cancelAnimationFrame(frame);
  }, [anchor]);
  return createPortal(
    <div ref={ref} className="fixed z-50 pointer-events-none" style={{ left: pos?.left ?? -9999, top: pos?.top ?? 0 }}>
      {children}
    </div>,
    document.body,
  );
}
