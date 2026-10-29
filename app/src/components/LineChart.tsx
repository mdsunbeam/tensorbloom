import { useState, useEffect, useRef } from "react";

/** Round numbers for axis ticks (1, 2 or 5 times a power of ten) covering [min, max] */
function niceTicks(min: number, max: number, count = 4): number[] {
  const span = max - min || Math.abs(max) || 1;
  const raw = span / count;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? 10 * pow;
  const start = Math.floor(min / step) * step;
  const ticks: number[] = [];
  for (let t = start; t <= max + step * 0.5; t += step) ticks.push(Number(t.toPrecision(12)));
  return ticks;
}

/** Line chart drawn at the container's real pixel size, with a hover readout in its header */
export default function LineChart({
  title,
  series,
  formatY,
  formatTick,
  zeroBased = false,
  maxY,
}: {
  title: string;
  series: Array<{ data: Array<{ x: number; y: number }>; color: string; label: string; dashed?: boolean }>;
  formatY: (v: number) => string;
  formatTick?: (v: number) => string;
  zeroBased?: boolean;
  maxY?: number;
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hoverX, setHoverX] = useState<number | null>(null);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: Math.floor(width), h: Math.floor(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const allPoints = series.flatMap((s) => s.data);
  const xs = Array.from(new Set(allPoints.map((p) => p.x))).sort((a, b) => a - b);
  const shownX = hoverX ?? xs[xs.length - 1];
  const tickFmt = formatTick ?? formatY;

  // Header doubles as the legend: each series' value at the hovered epoch, or the latest one
  const header = (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 mb-1 text-[11px] tabular-nums min-w-0">
      <span className="font-semibold whitespace-nowrap" style={{ color: "var(--text-primary)" }}>{title}</span>
      {series.map((s) => {
        const p = s.data.find((d) => d.x === shownX) ?? s.data[s.data.length - 1];
        return (
          <span key={s.label} className="flex items-center gap-1 whitespace-nowrap" style={{ color: "var(--text-muted)" }}>
            <svg width="14" height="4" className="self-center flex-shrink-0" aria-hidden="true">
              <line x1="1" y1="2" x2="13" y2="2" stroke={s.color} strokeWidth="2" strokeLinecap="round" strokeDasharray={s.dashed ? "3 3" : undefined} />
            </svg>
            {s.label} <span style={{ color: "var(--text-primary)" }}>{p ? formatY(p.y) : "\u2013"}</span>
          </span>
        );
      })}
      {hoverX != null && <span className="ml-auto flex-shrink-0" style={{ color: "var(--text-muted)" }}>Epoch {hoverX}</span>}
    </div>
  );

  const { w: W, h: H } = size;
  const PAD = { top: 6, right: 10, bottom: 20, left: 46 };
  const plotW = Math.max(1, W - PAD.left - PAD.right);
  const plotH = Math.max(1, H - PAD.top - PAD.bottom);

  let chart: React.ReactNode = null;
  if (allPoints.length > 0 && W > 0 && H > 0) {
    const xMin = xs[0];
    const xMax = Math.max(xs[xs.length - 1], xMin + 1);
    let lo = Math.min(...allPoints.map((p) => p.y));
    let hi = Math.max(...allPoints.map((p) => p.y));
    if (zeroBased) lo = Math.min(0, lo);
    if (hi === lo) { hi = lo + (Math.abs(lo) || 1) * 0.1; lo = zeroBased ? lo : lo - (hi - lo); }
    const yTicks = niceTicks(lo, hi, plotH < 90 ? 2 : 4);
    const yMin = yTicks[0];
    const yMax = maxY != null ? Math.min(maxY, yTicks[yTicks.length - 1]) : yTicks[yTicks.length - 1];
    const ticks = yTicks.filter((t) => t <= yMax + 1e-12);
    const yRange = yMax - yMin || 1;

    const xScale = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const yScale = (v: number) => PAD.top + plotH - ((v - yMin) / yRange) * plotH;
    const xEvery = Math.max(1, Math.ceil(xs.length / Math.max(2, Math.floor(plotW / 48))));
    const xLabels = xs.filter((_, i) => i % xEvery === 0 || i === xs.length - 1);
    const showDots = xs.length <= 30;

    const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const fx = xMin + ((e.clientX - rect.left - PAD.left) / plotW) * (xMax - xMin);
      let best = xs[0];
      for (const x of xs) if (Math.abs(x - fx) < Math.abs(best - fx)) best = x;
      setHoverX(best);
    };

    chart = (
      <svg width={W} height={H} className="block" onMouseMove={onMove} onMouseLeave={() => setHoverX(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={yScale(t)} y2={yScale(t)} stroke="var(--chart-grid)" strokeWidth="1" />
            <text x={PAD.left - 6} y={yScale(t) + 4} textAnchor="end" fill="var(--text-muted)" fontSize="11" className="tabular-nums">{tickFmt(t)}</text>
          </g>
        ))}
        {xLabels.map((x) => (
          <text key={x} x={xScale(x)} y={H - 4} textAnchor="middle" fill="var(--text-muted)" fontSize="11" className="tabular-nums">{x}</text>
        ))}
        {hoverX != null && (
          <line x1={xScale(hoverX)} x2={xScale(hoverX)} y1={PAD.top} y2={PAD.top + plotH} stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="3,3" />
        )}
        {series.map((s) => {
          if (s.data.length === 0) return null;
          const path = s.data.map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.x).toFixed(1)} ${yScale(Math.min(p.y, yMax)).toFixed(1)}`).join(" ");
          const hovered = hoverX != null ? s.data.find((p) => p.x === hoverX) : undefined;
          return (
            <g key={s.label}>
              <path d={path} fill="none" stroke={s.color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" strokeDasharray={s.dashed ? "5 4" : undefined} />
              {showDots && s.data.map((p) => (
                <circle key={p.x} cx={xScale(p.x)} cy={yScale(Math.min(p.y, yMax))} r="2.5" fill={s.color} />
              ))}
              {hovered && <circle cx={xScale(hovered.x)} cy={yScale(Math.min(hovered.y, yMax))} r="4.5" fill={s.color} stroke="var(--bg-secondary)" strokeWidth="2" />}
            </g>
          );
        })}
      </svg>
    );
  }

  return (
    <div className="min-h-0 min-w-0 flex flex-col">
      {header}
      <div ref={boxRef} className="flex-1 min-h-0 relative">
        {allPoints.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center text-[11px]" style={{ color: "var(--text-muted)" }}>Waiting for the first epoch…</div>
        ) : chart}
      </div>
    </div>
  );
}
