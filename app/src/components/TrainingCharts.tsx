import { useMemo, useState, useEffect } from "react";
import { useTrainingStore } from "../store/training-store";

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function TrainingCharts({ onSwitchToLogs }: { onSwitchToLogs: () => void }) {
  const metrics = useTrainingStore((s) => s.metrics);
  const status = useTrainingStore((s) => s.status);
  const currentEpoch = useTrainingStore((s) => s.currentEpoch);
  const config = useTrainingStore((s) => s.config);
  const error = useTrainingStore((s) => s.error);
  const startedAt = useTrainingStore((s) => s.startedAt);
  const logs = useTrainingStore((s) => s.logs);

  // Elapsed timer
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (status !== "running" && status !== "starting") return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [status]);

  const elapsed = startedAt ? now - startedAt : 0;
  const eta = currentEpoch > 0 && startedAt
    ? (elapsed / currentEpoch) * (config.epochs - currentEpoch)
    : 0;

  const epochData = useMemo(() => {
    const map = new Map<number, { losses: number[]; valLosses: number[]; accs: number[]; valAccs: number[]; lrs: number[] }>();
    for (const m of metrics) {
      if (!map.has(m.epoch)) map.set(m.epoch, { losses: [], valLosses: [], accs: [], valAccs: [], lrs: [] });
      const e = map.get(m.epoch)!;
      e.losses.push(m.trainLoss);
      if (m.valLoss != null) e.valLosses.push(m.valLoss);
      if (m.trainAcc != null) e.accs.push(m.trainAcc);
      if (m.valAcc != null) e.valAccs.push(m.valAcc);
      e.lrs.push(m.lr);
    }
    return Array.from(map.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([epoch, d]) => ({
        epoch,
        trainLoss: d.losses[d.losses.length - 1],
        valLoss: d.valLosses.length > 0 ? d.valLosses[d.valLosses.length - 1] : undefined,
        trainAcc: d.accs.length > 0 ? d.accs[d.accs.length - 1] : undefined,
        valAcc: d.valAccs.length > 0 ? d.valAccs[d.valAccs.length - 1] : undefined,
        lr: d.lrs[d.lrs.length - 1],
      }));
  }, [metrics]);

  if (metrics.length === 0 && (status === "idle" || status === "starting")) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2.5" style={{ background: "var(--bg-tertiary)" }}>
            {status === "starting" ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            )}
          </div>
          <div className="text-xs font-medium mb-1" style={{ color: "var(--text-muted)" }}>
            {status === "starting" ? "Initializing PyTorch..." : "Ready to train"}
          </div>
          <div className="text-[11px]" style={{ color: "var(--text-muted)", opacity: 0.6 }}>
            {status === "starting"
              ? "Loading model and preparing dataset"
              : "Configure hyperparameters in the Training tab, then click Start Training"}
          </div>
        </div>
      </div>
    );
  }

  const hasAcc = epochData.some((d) => d.trainAcc != null);
  const hasLrVariation = epochData.length > 1 && new Set(epochData.map((d) => d.lr)).size > 1;
  const latestGpu = metrics.length > 0 ? metrics[metrics.length - 1].gpu : undefined;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Metrics strip */}
      <div
        className="flex items-center gap-4 px-3 py-1.5 flex-shrink-0 text-[10px]"
        style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-tertiary)" }}
      >
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full" style={{
            background: status === "running" ? "#10b981" : status === "starting" ? "#f59e0b" : status === "finished" ? "var(--accent)" : status === "error" ? "var(--danger)" : "var(--text-muted)",
          }} />
          <span className="font-semibold uppercase" style={{ color: "var(--text-secondary)" }}>{status}</span>
        </div>
        <span className="tabular-nums" style={{ color: "var(--text-muted)" }}>Epoch {currentEpoch}/{config.epochs}</span>
        {epochData.length > 0 && (
          <>
            <span style={{ color: "var(--text-secondary)" }}>Loss: {epochData[epochData.length - 1].trainLoss.toFixed(4)}</span>
            {epochData[epochData.length - 1].trainAcc != null && (
              <span style={{ color: "var(--text-secondary)" }}>Acc: {(epochData[epochData.length - 1].trainAcc! * 100).toFixed(1)}%</span>
            )}
          </>
        )}
        {startedAt && (
          <>
            <span className="tabular-nums" style={{ color: "var(--text-muted)" }}>{formatTime(elapsed)}</span>
            {eta > 0 && status === "running" && <span className="tabular-nums" style={{ color: "var(--text-muted)" }}>~{formatTime(eta)}</span>}
          </>
        )}
        {error && <span style={{ color: "var(--danger)" }}>{error}</span>}
        {logs.length > 0 && (
          <button
            onClick={onSwitchToLogs}
            className="ml-auto text-[9px] px-1.5 py-0.5 rounded"
            style={{ color: "var(--text-muted)", background: "var(--bg-secondary)" }}
          >
            Logs ({logs.length})
          </button>
        )}
      </div>

      {/* Charts grid */}
      <div className="flex-1 min-h-0 grid gap-2 p-3" style={{ gridTemplateColumns: hasAcc ? "1fr 1fr" : "1fr" }}>
        <div className="min-h-0">
          <LineChart
            title="Loss"
            series={[
              { data: epochData.map((d) => ({ x: d.epoch, y: d.trainLoss })), color: "var(--accent)", label: "Train" },
              ...(epochData.some((d) => d.valLoss != null)
                ? [{ data: epochData.filter((d) => d.valLoss != null).map((d) => ({ x: d.epoch, y: d.valLoss! })), color: "#f97316", label: "Val" }]
                : []),
            ]}
          />
        </div>
        {hasAcc && (
          <div className="min-h-0">
            <LineChart
              title="Accuracy"
              series={[
                { data: epochData.filter((d) => d.trainAcc != null).map((d) => ({ x: d.epoch, y: d.trainAcc! })), color: "#10b981", label: "Train" },
                ...(epochData.some((d) => d.valAcc != null)
                  ? [{ data: epochData.filter((d) => d.valAcc != null).map((d) => ({ x: d.epoch, y: d.valAcc! })), color: "#ec4899", label: "Val" }]
                  : []),
              ]}
              formatY={(v) => `${(v * 100).toFixed(1)}%`}
            />
          </div>
        )}
        {hasLrVariation && (
          <div className="min-h-0">
            <LineChart
              title="Learning Rate"
              series={[
                { data: epochData.map((d) => ({ x: d.epoch, y: d.lr })), color: "#a78bfa", label: "LR" },
              ]}
              formatY={(v) => v.toExponential(1)}
            />
          </div>
        )}
        {latestGpu && (
          <div className="min-h-0 flex flex-col justify-center px-3 py-2 rounded" style={{ background: "var(--bg-primary)" }}>
            <div className="font-semibold uppercase tracking-wider mb-2 text-[9px]" style={{ color: "var(--text-muted)" }}>GPU Memory</div>
            <div className="space-y-1 text-[10px]" style={{ color: "var(--text-secondary)" }}>
              <div className="flex justify-between">
                <span>Allocated</span>
                <span className="tabular-nums">{latestGpu.allocated_mb.toFixed(0)} MB</span>
              </div>
              <div className="flex justify-between">
                <span>Peak</span>
                <span className="tabular-nums">{latestGpu.max_allocated_mb.toFixed(0)} MB</span>
              </div>
              <div className="mt-1 h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-tertiary)" }}>
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.min(100, (latestGpu.reserved_mb / ((latestGpu as Record<string, number>).total_mb || (latestGpu.reserved_mb + 512))) * 100)}%`,
                    background: "var(--accent)",
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Epoch history — compact horizontal row */}
      {epochData.length > 0 && (
        <div className="flex-shrink-0 px-3 pb-2">
          <div className="flex items-center gap-3 overflow-x-auto text-[9px] tabular-nums" style={{ color: "var(--text-muted)" }}>
            <span className="font-semibold uppercase tracking-wider flex-shrink-0" style={{ fontSize: "8px" }}>History</span>
            {epochData.map((d) => (
              <span key={d.epoch} className="flex-shrink-0" style={{ color: "var(--text-secondary)" }}>
                E{d.epoch}: {d.trainLoss.toFixed(3)}
                {d.trainAcc != null && ` / ${(d.trainAcc * 100).toFixed(0)}%`}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Reusable SVG line chart with multiple series */
function LineChart({
  title,
  series,
  formatY,
}: {
  title: string;
  series: Array<{ data: Array<{ x: number; y: number }>; color: string; label: string }>;
  formatY?: (v: number) => string;
}) {
  const allPoints = series.flatMap((s) => s.data);
  if (allPoints.length < 1) {
    return <div className="w-full h-full flex items-center justify-center text-[11px]" style={{ color: "var(--text-muted)" }}>Waiting...</div>;
  }

  const W = 500, H = 200;
  const PAD = { top: 22, right: 12, bottom: 22, left: 42 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const xMin = Math.min(...allPoints.map((p) => p.x));
  const xMax = Math.max(...allPoints.map((p) => p.x), xMin + 1);
  const yMin = Math.min(...allPoints.map((p) => p.y), 0);
  const yMax = Math.max(...allPoints.map((p) => p.y), 0.01);
  const yRange = yMax - yMin || 1e-8;

  const xScale = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const yScale = (v: number) => PAD.top + plotH - ((v - yMin) / yRange) * plotH;
  const fmt = formatY ?? ((v: number) => v < 0.01 ? v.toExponential(1) : v.toFixed(3));

  const yTicks = [yMin, yMin + yRange * 0.5, yMax];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" preserveAspectRatio="xMidYMid meet">
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={PAD.left} x2={W - PAD.right} y1={yScale(t)} y2={yScale(t)} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="3,3" />
          <text x={PAD.left - 4} y={yScale(t) + 3} textAnchor="end" fill="var(--text-muted)" fontSize="7.5" fontFamily="monospace">{fmt(t)}</text>
        </g>
      ))}

      {/* X labels */}
      {allPoints.filter((_, i, a) => i % Math.max(1, Math.floor(a.length / 6)) === 0 || i === a.length - 1)
        .map((p, i) => (
          <text key={i} x={xScale(p.x)} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize="7.5" fontFamily="monospace">{p.x}</text>
        ))}

      {/* Lines + dots */}
      {series.map((s, si) => {
        if (s.data.length === 0) return null;
        const path = s.data.map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.x).toFixed(1)} ${yScale(p.y).toFixed(1)}`).join(" ");
        return (
          <g key={si}>
            <path d={path} fill="none" stroke={s.color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            {s.data.map((p, i) => (
              <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r="2.5" fill={s.color} stroke="var(--bg-secondary)" strokeWidth="1" />
            ))}
          </g>
        );
      })}

      {/* Title + legend */}
      <text x={PAD.left} y={13} fill="var(--text-secondary)" fontSize="9" fontWeight="600">{title}</text>
      {series.map((s, i) => (
        <g key={i} transform={`translate(${PAD.left + 60 + i * 55}, 8)`}>
          <line x1="0" y1="5" x2="10" y2="5" stroke={s.color} strokeWidth="2" />
          <text x="13" y="8" fill="var(--text-muted)" fontSize="7.5">{s.label}</text>
        </g>
      ))}
    </svg>
  );
}
