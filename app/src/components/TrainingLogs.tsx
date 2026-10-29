import { useState, useEffect, useRef } from "react";
import { useTrainingStore } from "../store/training-store";

function formatTime(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

export default function TrainingLogs() {
  const logs = useTrainingStore((s) => s.logs);
  const status = useTrainingStore((s) => s.status);
  const currentEpoch = useTrainingStore((s) => s.currentEpoch);
  const config = useTrainingStore((s) => s.config);
  const error = useTrainingStore((s) => s.error);
  const startedAt = useTrainingStore((s) => s.startedAt);
  const metrics = useTrainingStore((s) => s.metrics);
  const clearLogs = useTrainingStore((s) => s.clearLogs);

  const bottomRef = useRef<HTMLDivElement>(null);

  // Elapsed timer
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    // Catch up when the status changes, so a short run doesn't show a time from before it started
    setNow(Date.now());
    if (status !== "running" && status !== "starting") return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [status]);

  const elapsed = startedAt ? now - startedAt : 0;

  // Current loss
  const latestLoss = metrics.length > 0 ? metrics[metrics.length - 1].trainLoss : undefined;

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Metrics strip */}
      <div
        className="flex items-center gap-4 px-3 py-1.5 flex-shrink-0 text-[11px]"
        style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-tertiary)" }}
      >
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full" style={{
            background: status === "running" ? "var(--success)" : status === "starting" ? "var(--warning)" : status === "finished" ? "var(--accent)" : status === "error" ? "var(--danger)" : "var(--text-muted)",
          }} />
          <span className="font-semibold uppercase" style={{ color: "var(--text-secondary)" }}>{status}</span>
        </div>
        <span className="tabular-nums" style={{ color: "var(--text-muted)" }}>Epoch {currentEpoch}/{config.epochs}</span>
        {latestLoss != null && <span style={{ color: "var(--text-secondary)" }}>Loss: {latestLoss.toFixed(4)}</span>}
        {startedAt && <span className="tabular-nums" style={{ color: "var(--text-muted)" }}>{formatTime(elapsed)}</span>}
        {error && <span style={{ color: "var(--danger)" }}>{error}</span>}
        <button
          onClick={clearLogs}
          className="ml-auto text-[11px] px-1.5 py-0.5 rounded"
          style={{ color: "var(--text-muted)", background: "var(--bg-secondary)" }}
        >
          Clear
        </button>
      </div>

      {/* Log content */}
      <div
        className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden font-mono px-3 py-1"
        style={{ background: "var(--bg-primary)", fontSize: "11px", lineHeight: "1.5" }}
      >
        {logs.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2.5" style={{ background: "var(--bg-tertiary)" }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" />
                </svg>
              </div>
              <div className="text-xs font-medium mb-1" style={{ color: "var(--text-muted)" }}>No logs yet</div>
              <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                Training output and debug messages will appear here when a run starts
              </div>
            </div>
          </div>
        ) : (
          logs.map((l, i) => (
            <div key={i} className="flex gap-2" style={{ color: l.stream === "stderr" ? "var(--warning)" : "var(--text-muted)" }}>
              <span className="flex-shrink-0 select-none" style={{ color: "var(--text-muted)" }}>{formatTimestamp(l.timestamp)}</span>
              {/* stderr gets a mark, not only a color */}
              <span className="flex-shrink-0 w-3 select-none" aria-label={l.stream === "stderr" ? "stderr" : undefined}>
                {l.stream === "stderr" && (
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" className="mt-[3px]" aria-hidden="true">
                    <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                )}
              </span>
              <span style={{ whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{l.text}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
