import { useMemo, useState, useEffect } from "react";
import LineChart from "./LineChart";
import { useTrainingStore, runFraction, type TrainingErrorKind, type TrainingResults } from "../store/training-store";
import { startTraining } from "../lib/training";
import { explainTrainingError } from "../lib/training-errors";
import Button from "./ui/Button";
import { formatLoss } from "./RunControls";
import { formatDuration } from "../lib/format-time";
import { useRunStore } from "../store/run-store";

function formatTime(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function TrainingCharts({ onSwitchToLogs, onShowRuns, onExport }: { onSwitchToLogs: () => void; onShowRuns: () => void; onExport: () => void }) {
  const metrics = useTrainingStore((s) => s.metrics);
  const status = useTrainingStore((s) => s.status);
  const currentEpoch = useTrainingStore((s) => s.currentEpoch);
  const config = useTrainingStore((s) => s.config);
  const error = useTrainingStore((s) => s.error);
  const errorKind = useTrainingStore((s) => s.errorKind);
  const startedAt = useTrainingStore((s) => s.startedAt);
  const fraction = useTrainingStore(runFraction);
  const downloading = useTrainingStore((s) => s.downloading);
  const results = useTrainingStore((s) => s.trainingResults);

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
  const eta = fraction > 0.02 && startedAt ? (elapsed / fraction) * (1 - fraction) : 0;

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

  if (metrics.length === 0) {
    if (status === "error" && error) {
      return (
        <div className="h-full overflow-y-auto p-3">
          <ErrorPanel message={error} kind={errorKind} onShowLogs={onSwitchToLogs} />
        </div>
      );
    }
    const busy = status === "starting" || status === "running";
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center max-w-xs">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2.5 bg-surface-raised text-fg-muted">
            {busy ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="animate-spin" aria-hidden="true">
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            )}
          </div>
          <div className="text-xs font-medium mb-1 text-fg-secondary" aria-live="polite">
            {status === "starting" ? "Checking the model…"
              : status === "running" ? (downloading ? `Downloading ${downloading}…` : "Preparing data…")
              : "Ready to train"}
          </div>
          <div className="text-[11px] text-fg-muted">
            {status === "starting" ? "Making sure the layers fit the data"
              : status === "running" ? (downloading ? "This happens once. Later runs use the saved copy." : "Loading the dataset and building the model")
              : "Settings are in the Training tab. Loss and accuracy appear here as the model trains."}
          </div>
          {!busy && (
            <Button variant="primary" className="mt-3" onClick={() => void startTraining()} title="Ctrl+Enter">
              Start training
            </Button>
          )}
        </div>
      </div>
    );
  }

  const hasAcc = epochData.some((d) => d.trainAcc != null);
  const hasValLoss = epochData.some((d) => d.valLoss != null);
  const hasValAcc = epochData.some((d) => d.valAcc != null);
  const hasLrVariation = epochData.length > 1 && new Set(epochData.map((d) => d.lr)).size > 1;
  const latestGpu = metrics.length > 0 ? metrics[metrics.length - 1].gpu : undefined;
  const last = epochData[epochData.length - 1];
  // Validation accuracy is the number that says how well the model generalizes, so it leads when there is one
  const headlineAcc = last?.valAcc ?? last?.trainAcc;
  const chartCount = 1 + (hasAcc ? 1 : 0) + (hasLrVariation ? 1 : 0);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Metrics strip */}
      <div
        className="flex items-center gap-4 px-3 py-1.5 flex-shrink-0 text-[11px] tabular-nums"
        style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-tertiary)" }}
      >
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full" style={{
            background: status === "running" ? "var(--success)" : status === "starting" ? "var(--warning)" : status === "finished" ? "var(--accent)" : status === "error" ? "var(--danger)" : "var(--text-muted)",
          }} />
          <span className="font-semibold" style={{ color: "var(--text-primary)" }}>{STATUS_LABELS[status] ?? status}</span>
        </div>
        <span style={{ color: "var(--text-secondary)" }}>Epoch {currentEpoch}/{config.epochs}</span>
        {last && (
          <>
            <Stat label={last.valLoss != null ? "Val loss" : "Loss"} value={(last.valLoss ?? last.trainLoss).toFixed(4)} />
            {headlineAcc != null && <Stat label={last.valAcc != null ? "Val acc" : "Acc"} value={`${(headlineAcc * 100).toFixed(1)}%`} />}
          </>
        )}
        {startedAt && (
          <>
            <span style={{ color: "var(--text-secondary)" }}>{formatTime(elapsed)}</span>
            {eta > 0 && status === "running" && <span style={{ color: "var(--text-muted)" }}>about {formatTime(eta)} left</span>}
          </>
        )}
        {latestGpu && (
          <span style={{ color: "var(--text-muted)" }} title="GPU memory allocated now, and the peak during this run">
            GPU {latestGpu.allocated_mb.toFixed(0)} MB <span style={{ opacity: 0.8 }}>(peak {latestGpu.max_allocated_mb.toFixed(0)})</span>
          </span>
        )}
      </div>

      {status === "error" && error && (
        <div className="px-3 pt-3 flex-shrink-0">
          <ErrorPanel message={error} kind={errorKind} onShowLogs={onSwitchToLogs} />
        </div>
      )}
      {status === "finished" && results && (
        <div className="px-3 pt-3 flex-shrink-0">
          <RunSummary results={results} onExport={onExport} onShowRuns={onShowRuns} />
        </div>
      )}

      {/* Charts, one column each so they use the panel's full height */}
      <div className="flex-1 min-h-0 grid gap-3 p-3" style={{ gridTemplateColumns: `repeat(${chartCount}, minmax(0, 1fr))` }}>
        <LineChart
          title="Loss"
          zeroBased
          series={[
            { data: epochData.map((d) => ({ x: d.epoch, y: d.trainLoss })), color: "var(--chart-train)", label: "Train" },
            ...(hasValLoss
              ? [{ data: epochData.filter((d) => d.valLoss != null).map((d) => ({ x: d.epoch, y: d.valLoss! })), color: "var(--chart-val)", label: "Val", dashed: true }]
              : []),
          ]}
          formatY={(v) => (v !== 0 && Math.abs(v) < 0.01 ? v.toExponential(1) : v.toFixed(3))}
          formatTick={(v) => String(Number(v.toPrecision(3)))}
        />
        {hasAcc && (
          <LineChart
            title="Accuracy"
            maxY={1}
            series={[
              { data: epochData.filter((d) => d.trainAcc != null).map((d) => ({ x: d.epoch, y: d.trainAcc! })), color: "var(--chart-train)", label: "Train" },
              ...(hasValAcc
                ? [{ data: epochData.filter((d) => d.valAcc != null).map((d) => ({ x: d.epoch, y: d.valAcc! })), color: "var(--chart-val)", label: "Val", dashed: true }]
                : []),
            ]}
            formatY={(v) => `${(v * 100).toFixed(1)}%`}
            formatTick={(v) => `${Math.round(v * 100)}%`}
          />
        )}
        {hasLrVariation && (
          <LineChart
            title="Learning rate"
            zeroBased
            series={[{ data: epochData.map((d) => ({ x: d.epoch, y: d.lr })), color: "var(--chart-lr)", label: "LR" }]}
            formatY={(v) => (v === 0 ? "0" : v.toExponential(2))}
            formatTick={(v) => (v === 0 ? "0" : v.toExponential(0))}
          />
        )}
      </div>
    </div>
  );
}

function RunSummary({ results, onExport, onShowRuns }: { results: TrainingResults; onExport: () => void; onShowRuns: () => void }) {
  const planned = results.plannedEpochs ?? results.totalEpochs;
  const title = results.stoppedByUser ? `Stopped at epoch ${results.totalEpochs} of ${planned}`
    : results.earlyStopped ? `Stopped early at epoch ${results.totalEpochs} of ${planned}, when validation stopped improving`
    : `Done in ${formatDuration(results.durationMs)}`;
  const best = results.bestValAcc != null
    ? <>Best val accuracy <Strong>{(results.bestValAcc * 100).toFixed(1)}%</Strong>{results.bestValAccEpoch != null && <> at epoch {results.bestValAccEpoch}</>}</>
    : results.bestValLoss != null
      ? <>Best val loss <Strong>{formatLoss(results.bestValLoss)}</Strong>{results.bestValEpoch != null && <> at epoch {results.bestValEpoch}</>}</>
      : <>Final train loss <Strong>{formatLoss(results.finalTrainLoss)}</Strong></>;
  const stoppedEarly = results.stoppedByUser || results.earlyStopped;
  // The run that just ended is the last one kept; offer to put it next to the one before
  const runs = useRunStore((s) => s.runs);
  const previous = runs.length >= 2 ? runs[runs.length - 2] : null;

  return (
    <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg border border-line bg-surface-raised" role="status">
      <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${stoppedEarly ? "bg-surface text-fg-secondary" : "bg-success-subtle text-success"}`}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-xs font-semibold text-fg">{title}</div>
        <div className="text-[11px] text-fg-secondary">
          {best}
          {stoppedEarly && <> · ran for {formatDuration(results.durationMs)}</>}
          {results.finalTrainAcc != null && <> · final train accuracy {(results.finalTrainAcc * 100).toFixed(1)}%</>}
        </div>
      </div>
      {previous && (
        <Button size="sm" variant="ghost" onClick={() => { useRunStore.getState().setCompare([previous.id, runs[runs.length - 1].id]); onShowRuns(); }}>
          Compare with Run {previous.number}
        </Button>
      )}
      <Button size="sm" onClick={onExport}>Export model</Button>
      <Button size="sm" variant="primary" onClick={() => void startTraining()} title="Ctrl+Enter">Train again</Button>
    </div>
  );
}

function Strong({ children }: { children: React.ReactNode }) {
  return <span className="font-semibold text-fg tabular-nums">{children}</span>;
}

/** Why the run didn't start or didn't finish, what to try, and the raw details */
function ErrorPanel({ message, kind, onShowLogs }: { message: string; kind: TrainingErrorKind; onShowLogs: () => void }) {
  const logs = useTrainingStore((s) => s.logs);
  const [copied, setCopied] = useState(false);
  const hint = kind === "run" ? explainTrainingError(message) : null;
  const problems = kind === "check" ? message.split("\n").filter(Boolean) : [];
  // The traceback the sidecar printed, for the copy and for people who want the details
  const traceback = kind === "run" ? logs.filter((l) => l.stream === "stderr").slice(-30).map((l) => l.text).join("\n") : "";

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(traceback ? `${message}\n\n${traceback}` : message);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard unavailable */ }
  };

  return (
    <div role="alert" className="rounded-lg border border-danger bg-danger-subtle px-3 py-2.5">
      <div className="flex items-start gap-2.5">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 mt-px text-danger" aria-hidden="true">
          <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold text-fg">
            {kind === "check" ? (problems.length === 1 ? "Fix this before training" : "Fix these before training") : "Training stopped with an error"}
          </div>
          {hint && <p className="mt-1 text-xs leading-relaxed text-fg">{hint}</p>}
          {kind === "check" ? (
            <ul className="mt-1 list-disc pl-4 space-y-0.5 text-xs leading-relaxed text-fg-secondary">
              {problems.map((p, i) => <li key={i}>{p}</li>)}
            </ul>
          ) : (
            <pre className="mt-1.5 max-h-24 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-fg-secondary select-text">{message}</pre>
          )}
          <div className="mt-2 flex items-center gap-2">
            <Button size="sm" variant="primary" onClick={() => void startTraining()}>Try again</Button>
            {kind === "run" && <Button size="sm" onClick={onShowLogs}>Show logs</Button>}
            <Button size="sm" variant="ghost" onClick={copy}>{copied ? "Copied" : "Copy error"}</Button>
          </div>
        </div>
      </div>
    </div>
  );
}

const STATUS_LABELS: Record<string, string> = {
  idle: "Idle", starting: "Starting", running: "Training", finished: "Finished", error: "Failed", paused: "Paused",
};

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span style={{ color: "var(--text-muted)" }}>
      {label} <span style={{ color: "var(--text-primary)" }}>{value}</span>
    </span>
  );
}
