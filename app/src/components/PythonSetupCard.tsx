import { useEffect, useRef, useState } from "react";
import { usePythonStore, gpuUpgradeAvailable } from "../store/python-store";
import { useGraphStore } from "../store/graph-store";
import { startTraining } from "../lib/training";
import Button, { IconButton } from "./ui/Button";

function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} kB`;
}

/**
 * Sets up Python and PyTorch from a card on the canvas, so people can build while it installs.
 * Opens on its own at startup when Python is missing, and again when something needs Python.
 * "inline" sits in the empty canvas's column above the welcome card; "floating" keeps to the
 * top right corner, clear of the graph.
 */
export default function PythonSetupCard({ placement }: { placement: "inline" | "floating" }) {
  const status = usePythonStore((s) => s.status);
  const cardOpen = usePythonStore((s) => s.cardOpen);
  const reason = usePythonStore((s) => s.cardReason);
  const justInstalled = usePythonStore((s) => s.justInstalled);
  const step = usePythonStore((s) => s.step);
  const download = usePythonStore((s) => s.download);
  const log = usePythonStore((s) => s.log);
  const error = usePythonStore((s) => s.error);
  const venvPath = usePythonStore((s) => s.venvPath);
  const task = usePythonStore((s) => s.task);
  const nvidiaGpu = usePythonStore((s) => s.nvidiaGpu);
  const offerGpu = usePythonStore(gpuUpgradeAvailable);
  const { install, installGpu, closeCard } = usePythonStore.getState();
  const hasNodes = useGraphStore((s) => s.nodes.length > 0);
  const [showLog, setShowLog] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (showLog && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log, showLog]);

  if (!cardOpen) return null;
  // A failed GPU install leaves the CPU build working, so Python is still "ready"
  const view = status === "ready" && task === "gpu" && error ? "failed" : status;
  if (view === "ready" && !justInstalled) return null;

  const lastLine = log.length > 0 ? log[log.length - 1].trim() : "";

  let title: string;
  let body: React.ReactNode;
  let actions: React.ReactNode;

  if (view === "checking") {
    title = "Checking for Python";
    body = <p className="text-fg-secondary">One moment…</p>;
    actions = null;
  } else if (view === "missing") {
    title = "Set up Python to train";
    body = (
      <>
        {reason && <p className="font-medium text-fg mb-1.5">{reason}</p>}
        <p className="text-fg-secondary leading-relaxed">
          TensorBloom trains models with PyTorch, in a Python environment of its own. Setup downloads Python and PyTorch,
          about 500 MB, and takes a few minutes. You can keep building while it runs.
        </p>
        <p className="text-fg-muted mt-1.5">Nothing else needs to be installed first.</p>
      </>
    );
    actions = (
      <>
        <Button size="sm" onClick={closeCard}>Later</Button>
        <Button size="sm" variant="primary" onClick={() => void install()}>Set up now</Button>
      </>
    );
  } else if (view === "installing") {
    const fraction = download && download.total > 0 ? Math.min(1, download.done / download.total) : null;
    title = task === "gpu" ? "Adding GPU support" : "Setting up Python";
    body = (
      <>
        <div className="flex items-baseline justify-between gap-2 mb-1.5">
          <span className="font-medium text-fg" aria-live="polite">{step?.label ?? "Starting"}…</span>
          {step && <span className="text-fg-muted tabular-nums whitespace-nowrap">Step {step.step} of {step.total}</span>}
        </div>
        <div className="flex gap-1 mb-2.5" role="progressbar" aria-label="Setup progress"
          aria-valuemin={0} aria-valuemax={step?.total ?? 6} aria-valuenow={step ? step.step - 1 : 0}>
          {Array.from({ length: step?.total ?? 6 }, (_, i) => {
            const n = i + 1;
            const state = !step || n > step.step ? "todo" : n < step.step ? "done" : "now";
            return (
              <span key={n} className={`flex-1 h-1 rounded-full ${
                state === "done" ? "bg-accent" : state === "now" ? "bg-accent animate-pulse" : "bg-surface-raised"
              }`} />
            );
          })}
        </div>
        {download && fraction != null && (
          <div className="mb-2">
            <div className="flex justify-between gap-2 text-fg-secondary tabular-nums mb-1">
              <span>Downloaded</span>
              <span className="whitespace-nowrap">{formatBytes(download.done)} of {formatBytes(download.total)}</span>
            </div>
            <div className="h-1 rounded-full bg-surface-raised overflow-hidden" role="progressbar" aria-label="Download progress"
              aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)}>
              <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${fraction * 100}%` }} />
            </div>
          </div>
        )}
        {!showLog && !(download && fraction != null) && lastLine && <p className="font-mono text-fg-muted truncate" title={lastLine}>{lastLine}</p>}
        <p className="text-fg-muted mt-1.5">You can keep building while this runs.</p>
      </>
    );
    actions = <Button size="sm" onClick={closeCard}>Hide</Button>;
  } else if (view === "failed") {
    title = task === "gpu" ? "GPU support didn't install" : "Python setup didn't finish";
    body = (
      <>
        <p className="text-danger leading-relaxed line-clamp-4 break-words" title={error ?? undefined}>{error}</p>
        <p className="text-fg-muted mt-1.5">
          Check your internet connection and free disk space, then try again.
          {task === "gpu" && " Training still works on the CPU in the meantime."}
        </p>
      </>
    );
    actions = (
      <>
        <Button size="sm" variant="primary" onClick={() => void (task === "gpu" ? installGpu() : install())}>Try again</Button>
      </>
    );
  } else if (offerGpu) {
    title = "Python is ready";
    body = (
      <>
        <p className="text-fg-secondary leading-relaxed">
          You can train now, on the CPU. This computer also has {nvidiaGpu ? <span className="font-medium text-fg">{nvidiaGpu}</span> : "an NVIDIA GPU"},
          which trains much faster with GPU support (CUDA). Adding it downloads about 3 GB.
        </p>
        <p className="text-fg-muted mt-1.5">You can add it later from Help, About TensorBloom.</p>
      </>
    );
    actions = (
      <>
        <Button size="sm" onClick={closeCard}>Not now</Button>
        <Button size="sm" variant="primary" onClick={() => void installGpu()}>Add GPU support</Button>
      </>
    );
  } else {
    title = task === "gpu" ? "GPU support is ready" : "Python is ready";
    body = <p className="text-fg-secondary">{task === "gpu" ? "Training can use your NVIDIA GPU now." : "PyTorch is installed. You can train now."}</p>;
    actions = hasNodes ? <Button size="sm" variant="primary" onClick={() => { closeCard(); void startTraining(); }}>Train now</Button> : null;
  }

  const showLogToggle = log.length > 0 && (view === "installing" || view === "failed");

  return (
    <section
      aria-label="Python setup"
      className={`rounded-xl bg-surface border border-line text-[11px] overflow-hidden pointer-events-auto ${
        placement === "floating" ? "absolute top-3 right-3 z-20 w-[340px]" : "w-full max-w-[520px] flex-shrink-0"
      }`}
      style={{ boxShadow: "var(--node-shadow-hover)" }}
    >
      <div className="flex items-start gap-2.5 pl-3.5 pr-2 pt-3">
        <StatusIcon status={view} />
        <div className="flex-1 min-w-0 pt-px">
          <h2 className="text-xs font-semibold text-fg mb-1.5">{title}</h2>
          {body}
        </div>
        {view !== "installing" && view !== "checking" && (
          <IconButton size="sm" label="Close" onClick={closeCard} className="-mt-0.5">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <line x1="6" y1="6" x2="18" y2="18" /><line x1="18" y1="6" x2="6" y2="18" />
            </svg>
          </IconButton>
        )}
      </div>

      {showLog && (
        <pre ref={logRef} className="mx-3.5 mt-2.5 p-2 max-h-40 overflow-auto rounded-md bg-surface-sunken text-[11px] leading-snug font-mono text-fg-secondary whitespace-pre-wrap break-all">
          {log.join("\n")}
        </pre>
      )}

      {(actions || showLogToggle) && (
        <div className="flex items-center gap-1.5 px-3.5 py-3">
          {showLogToggle && (
            <button type="button" onClick={() => setShowLog(!showLog)} aria-expanded={showLog}
              className="text-fg-muted hover:text-fg underline-offset-2 hover:underline mr-auto">
              {showLog ? "Hide details" : "Show details"}
            </button>
          )}
          <div className="flex items-center gap-1.5 ml-auto">{actions}</div>
        </div>
      )}

      {status === "missing" && venvPath && (
        <div className="px-3.5 py-2 border-t border-line-subtle font-mono text-fg-muted truncate" title={venvPath}>
          {venvPath}
        </div>
      )}
    </section>
  );
}

function StatusIcon({ status }: { status: string }) {
  const common = { width: 16, height: 16, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  if (status === "failed") {
    return (
      <span className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 bg-danger-subtle text-danger">
        <svg {...common}><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
      </span>
    );
  }
  if (status === "ready") {
    return (
      <span className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 bg-success-subtle text-success">
        <svg {...common} strokeWidth={2.5}><polyline points="20 6 9 17 4 12" /></svg>
      </span>
    );
  }
  return (
    <span className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 bg-accent-subtle text-accent-text">
      {status === "installing" || status === "checking" ? (
        <svg {...common} className="animate-spin" style={{ animationDuration: "1.4s" }}><path d="M21 12a9 9 0 1 1-6.2-8.56" /></svg>
      ) : (
        // A package, for "needs installing"
        <svg {...common}><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" /><polyline points="3.27 6.96 12 12.01 20.73 6.96" /><line x1="12" y1="22.08" x2="12" y2="12" /></svg>
      )}
    </span>
  );
}
