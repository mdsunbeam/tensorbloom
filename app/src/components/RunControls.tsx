import { useGraphStore } from "../store/graph-store";
import { useTrainingStore, runFraction } from "../store/training-store";
import { startTraining, stopTraining } from "../lib/training";
import { usePythonStore } from "../store/python-store";
import Button from "./ui/Button";

/** Train/Stop button with the run's live status, for the toolbar */
export default function RunControls({ onShowRun }: { onShowRun: () => void }) {
  const status = useTrainingStore((s) => s.status);
  const currentEpoch = useTrainingStore((s) => s.currentEpoch);
  const epochs = useTrainingStore((s) => s.config.epochs);
  const fraction = useTrainingStore(runFraction);
  const downloading = useTrainingStore((s) => s.downloading);
  const results = useTrainingStore((s) => s.trainingResults);
  const errorKind = useTrainingStore((s) => s.errorKind);
  const hasNodes = useGraphStore((s) => s.nodes.length > 0);
  const stopping = useTrainingStore((s) => s.stopRequested && s.status === "running");
  const pythonStatus = usePythonStore((s) => s.status);
  const pythonStep = usePythonStore((s) => s.step);
  const setupCardOpen = usePythonStore((s) => s.cardOpen);

  let chip: React.ReactNode = null;
  let onChipClick = onShowRun;
  if ((pythonStatus === "installing" || pythonStatus === "failed") && !setupCardOpen) {
    // The setup card is hidden; keep its state visible here
    onChipClick = () => usePythonStore.getState().openCard();
    chip = pythonStatus === "installing" ? (
      <>
        <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" aria-hidden="true" />
        <span className="text-fg-secondary" aria-live="polite">
          Setting up Python{pythonStep ? `, step ${pythonStep.step} of ${pythonStep.total}` : "…"}
        </span>
      </>
    ) : (
      <>
        <AlertIcon className="text-danger" />
        <span className="text-danger">Python setup didn't finish</span>
      </>
    );
  } else if (status === "running" || status === "starting") {
    const label = status === "starting" ? "Checking the model…"
      : stopping ? "Stopping…"
      : downloading ? `Downloading ${downloading}…`
      : currentEpoch === 0 ? "Preparing data…"
      : `Epoch ${currentEpoch} of ${epochs}`;
    chip = (
      <>
        <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" aria-hidden="true" />
        <span className="text-fg-secondary" aria-live="polite">{label}</span>
        {status === "running" && currentEpoch > 0 && !downloading && (
          <>
            <span className="w-16 h-1 rounded-full bg-surface-raised overflow-hidden" role="progressbar" aria-label="Training progress"
              aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)}>
              <span className="block h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${fraction * 100}%` }} />
            </span>
            <span className="text-fg-muted w-7 text-right">{Math.round(fraction * 100)}%</span>
          </>
        )}
      </>
    );
  } else if (status === "finished" && results) {
    const stoppedEarly = results.stoppedByUser || results.earlyStopped;
    const metric = results.bestValAcc != null ? `${(results.bestValAcc * 100).toFixed(1)}% val acc`
      : results.bestValLoss != null ? `val loss ${formatLoss(results.bestValLoss)}`
      : `loss ${formatLoss(results.finalTrainLoss)}`;
    chip = (
      <>
        <CheckIcon className={stoppedEarly ? "text-fg-muted" : "text-success"} />
        <span className="text-fg-secondary">{results.stoppedByUser ? `Stopped at epoch ${results.totalEpochs}` : "Done"}</span>
        <span className="text-fg-muted">{metric}</span>
      </>
    );
  } else if (status === "error") {
    chip = (
      <>
        <AlertIcon className="text-danger" />
        <span className="text-danger">{errorKind === "check" ? "Can't train yet" : "Run failed"}</span>
      </>
    );
  }

  return (
    <div className="flex items-center gap-1.5">
      {chip && (
        <button
          type="button"
          onClick={onChipClick}
          title={onChipClick === onShowRun ? "Show the run" : "Show Python setup"}
          className="flex items-center gap-2 h-7 px-2.5 rounded-lg text-[11px] tabular-nums whitespace-nowrap hover:bg-surface-raised transition-colors"
        >
          {chip}
        </button>
      )}
      {status === "running" ? (
        <Button size="sm" onClick={() => void stopTraining()} disabled={stopping} title="Stop training">
          <span className="w-2 h-2 rounded-[2px] bg-danger" aria-hidden="true" />
          Stop
        </Button>
      ) : (
        <Button
          size="sm"
          variant="primary"
          onClick={() => void startTraining()}
          disabled={status === "starting" || !hasNodes}
          title={hasNodes ? "Start training (Ctrl+Enter)" : "Add layers to the canvas to train"}
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 4l15 8-15 8z" /></svg>
          {status === "finished" || status === "error" ? "Train again" : "Train"}
        </Button>
      )}
    </div>
  );
}

export function formatLoss(v: number): string {
  return v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(3) : v.toPrecision(3);
}

function CheckIcon({ className = "" }: { className?: string }) {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function AlertIcon({ className = "" }: { className?: string }) {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}
