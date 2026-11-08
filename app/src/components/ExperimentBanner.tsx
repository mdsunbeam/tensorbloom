import { useExperimentStore, useExperimentDiff } from "../store/experiment-store";
import { useTrainingStore } from "../store/training-store";
import { startTraining } from "../lib/training";
import { changeCount, describeChanges, nodeMarks } from "../lib/experiment-diff";
import { useEffect } from "react";
import Button from "./ui/Button";

/**
 * Shown on the canvas during a "change one thing" experiment: what has changed since the run it
 * started from, a nudge to keep it to one change, and Train.
 */
export default function ExperimentBanner() {
  const exp = useExperimentDiff();
  const end = useExperimentStore((s) => s.end);
  const busy = useTrainingStore((s) => s.status === "running" || s.status === "starting");
  const setMarks = useExperimentStore((s) => s.setMarks);
  // Mark the changed layers on the canvas
  useEffect(() => { setMarks(exp ? nodeMarks(exp.diff) : new Map()); }, [exp, setMarks]);
  if (!exp) return null;

  const { base, diff } = exp;
  const count = changeCount(diff);
  const changes = describeChanges(diff);

  return (
    <div
      className="absolute top-3 left-1/2 -translate-x-1/2 z-10 w-[420px] max-w-[calc(100%-24px)] rounded-lg border bg-surface p-3 text-[11px] space-y-2"
      style={{ boxShadow: "var(--node-shadow)", borderColor: count > 1 ? "var(--warning)" : "var(--accent)" }}
      role="region"
      aria-label="Change one thing"
    >
      <div className="flex items-baseline gap-2">
        <span className="text-xs font-semibold text-fg">Change one thing</span>
        <span className="text-fg-muted">from Run {base.number}</span>
        <button type="button" className="ml-auto text-fg-muted hover:text-fg" onClick={end} title="Stop the experiment and keep your changes">
          Cancel
        </button>
      </div>

      {count === 0 && (
        <p className="text-fg-secondary">
          Change one layer setting, add or remove a layer, or change one training setting. Then train, and the run is put
          next to Run {base.number} so you can see what that change did.
        </p>
      )}

      {count > 0 && (
        <ul className="space-y-0.5">
          {changes.slice(0, 5).map((c) => (
            <li key={c} className="flex gap-1.5 text-fg">
              <span aria-hidden="true" className={count > 1 ? "text-warning" : "text-accent-text"}>●</span>
              <span className="font-mono truncate" title={c}>{c}</span>
            </li>
          ))}
          {changes.length > 5 && <li className="text-fg-muted">and {changes.length - 5} more</li>}
        </ul>
      )}

      {count > 1 && (
        <p className="text-warning">
          That's {count} changes. With more than one, you can't tell which one made the difference. Ctrl+Z takes one back.
        </p>
      )}

      {count > 0 && (
        <div className="flex justify-end">
          <Button size="sm" variant={count === 1 ? "primary" : "secondary"} disabled={busy} onClick={() => void startTraining()} title="Ctrl+Enter">
            {count === 1 ? "Train and compare" : "Train anyway"}
          </Button>
        </div>
      )}
    </div>
  );
}
