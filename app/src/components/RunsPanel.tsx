import { useMemo, useState } from "react";
import LineChart from "./LineChart";
import Button from "./ui/Button";
import Segmented from "./ui/Segmented";
import { toast } from "./Toast";
import { useRunStore, runHeadline, runWeights, type RunRecord } from "../store/run-store";
import { useTrainingStore } from "../store/training-store";
import { useGraphStore } from "../store/graph-store";
import { replaceGraph } from "../lib/document-actions";
import { isDatasetNode, extractDatasetConfig } from "../lib/dataset-metadata";
import { formatParamCount } from "../lib/model-summary";
import { formatDuration, timeAgo } from "../lib/format-time";
import { formatLoss } from "./RunControls";

/** Each run keeps one color wherever it's drawn, picked by its number */
export function runColor(run: RunRecord): string {
  return `var(--run-${((run.number - 1) % 6) + 1})`;
}

const STATUS_TEXT: Record<RunRecord["status"], string> = { finished: "Done", stopped: "Stopped", failed: "Failed" };

/** The project's past runs: pick some to compare their curves and settings, or bring a run's model back */
export default function RunsPanel({ onTryModel }: { onTryModel: (runId: string) => void }) {
  const runs = useRunStore((s) => s.runs);
  const compareIds = useRunStore((s) => s.compareIds);
  const [view, setView] = useState<"curves" | "settings">("curves");

  // Compared runs in run order, so columns and colors don't shuffle as runs are picked
  const selected = useMemo(() => runs.filter((r) => compareIds.includes(r.id)), [runs, compareIds]);

  if (runs.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="text-center max-w-sm">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2.5 bg-surface-raised text-fg-muted">
            <HistoryIcon />
          </div>
          <div className="text-xs font-medium mb-1 text-fg-secondary">No runs yet</div>
          <p className="text-[11px] leading-relaxed text-fg-muted">
            Each run you train is kept here with its model and settings, so you can compare runs and go back to an earlier one.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex min-h-0">
      <RunList runs={runs} compareIds={compareIds} onTryModel={onTryModel} />
      <section aria-label="Compare runs" className="flex-1 min-w-0 flex flex-col">
        <div className="flex items-center gap-3 px-3 py-1.5 flex-shrink-0 border-b border-line">
          <Segmented
            label="Compare"
            size="sm"
            className="w-48"
            value={view}
            onChange={setView}
            options={[{ value: "curves", label: "Curves" }, { value: "settings", label: "Settings" }]}
          />
          <span className="text-[11px] text-fg-muted">
            {selected.length === 0 ? "Pick runs on the left to compare them"
              : selected.length === 1 ? `Run ${selected[0].number}. Pick another run to compare with it.`
              : `Comparing ${selected.length} runs`}
          </span>
        </div>
        <div className="flex-1 min-h-0">
          {selected.length === 0 ? null : view === "curves" ? <RunCurves runs={selected} /> : <RunSettings runs={selected} />}
        </div>
      </section>
    </div>
  );
}

function RunList({ runs, compareIds, onTryModel }: { runs: RunRecord[]; compareIds: string[]; onTryModel: (runId: string) => void }) {
  const toggleCompare = useRunStore((s) => s.toggleCompare);
  const removeRun = useRunStore((s) => s.removeRun);
  const training = useTrainingStore((s) => s.status === "running" || s.status === "starting");
  const liveRunId = useRunStore((s) => s.liveRunId);

  const restore = (run: RunRecord) => {
    replaceGraph(run.graph.nodes, run.graph.edges, useGraphStore.getState().projectName);
    // The device is this computer's, not the run's
    const { device } = useTrainingStore.getState().config;
    useTrainingStore.getState().updateConfig({ ...run.config, device });
    toast.success(`Restored Run ${run.number}'s model and training settings. Ctrl+Z brings back your previous graph.`);
  };

  return (
    <div className="w-[320px] flex-shrink-0 flex flex-col border-r border-line min-h-0">
      <div className="flex items-baseline justify-between px-3 py-2 flex-shrink-0 border-b border-line">
        <span className="text-[11px] font-semibold text-fg-secondary">{runs.length} run{runs.length === 1 ? "" : "s"}</span>
        <span className="text-[11px] text-fg-muted">Saved with the project</span>
      </div>
      <ul className="flex-1 min-h-0 overflow-y-auto py-1">
        {[...runs].reverse().map((run) => {
          const checked = compareIds.includes(run.id);
          const headline = runHeadline(run);
          return (
            <li key={run.id} className="group relative flex items-start gap-2.5 px-3 py-1.5 hover:bg-surface-raised focus-within:bg-surface-raised">
              <label className="flex items-start gap-2.5 flex-1 min-w-0 cursor-pointer">
                <input type="checkbox" className="peer sr-only" checked={checked} onChange={() => toggleCompare(run.id)} aria-label={`Compare Run ${run.number}`} />
                <span
                  className="mt-0.5 w-3.5 h-3.5 rounded flex items-center justify-center flex-shrink-0 border-2 peer-focus-visible:ring-2 peer-focus-visible:ring-accent peer-focus-visible:ring-offset-1 peer-focus-visible:ring-offset-surface"
                  style={{ borderColor: runColor(run), background: checked ? runColor(run) : "transparent" }}
                  aria-hidden="true"
                >
                  {checked && (
                    <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="var(--bg-primary)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  )}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-baseline gap-2">
                    <span className="text-xs font-medium text-fg truncate">Run {run.number} <span className="font-normal text-fg-muted">· {run.modelName}</span></span>
                    {headline && (
                      <span className="ml-auto flex-shrink-0 text-[11px] tabular-nums text-fg-secondary group-hover:invisible group-focus-within:invisible">
                        <span className="font-semibold text-fg">{headline.value}</span> {headline.label}
                      </span>
                    )}
                  </span>
                  <span className="flex items-center gap-1.5 text-[11px] text-fg-muted tabular-nums">
                    <StatusMark status={run.status} />
                    <span className={run.status === "failed" ? "text-danger" : undefined}>{STATUS_TEXT[run.status]}</span>
                    <span aria-hidden="true">·</span>
                    <span title={new Date(run.startedAt).toLocaleString()}>{timeAgo(Date.parse(run.startedAt))}</span>
                    {run.status !== "failed" && <><span aria-hidden="true">·</span><span>{formatDuration(run.durationMs)}</span></>}
                  </span>
                  {run.error && <span className="block text-[11px] text-fg-muted truncate" title={run.error}>{run.error}</span>}
                </span>
              </label>
              {/* Shown on hover or keyboard focus, in place of the headline */}
              <span className="absolute right-2 top-1 flex gap-1 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100">
                {run.status !== "failed" && (
                  <Button size="sm" disabled={training || !runWeights(run, liveRunId)} onClick={() => onTryModel(run.id)}
                    title={training ? "Wait for training to finish" : runWeights(run, liveRunId) ? "See what this run's model predicts" : "This run's weights weren't kept"}>
                    Try
                  </Button>
                )}
                <Button size="sm" disabled={training} onClick={() => restore(run)} title={training ? "Wait for training to finish" : "Open this run's model and training settings"}>
                  Restore
                </Button>
                <Button size="sm" variant="danger" onClick={() => removeRun(run.id)} aria-label={`Delete Run ${run.number}`}>
                  Delete
                </Button>
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function StatusMark({ status }: { status: RunRecord["status"] }) {
  const common = { width: 11, height: 11, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2.6, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  if (status === "finished") return <svg {...common} className="text-success"><polyline points="20 6 9 17 4 12" /></svg>;
  if (status === "stopped") return <svg {...common} className="text-fg-muted"><rect x="6" y="6" width="12" height="12" rx="1" /></svg>;
  return <svg {...common} className="text-danger"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>;
}

type Point = { x: number; y: number };
type Series = { data: Point[]; color: string; label: string; dashed?: boolean };

/**
 * Loss and accuracy per epoch. One run shows its train and validation lines;
 * several show one line each, validation where the run has it.
 */
function RunCurves({ runs }: { runs: RunRecord[] }) {
  const charts = useMemo(() => {
    const pick = (run: RunRecord, val: "valLoss" | "valAcc", train: "trainLoss" | "trainAcc"): { data: Point[]; isVal: boolean } => {
      const v = run.epochs.filter((e) => e[val] != null).map((e) => ({ x: e.epoch, y: e[val]! }));
      if (v.length) return { data: v, isVal: true };
      return { data: run.epochs.filter((e) => e[train] != null).map((e) => ({ x: e.epoch, y: e[train]! })), isVal: false };
    };

    if (runs.length === 1) {
      const run = runs[0];
      const both = (train: "trainLoss" | "trainAcc", val: "valLoss" | "valAcc"): Series[] => [
        { data: run.epochs.filter((e) => e[train] != null).map((e) => ({ x: e.epoch, y: e[train]! })), color: runColor(run), label: "Train" },
        { data: run.epochs.filter((e) => e[val] != null).map((e) => ({ x: e.epoch, y: e[val]! })), color: runColor(run), label: "Val", dashed: true },
      ].filter((s) => s.data.length > 0);
      return { loss: both("trainLoss", "valLoss"), acc: both("trainAcc", "valAcc"), lossTitle: "Loss", accTitle: "Accuracy" };
    }

    const lossPicks = runs.map((r) => ({ run: r, ...pick(r, "valLoss", "trainLoss") }));
    const accPicks = runs.map((r) => ({ run: r, ...pick(r, "valAcc", "trainAcc") }));
    const toSeries = (picks: typeof lossPicks, allVal: boolean): Series[] => picks
      .filter((p) => p.data.length > 0)
      .map((p) => ({ data: p.data, color: runColor(p.run), label: `Run ${p.run.number}${p.isVal || allVal ? "" : " (train)"}` }));
    const lossAllVal = lossPicks.every((p) => p.isVal);
    const accAllVal = accPicks.every((p) => p.isVal || p.data.length === 0);
    return {
      loss: toSeries(lossPicks, lossAllVal),
      acc: toSeries(accPicks, accAllVal),
      lossTitle: lossAllVal ? "Val loss" : "Loss",
      accTitle: accAllVal ? "Val accuracy" : "Accuracy",
    };
  }, [runs]);

  if (charts.loss.length === 0) {
    return <div className="h-full flex items-center justify-center text-[11px] text-fg-muted">{runs.length === 1 ? "This run" : "These runs"} ended before the first epoch, so there's nothing to draw.</div>;
  }

  const hasAcc = charts.acc.length > 0;
  return (
    <div className="h-full grid gap-3 p-3" style={{ gridTemplateColumns: `repeat(${hasAcc ? 2 : 1}, minmax(0, 1fr))` }}>
      <LineChart title={charts.lossTitle} series={charts.loss} formatY={formatLoss} />
      {hasAcc && <LineChart title={charts.accTitle} series={charts.acc} formatY={(v) => `${(v * 100).toFixed(1)}%`} formatTick={(v) => `${Math.round(v * 100)}%`} zeroBased maxY={1} />}
    </div>
  );
}

interface Row {
  label: string;
  values: string[];
  /** Index of the best value, for results */
  best?: number | null;
}

/** Results, model and training settings side by side, keeping to what differs unless asked for all */
function RunSettings({ runs }: { runs: RunRecord[] }) {
  const [showAll, setShowAll] = useState(false);
  const sections = useMemo(() => settingsRows(runs), [runs]);
  const comparing = runs.length > 1;

  const differs = (r: Row) => new Set(r.values).size > 1;
  const visible = sections.map((s) => ({
    ...s,
    rows: !comparing || showAll || s.always ? s.rows : s.rows.filter(differs),
  }));
  const hidden = sections.reduce((n, s, i) => n + s.rows.length - visible[i].rows.length, 0);

  return (
    <div className="h-full overflow-auto">
      <table className="w-full text-[11px] tabular-nums border-separate border-spacing-0">
        <thead className="sticky top-0 z-10 bg-surface">
          <tr>
            <th scope="col" className="w-40 px-3 py-1.5 text-left font-medium text-fg-muted border-b border-line"><span className="sr-only">Setting</span></th>
            {runs.map((r) => (
              <th key={r.id} scope="col" className="px-3 py-1.5 text-left font-semibold text-fg border-b border-line whitespace-nowrap">
                <span className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle" style={{ background: runColor(r) }} aria-hidden="true" />
                Run {r.number}
              </th>
            ))}
            <th className="border-b border-line" aria-hidden="true" />
          </tr>
        </thead>
        {visible.map((s) => (
          <tbody key={s.title}>
            <tr>
              <th scope="colgroup" colSpan={runs.length + 2} className="px-3 pt-2.5 pb-1 text-left text-[11px] font-semibold uppercase tracking-wide text-fg-muted">
                {s.title}
              </th>
            </tr>
            {s.rows.length === 0 ? (
              <tr><td colSpan={runs.length + 2} className="px-3 py-1 text-fg-muted">Same for all</td></tr>
            ) : s.rows.map((row) => {
              const marked = comparing && !s.always && differs(row);
              return (
                <tr key={row.label} className="hover:bg-surface-raised">
                  <th scope="row" className="px-3 py-1 text-left font-normal text-fg-secondary whitespace-nowrap">
                    {row.label}
                    {marked && showAll && <><span className="ml-1.5 inline-block w-1.5 h-1.5 rounded-full bg-accent align-middle" aria-hidden="true" /><span className="sr-only"> (differs)</span></>}
                  </th>
                  {row.values.map((v, i) => (
                    <td key={i} className={`px-3 py-1 whitespace-nowrap ${row.best === i ? "font-semibold text-fg" : "text-fg-secondary"}`}>
                      {v}
                      {row.best === i && <span className="ml-1.5 text-success font-medium">best</span>}
                    </td>
                  ))}
                  <td />
                </tr>
              );
            })}
          </tbody>
        ))}
      </table>
      {comparing && (hidden > 0 || showAll) && (
        <div className="px-3 py-2">
          <Button size="sm" variant="ghost" onClick={() => setShowAll(!showAll)}>
            {showAll ? "Show only what differs" : `Show all settings (${hidden} the same)`}
          </Button>
        </div>
      )}
    </div>
  );
}

function settingsRows(runs: RunRecord[]): { title: string; rows: Row[]; always?: boolean }[] {
  const pct = (v: number | undefined) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
  const num = (v: number | undefined) => (v == null ? "–" : formatLoss(v));
  const onOff = (v: boolean) => (v ? "On" : "Off");
  // Where higher or lower is better and more than one run has a value, the best one gets marked,
  // unless another run shows the same number once rounded
  const bestOf = (vals: (number | undefined)[], better: "high" | "low", format: (v: number) => string): number | null => {
    const present = vals.map((v, i) => ({ v, i })).filter((x): x is { v: number; i: number } => x.v != null);
    if (runs.length < 2 || present.length < 2) return null;
    const top = present.reduce((a, b) => ((better === "high" ? b.v > a.v : b.v < a.v) ? b : a));
    return present.some((x) => x.i !== top.i && format(x.v) === format(top.v)) ? null : top.i;
  };

  const results: Row[] = [];
  const res = runs.map((r) => r.results);
  results.push({
    label: "Status",
    values: runs.map((r) => (r.status === "stopped" || r.results?.earlyStopped ? `Stopped at epoch ${r.epochs.length}` : STATUS_TEXT[r.status])),
  });
  if (res.some((r) => r?.bestValAcc != null)) {
    results.push({
      label: "Best val accuracy",
      values: res.map((r) => (r?.bestValAcc == null ? "–" : `${pct(r.bestValAcc)}${r.bestValAccEpoch != null ? ` (epoch ${r.bestValAccEpoch})` : ""}`)),
      best: bestOf(res.map((r) => r?.bestValAcc), "high", pct),
    });
  }
  if (res.some((r) => r?.bestValLoss != null)) {
    results.push({
      label: "Best val loss",
      values: res.map((r) => (r?.bestValLoss == null ? "–" : `${num(r.bestValLoss)}${r.bestValEpoch != null ? ` (epoch ${r.bestValEpoch})` : ""}`)),
      best: bestOf(res.map((r) => r?.bestValLoss), "low", num),
    });
  }
  results.push({ label: "Final train loss", values: res.map((r) => num(r?.finalTrainLoss)), best: bestOf(res.map((r) => r?.finalTrainLoss), "low", num) });
  if (res.some((r) => r?.finalTrainAcc != null)) results.push({ label: "Final train accuracy", values: res.map((r) => pct(r?.finalTrainAcc)) });
  results.push({ label: "Epochs run", values: runs.map((r) => String(r.results?.totalEpochs ?? r.epochs.length)) });
  results.push({ label: "Time", values: runs.map((r) => formatDuration(r.durationMs)) });

  // Name each distinct architecture A, B, C... so a changed model stands out
  const keys = [...new Set(runs.map((r) => r.model.key))];
  const model: Row[] = [
    { label: "Architecture", values: runs.map((r) => (keys.length === 1 ? "Same" : `Model ${String.fromCharCode(65 + keys.indexOf(r.model.key))}`)) },
    { label: "Parameters", values: runs.map((r) => formatParamCount(r.model.params)) },
    { label: "Layers", values: runs.map((r) => String(r.model.layers)) },
  ];

  const data = runs.map(runData);
  const c = runs.map((r) => r.config);
  const training: Row[] = [
    { label: "Data", values: data.map((d) => d.source) },
    { label: "Validation split", values: data.map((d) => `${Math.round(d.valSplit * 100)}%`) },
    { label: "Augmentation", values: data.map((d) => d.augmentation) },
    { label: "Optimizer", values: c.map((x) => x.optimizer.type) },
    { label: "Learning rate", values: c.map((x) => String(x.optimizer.lr)) },
    { label: "Weight decay", values: c.map((x) => String(x.optimizer.weightDecay)) },
    { label: "Momentum", values: c.map((x) => (x.optimizer.type === "SGD" || x.optimizer.type === "RMSprop" ? String(x.optimizer.momentum ?? 0) : "–")) },
    { label: "LR scheduler", values: c.map((x) => x.scheduler?.type ?? "None") },
    { label: "Epochs", values: c.map((x) => String(x.epochs)) },
    { label: "Batch size", values: c.map((x) => String(x.batchSize)) },
    { label: "Loss function", values: c.map((x) => x.loss) },
    { label: "Weight initialization", values: c.map((x) => x.weightInit) },
    { label: "Random seed", values: c.map((x) => (x.seed == null ? "Random" : String(x.seed))) },
    { label: "Mixed precision", values: c.map((x) => onOff(x.mixedPrecision)) },
    { label: "Gradient clipping", values: c.map((x) => (x.gradientClipping.enabled ? `Max norm ${x.gradientClipping.maxNorm}` : "Off")) },
    { label: "Early stopping", values: c.map((x) => (x.earlyStopping.enabled ? `Patience ${x.earlyStopping.patience}` : "Off")) },
    { label: "Device", values: c.map((x) => x.device) },
  ];

  return [
    { title: "Results", rows: results, always: true },
    { title: "Model", rows: model },
    { title: "Training", rows: training },
  ];
}

/** What a run trained on, as its Data node set it up (falling back to the Training panel's settings) */
function runData(run: RunRecord): { source: string; valSplit: number; augmentation: string } {
  const node = run.graph.nodes.find((n) => isDatasetNode(n.data.moduleType));
  const fromNode = node ? extractDatasetConfig(node.data.moduleType, node.data.params) : null;
  const params = node?.data.params ?? {};
  const src = String(params.source ?? node?.data.moduleType ?? run.config.dataset.type);
  // Name the files for the user's own data, the dataset for Hugging Face, and the preset otherwise
  const source = src === "HuggingFace" && params.dataset_name ? `Hugging Face: ${params.dataset_name}`
    : ["ImageFolder", "CSV", "TensorFile"].includes(src) && params.path ? `${src}: ${String(params.path).split(/[/\\]/).filter(Boolean).pop()}`
    : src;
  const aug = fromNode?.augmentation ?? run.config.augmentation;
  const parts = [
    aug.horizontalFlip && "flip",
    aug.verticalFlip && "vertical flip",
    aug.randomCrop && "crop",
    aug.colorJitter && "color jitter",
    aug.rotation > 0 && `rotate ${aug.rotation}°`,
  ].filter(Boolean) as string[];
  return {
    source,
    valSplit: fromNode?.valSplit ?? run.config.valSplit,
    augmentation: parts.length ? parts.join(", ") : "None",
  };
}

function HistoryIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><polyline points="3 3 3 8 8 8" /><polyline points="12 7 12 12 15 15" />
    </svg>
  );
}
