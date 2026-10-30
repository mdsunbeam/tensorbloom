import { useCallback, useEffect, useRef, useState } from "react";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";
import Segmented from "./ui/Segmented";
import { sidecarCall } from "../lib/sidecar";
import { buildSidecarGraph } from "../lib/sidecar-graph";
import { sidecarTrainConfig, startTraining } from "../lib/training";
import { useRunStore, runWeights } from "../store/run-store";
import { useTrainingStore } from "../store/training-store";
import { formatLoss } from "./RunControls";

/** What the sidecar's model.predict sends back for one input */
type Payload =
  | { kind: "image"; png: string; width: number; height: number }
  | { kind: "values"; values: number[]; shape: number[]; total: number };

interface Sample {
  index: number | null;
  input: Payload;
  target: { kind: "class"; index: number } | Payload | null;
  output: { kind: "classes"; top: { index: number; prob: number }[] } | Payload;
  correct: boolean | null;
}

interface PredictResult {
  split: "validation" | "training";
  dataset_size: number;
  class_names: string[] | null;
  weights: "trained" | "checkpoint";
  accepts_images: boolean;
  task: "prediction" | "reconstruction";
  samples: Sample[];
  scanned?: number;
  scanned_correct?: number;
}

const COUNT = 12;

/** Runs a run's trained model on samples it didn't train on, or on the user's own image, and shows what it predicts */
export default function TryModelDialog({ runId, onClose }: { runId: string | null; onClose: () => void }) {
  const run = useRunStore((s) => s.runs.find((r) => r.id === runId));
  const liveRunId = useRunStore((s) => s.liveRunId);
  const training = useTrainingStore((s) => s.status === "running" || s.status === "starting");
  const [mode, setMode] = useState<"all" | "mistakes">("all");
  const [sampleSeed, setSampleSeed] = useState(() => Math.floor(Math.random() * 1e9));
  const [result, setResult] = useState<PredictResult | null>(null);
  const [own, setOwn] = useState<{ name: string; sample: Sample } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Answers to earlier requests are dropped when a newer one was made
  const request = useRef(0);

  const weights = run ? runWeights(run, liveRunId) : null;

  const predict = useCallback(async (extra: Record<string, unknown>) => {
    if (!run || !weights) throw new Error("This run's weights weren't kept.");
    return sidecarCall<PredictResult>("model.predict", {
      graph: buildSidecarGraph(run.graph.nodes, run.graph.edges),
      config: sidecarTrainConfig(run.graph.nodes, run.config),
      weights: weights.weights,
      checkpoint_path: weights.weights === "checkpoint" ? weights.path : null,
      count: COUNT,
      ...extra,
    });
  }, [run, weights?.weights]); // eslint-disable-line react-hooks/exhaustive-deps -- weights is derived from run and liveRunId

  useEffect(() => {
    if (!runId || !run || training) return;
    if (!weights) { setError(null); setResult(null); return; }
    const id = ++request.current;
    setLoading(true);
    setError(null);
    predict({ mistakes_only: mode === "mistakes", sample_seed: sampleSeed })
      .then((res) => { if (id === request.current) setResult(res); })
      .catch((e) => {
        if (id !== request.current) return;
        const message = e instanceof Error ? e.message : String(e);
        // Python was restarted and lost the run's weights: fall back to its checkpoint, if it has one
        if (weights?.weights === "trained" && message.includes("No trained weights")) {
          useRunStore.setState({ liveRunId: null });
          return;
        }
        setError(message);
      })
      .finally(() => { if (id === request.current) setLoading(false); });
  }, [runId, mode, sampleSeed, predict, training]); // eslint-disable-line react-hooks/exhaustive-deps

  // A different run starts fresh
  useEffect(() => { setResult(null); setOwn(null); setMode("all"); setError(null); }, [runId]);

  const tryOwnImage = async () => {
    let path: string | null = null;
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const picked = await open({ multiple: false, directory: false, title: "Choose an image", filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "bmp", "webp", "gif"] }] });
      path = typeof picked === "string" ? picked : null;
    } catch { /* dialog unavailable outside Tauri */ }
    if (!path) return;
    const id = ++request.current;
    setLoading(true);
    setError(null);
    try {
      const res = await predict({ image_path: path });
      if (id === request.current && res.samples[0]) setOwn({ name: path.split(/[/\\]/).pop() ?? path, sample: res.samples[0] });
    } catch (e) {
      if (id === request.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (id === request.current) setLoading(false);
    }
  };

  if (!runId || !run) return null;

  const isClassifier = result?.samples.some((s) => s.output.kind === "classes") ?? false;
  const shown = result?.samples ?? [];
  const right = shown.filter((s) => s.correct).length;
  const weightsNote = !weights ? null
    : weights.weights === "trained" ? "Uses the weights from the end of this run"
    : weights.epoch < (run.results?.totalEpochs ?? run.epochs.length) ? `Uses the checkpoint from epoch ${weights.epoch}, the last one the run saved`
    : "Uses the run's final checkpoint";

  return (
    <Dialog
      open
      onClose={onClose}
      width={760}
      title={`Try Run ${run.number}'s model`}
      description={result
        ? `Predictions on ${result.split === "validation" ? "validation samples it didn't train on" : "samples from its training data"}`
        : "See what the trained model predicts"}
      footer={
        <>
          {weightsNote && <span className="mr-auto text-[11px] text-fg-muted">{weightsNote}</span>}
          <Button onClick={onClose}>Close</Button>
        </>
      }
    >
      <div className="p-5 space-y-4">
        {!weights ? (
          <Notice
            title="This run's weights weren't kept"
            body="The app keeps the latest run's weights until it closes, and other runs' last checkpoint. This run has neither, so train it again to try it."
            action={<Button size="sm" variant="primary" disabled={training} onClick={() => { onClose(); void startTraining(); }}>Train again</Button>}
          />
        ) : training ? (
          <Notice title="Training is running" body="The model can be tried once training finishes." />
        ) : (
          <>
            <div className="flex items-center gap-2 flex-wrap">
              {(isClassifier || mode === "mistakes") && (
                <Segmented
                  label="Show"
                  size="sm"
                  className="w-48"
                  value={mode}
                  onChange={setMode}
                  options={[{ value: "all", label: "Any samples" }, { value: "mistakes", label: "Mistakes" }]}
                />
              )}
              <Button size="sm" variant="ghost" disabled={loading} onClick={() => setSampleSeed(Math.floor(Math.random() * 1e9))}>
                <ShuffleIcon /> Other samples
              </Button>
              {result?.accepts_images && (
                <Button size="sm" disabled={loading} onClick={tryOwnImage}>
                  <ImageIcon /> Try your own image…
                </Button>
              )}
              <span className="ml-auto text-[11px] text-fg-secondary tabular-nums" aria-live="polite">
                {loading ? "Running the model…"
                  : result && mode === "all" && isClassifier ? <><b className="text-fg">{right} of {shown.length}</b> right</>
                  : result && mode === "mistakes" ? (shown.length
                    ? <>{shown.length} mistake{shown.length === 1 ? "" : "s"} in {result.scanned} samples ({pct((result.scanned_correct ?? 0) / Math.max(1, result.scanned ?? 1))} right)</>
                    : <>No mistakes in {result.scanned} samples</>)
                  : null}
              </span>
            </div>

            {error && <Notice tone="danger" title="The model couldn't be run" body={error} />}

            {own && (
              <div>
                <div className="text-[11px] font-semibold text-fg-secondary mb-1.5">Your image <span className="font-normal text-fg-muted">· {own.name}</span></div>
                <div className="w-[200px]"><SampleCard sample={own.sample} classNames={result?.class_names ?? null} task={result?.task ?? "prediction"} /></div>
              </div>
            )}

            {!result && loading ? (
              <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))" }} aria-hidden="true">
                {Array.from({ length: 8 }, (_, i) => <div key={i} className="h-[196px] rounded-lg border border-line bg-surface-raised animate-pulse" />)}
              </div>
            ) : result && shown.length > 0 ? (
              <ul className={`grid gap-3 transition-opacity ${loading ? "opacity-60" : ""}`} style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))" }}>
                {shown.map((s, i) => (
                  <li key={`${s.index}-${i}`}><SampleCard sample={s} classNames={result.class_names} task={result.task} /></li>
                ))}
              </ul>
            ) : null}
          </>
        )}
      </div>
    </Dialog>
  );
}

function pct(v: number) {
  return `${(v * 100).toFixed(v >= 0.995 && v < 1 ? 1 : 0)}%`;
}

function className(names: string[] | null, i: number) {
  return names?.[i] ?? `Class ${i}`;
}

function SampleCard({ sample, classNames, task }: { sample: Sample; classNames: string[] | null; task: PredictResult["task"] }) {
  const out = sample.output;
  const tone = sample.correct === false ? "border-danger" : "border-line";

  return (
    <div className={`h-full rounded-lg border ${tone} bg-surface-raised overflow-hidden flex flex-col`}>
      {task === "reconstruction" && out.kind !== "classes" ? (
        <div className="grid grid-cols-2 gap-px bg-line">
          <Figure payload={sample.input} caption="Input" />
          <Figure payload={out} caption="Output" />
        </div>
      ) : (
        <Figure payload={sample.input} />
      )}
      <div className="p-2.5 space-y-1.5 text-[11px] flex-1">
        {out.kind === "classes" ? (
          <>
            <Verdict sample={sample} classNames={classNames} />
            <ol className="space-y-1" aria-label="Top predictions">
              {out.top.map((t, i) => (
                <li key={i}>
                  <div className="flex items-baseline justify-between gap-2">
                    <span className={`truncate ${i === 0 ? "text-fg font-medium" : "text-fg-secondary"}`} title={className(classNames, t.index)}>{className(classNames, t.index)}</span>
                    <span className="tabular-nums text-fg-muted flex-shrink-0">{pct(t.prob)}</span>
                  </div>
                  <div className="h-1 mt-0.5 rounded-full bg-surface-sunken overflow-hidden" aria-hidden="true">
                    <div className={`h-full rounded-full ${i === 0 ? (sample.correct === false ? "bg-danger" : "bg-accent") : "bg-fg-muted opacity-50"}`} style={{ width: `${Math.max(2, t.prob * 100)}%` }} />
                  </div>
                </li>
              ))}
            </ol>
          </>
        ) : task === "reconstruction" ? null : (
          <>
            <div className="text-fg-secondary">Predicted <Values payload={out} /></div>
            {sample.target && <div className="text-fg-muted">Actual {sample.target.kind === "class" ? <span className="text-fg">{className(classNames, sample.target.index)}</span> : <Values payload={sample.target} />}</div>}
          </>
        )}
      </div>
    </div>
  );
}

/** Whether the top guess was right, said in words and an icon, not only by color */
function Verdict({ sample, classNames }: { sample: Sample; classNames: string[] | null }) {
  if (sample.correct == null || !sample.target || sample.target.kind !== "class") return null;
  return sample.correct ? (
    <div className="flex items-center gap-1 font-medium text-success">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12" /></svg>
      Right
    </div>
  ) : (
    <div className="flex items-center gap-1 font-medium text-danger min-w-0">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
      <span className="truncate">Wrong, it's {className(classNames, sample.target.index)}</span>
    </div>
  );
}

function Figure({ payload, caption }: { payload: Payload; caption?: string }) {
  return (
    <figure className="bg-surface-sunken flex flex-col items-center justify-center h-[104px] p-2">
      {payload.kind === "image" ? (
        <img
          src={`data:image/png;base64,${payload.png}`}
          alt={caption ?? "Sample"}
          // Small images are scaled up with hard pixel edges so they stay legible
          style={{ imageRendering: payload.width < 64 ? "pixelated" : "auto", width: Math.min(88, payload.width * 3), height: "auto", maxHeight: 88, objectFit: "contain" }}
        />
      ) : (
        <div className="w-full text-[11px] font-mono text-fg-secondary leading-snug break-all line-clamp-5">
          <Values payload={payload} />
        </div>
      )}
      {caption && <figcaption className="mt-1 text-[11px] text-fg-muted">{caption}</figcaption>}
    </figure>
  );
}

function Values({ payload }: { payload: Payload }) {
  if (payload.kind !== "values") return null;
  if (payload.total === 1) return <span className="font-mono text-fg">{formatLoss(payload.values[0])}</span>;
  const more = payload.total - payload.values.length;
  return (
    <span className="font-mono text-fg" title={`Shape [${payload.shape.join(", ")}]`}>
      [{payload.values.map((v) => formatLoss(v)).join(", ")}{more > 0 ? `, … ${more} more` : ""}]
    </span>
  );
}

function Notice({ title, body, action, tone = "neutral" }: { title: string; body: string; action?: React.ReactNode; tone?: "neutral" | "danger" }) {
  return (
    <div role={tone === "danger" ? "alert" : "status"} className={`rounded-lg border px-3 py-2.5 ${tone === "danger" ? "border-danger bg-danger-subtle" : "border-line bg-surface-raised"}`}>
      <div className="text-xs font-semibold text-fg">{title}</div>
      <p className="mt-0.5 text-xs leading-relaxed text-fg-secondary whitespace-pre-wrap">{body}</p>
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

function ShuffleIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="16 3 21 3 21 8" /><line x1="4" y1="20" x2="21" y2="3" /><polyline points="21 16 21 21 16 21" /><line x1="15" y1="15" x2="21" y2="21" /><line x1="4" y1="4" x2="9" y2="9" />
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" />
    </svg>
  );
}

