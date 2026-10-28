import { useState, useMemo, useRef, useEffect } from "react";
import { useTrainingStore } from "../store/training-store";
import { useGraphStore } from "../store/graph-store";
import { toast } from "./Toast";
import NumberField from "./NumberField";
import {
  DATASET_METADATA,
  checkDatasetCompatibility,
  inferModelDomain,
  getRecommendedDatasets,
  isDatasetNode,
} from "../lib/dataset-metadata";

const OPTIMIZERS = ["SGD", "Adam", "AdamW", "RMSprop", "Adadelta", "Adagrad"];
const SCHEDULERS = ["None", "StepLR", "CosineAnnealingLR", "OneCycleLR", "ReduceLROnPlateau", "ExponentialLR"];
const LOSSES = ["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "L1Loss", "NLLLoss", "HuberLoss"];
const DATASETS = [
  "CIFAR10", "CIFAR100", "MNIST", "FashionMNIST",
  "TinyShakespeare", "WikiText2", "IMDB", "AGNews",
  "SpeechCommands",
  "ImageFolder", "Custom CSV", "HuggingFace",
];
const DEVICES = ["auto", "cpu", "cuda:0", "cuda:1"];
const WEIGHT_INITS = ["default", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal"];

export default function TrainingPanel() {
  const config = useTrainingStore((s) => s.config);
  const status = useTrainingStore((s) => s.status);
  const updateConfig = useTrainingStore((s) => s.updateConfig);
  const updateOptimizer = useTrainingStore((s) => s.updateOptimizer);
  const lastAppliedTemplate = useTrainingStore((s) => s.lastAppliedTemplate);
  const hasDatasetNode = useGraphStore((s) => s.nodes.some((n) => isDatasetNode(n.data.moduleType)));

  const locked = status === "running" || status === "starting";
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam);

  // Auto-sync: when dataset type changes, auto-fix Input node shape if it matches the old dataset
  const prevDatasetRef = useRef(config.dataset.type);
  useEffect(() => {
    const prev = prevDatasetRef.current;
    const next = config.dataset.type;
    prevDatasetRef.current = next;
    if (prev === next) return;

    const nodes = useGraphStore.getState().nodes;
    const inputNode = nodes.find((n) => n.data.moduleType === "Input");
    if (!inputNode) return;

    // Get previous dataset's expected shape
    const prevMeta = DATASET_METADATA[prev];
    if (!prevMeta) return;

    // Parse current input node shape
    let currentShape: number[] | null = null;
    try {
      const raw = String(inputNode.data.params.shape ?? "");
      currentShape = JSON.parse(raw.replace(/\(/g, "[").replace(/\)/g, "]"));
    } catch { /* empty */ }

    // Only auto-fix if the current shape matches the OLD dataset's expected shape
    if (!currentShape || JSON.stringify(currentShape) !== JSON.stringify(prevMeta.inputShape)) return;

    const nextMeta = DATASET_METADATA[next];
    if (!nextMeta) return;

    updateNodeParam(inputNode.id, "shape", JSON.stringify(nextMeta.inputShape));
    toast.info(`Input shape updated for ${next}: ${JSON.stringify(nextMeta.inputShape)}`);
  }, [config.dataset.type, updateNodeParam]);

  return (
    <div className="flex-1 overflow-y-auto" style={locked ? { pointerEvents: "none", opacity: 0.5 } : undefined}>
      {/* Template info banner */}
      {lastAppliedTemplate && (
        <div className="mx-4 mt-3 mb-1 px-3 py-2 rounded-lg text-[11px]" style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}>
          Config auto-filled for <span className="font-semibold">{lastAppliedTemplate}</span>. Adjust as needed.
        </div>
      )}

      {/* ── Data section: hidden when a Dataset node is in the graph ── */}
      {hasDatasetNode ? (
        <div className="mx-4 mt-3 mb-1 px-3 py-2 rounded-lg text-[11px]"
          style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}>
          Dataset configured via node in graph. Select the Dataset node to edit its properties.
        </div>
      ) : (
        <Section title="Data">
          <Field label="Dataset">
            <Select value={config.dataset.type} options={DATASETS}
              onChange={(v) => updateConfig({ dataset: { ...config.dataset, type: v } })} />
          </Field>
          <Field label="Data Path">
            <TextInput value={config.dataset.root}
              onChange={(v) => updateConfig({ dataset: { ...config.dataset, root: v } })} />
          </Field>
          <DatasetCompatibilityBanner datasetType={config.dataset.type} />
          <DatasetInfo datasetType={config.dataset.type} config={config} updateConfig={updateConfig} />
          <DomainRecommendations currentDataset={config.dataset.type} updateConfig={updateConfig} config={config} />
          <Field label="Validation Split">
            <NumberInput value={config.valSplit} step={0.05} min={0} max={0.5}
              onChange={(v) => updateConfig({ valSplit: v })} />
          </Field>
        </Section>
      )}

      {/* ── Always visible: Optimizer ── */}
      <Section title="Optimizer">
        <div className="grid grid-cols-2 gap-2">
          <Field label="Type">
            <Select value={config.optimizer.type} options={OPTIMIZERS}
              onChange={(v) => updateOptimizer({ type: v })} />
          </Field>
          <Field label="Learning Rate">
            <NumberInput value={config.optimizer.lr} step={0.0001} min={0}
              onChange={(v) => updateOptimizer({ lr: v })} />
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Weight Decay">
            <NumberInput value={config.optimizer.weightDecay} step={0.001} min={0}
              onChange={(v) => updateOptimizer({ weightDecay: v })} />
          </Field>
          {(config.optimizer.type === "SGD") && (
            <Field label="Momentum">
              <NumberInput value={config.optimizer.momentum ?? 0.9} step={0.01} min={0} max={1}
                onChange={(v) => updateOptimizer({ momentum: v })} />
            </Field>
          )}
        </div>
        <Field label="LR Scheduler">
          <Select
            value={config.scheduler?.type ?? "None"}
            options={SCHEDULERS}
            onChange={(v) => updateConfig({
              scheduler: v === "None" ? null : { type: v, params: {} },
            })}
          />
        </Field>
      </Section>

      {/* ── Always visible: Run ── */}
      <Section title="Run">
        <div className="grid grid-cols-2 gap-2">
          <Field label="Epochs">
            <NumberInput value={config.epochs} step={1} min={1}
              onChange={(v) => updateConfig({ epochs: Math.round(v) })} />
          </Field>
          <Field label="Batch Size">
            <NumberInput value={config.batchSize} step={1} min={1}
              onChange={(v) => updateConfig({ batchSize: Math.round(v) })} />
          </Field>
        </div>
        <Field label="Loss Function">
          <Select value={config.loss} options={LOSSES}
            onChange={(v) => updateConfig({ loss: v })} />
        </Field>
        <Field label="Device">
          <Select value={config.device} options={DEVICES}
            onChange={(v) => updateConfig({ device: v })} />
        </Field>
      </Section>

      {/* ── Collapsible: Data Augmentation — hidden when Dataset node owns it ── */}
      {!hasDatasetNode && (
        <CollapsibleSection title="Data Augmentation" defaultOpen={false}>
          <AugToggle label="Random Horizontal Flip" checked={config.augmentation.horizontalFlip}
            onChange={(v) => updateConfig({ augmentation: { ...config.augmentation, horizontalFlip: v } })} />
          <AugToggle label="Random Vertical Flip" checked={config.augmentation.verticalFlip}
            onChange={(v) => updateConfig({ augmentation: { ...config.augmentation, verticalFlip: v } })} />
          <AugToggle label="Random Crop" checked={config.augmentation.randomCrop}
            onChange={(v) => updateConfig({ augmentation: { ...config.augmentation, randomCrop: v } })} />
          <AugToggle label="Color Jitter" checked={config.augmentation.colorJitter}
            onChange={(v) => updateConfig({ augmentation: { ...config.augmentation, colorJitter: v } })} />
          <Field label="Random Rotation (degrees)">
            <NumberInput value={config.augmentation.rotation} step={5} min={0} max={180}
              onChange={(v) => updateConfig({ augmentation: { ...config.augmentation, rotation: Math.round(v) } })} />
          </Field>
        </CollapsibleSection>
      )}

      {/* ── Collapsible: Advanced ── */}
      <CollapsibleSection title={<>Advanced {config.seed != null && config.deterministicMode && (
        <span className="text-[8px] px-1.5 py-0.5 rounded-full"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}>Reproducible</span>
      )}</>} defaultOpen={false}>
        <Field label="Random Seed">
          <div className="flex gap-1.5">
            <input
              type="number"
              value={config.seed ?? ""}
              placeholder="Auto (set on first train)"
              onChange={(e) => {
                const v = e.target.value;
                updateConfig({ seed: v === "" ? null : parseInt(v, 10) || null });
              }}
              className={inputClass + " flex-1"}
              style={inputStyle}
            />
            <button
              onClick={() => updateConfig({ seed: Math.floor(Math.random() * 2147483647) })}
              className="px-2 py-[6px] text-xs rounded-lg shrink-0"
              style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
              title="Randomize seed"
            >
              🎲
            </button>
          </div>
        </Field>
        <Field label="Deterministic Mode">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={config.deterministicMode}
              onChange={(e) => updateConfig({ deterministicMode: e.target.checked })}
              className="accent-[var(--accent)]" />
            <span className="text-xs" style={{ color: "var(--text-primary)" }}>
              {config.deterministicMode ? "Enabled" : "Disabled"}
            </span>
          </label>
          {config.deterministicMode && (
            <div className="text-[10px] mt-1" style={{ color: "var(--text-muted)" }}>
              May reduce performance. Ensures identical results with same seed.
            </div>
          )}
        </Field>
        <Field label="Weight Initialization">
          <Select value={config.weightInit} options={WEIGHT_INITS}
            onChange={(v) => updateConfig({ weightInit: v })} />
        </Field>
        <Field label="Mixed Precision">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={config.mixedPrecision}
              onChange={(e) => updateConfig({ mixedPrecision: e.target.checked })}
              className="accent-[var(--accent)]" />
            <span className="text-xs" style={{ color: "var(--text-primary)" }}>
              {config.mixedPrecision ? "Enabled" : "Disabled"}
            </span>
          </label>
        </Field>
        <Field label="Gradient Clipping">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={config.gradientClipping.enabled}
              onChange={(e) => updateConfig({ gradientClipping: { ...config.gradientClipping, enabled: e.target.checked } })}
              className="accent-[var(--accent)]" />
            <span className="text-xs" style={{ color: "var(--text-primary)" }}>
              {config.gradientClipping.enabled ? "Enabled" : "Disabled"}
            </span>
          </label>
        </Field>
        {config.gradientClipping.enabled && (
          <Field label="Max Norm">
            <NumberInput value={config.gradientClipping.maxNorm} step={0.1} min={0}
              onChange={(v) => updateConfig({ gradientClipping: { ...config.gradientClipping, maxNorm: v } })} />
          </Field>
        )}
        <Field label="Early Stopping">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={config.earlyStopping.enabled}
              onChange={(e) => updateConfig({ earlyStopping: { ...config.earlyStopping, enabled: e.target.checked } })}
              className="accent-[var(--accent)]" />
            <span className="text-xs" style={{ color: "var(--text-primary)" }}>
              {config.earlyStopping.enabled ? "Enabled" : "Disabled"}
            </span>
          </label>
        </Field>
        {config.earlyStopping.enabled && (
          <>
            <Field label="Patience (epochs)">
              <NumberInput value={config.earlyStopping.patience} step={1} min={1}
                onChange={(v) => updateConfig({ earlyStopping: { ...config.earlyStopping, patience: Math.round(v) } })} />
            </Field>
            <Field label="Min Delta">
              <NumberInput value={config.earlyStopping.minDelta} step={0.0001} min={0}
                onChange={(v) => updateConfig({ earlyStopping: { ...config.earlyStopping, minDelta: v } })} />
            </Field>
          </>
        )}
        <Field label="Resume Checkpoint">
          <TextInput value={config.resumeCheckpoint}
            onChange={(v) => updateConfig({ resumeCheckpoint: v })} />
        </Field>
        <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>
          Leave empty to train from scratch
        </div>
      </CollapsibleSection>
    </div>
  );
}

function DatasetCompatibilityBanner({ datasetType }: { datasetType: string }) {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam);

  const result = useMemo(() => {
    const inputNode = nodes.find((n) => n.data.moduleType === "Input");
    let inputShape: number[] | null = null;
    if (inputNode) {
      const raw = String(inputNode.data.params.shape ?? "");
      try { inputShape = JSON.parse(raw.replace(/\(/g, "[").replace(/\)/g, "]")); } catch { /* empty */ }
    }

    // Find last Linear before Output for output features check
    const outgoing = new Map<string, string[]>();
    for (const e of edges) {
      const list = outgoing.get(e.source) ?? [];
      list.push(e.target);
      outgoing.set(e.source, list);
    }
    const outputIds = new Set(nodes.filter((n) => n.data.moduleType === "Output").map((n) => n.id));
    let outputFeatures: number | null = null;
    let finalLinearId: string | null = null;
    for (const n of nodes) {
      if (n.data.moduleType !== "Linear") continue;
      const targets = outgoing.get(n.id) ?? [];
      if (targets.some((t) => outputIds.has(t))) {
        outputFeatures = Number(n.data.params.out_features ?? 0) || null;
        finalLinearId = n.id;
        break;
      }
    }

    return { ...checkDatasetCompatibility(datasetType, inputShape, outputFeatures), inputNodeId: inputNode?.id, finalLinearId };
  }, [datasetType, nodes, edges]);

  if (result.status === "unknown") {
    return (
      <div className="text-[10px] rounded-md p-2" style={{ color: "var(--text-muted)", background: "var(--bg-tertiary)" }}>
        {result.message}
      </div>
    );
  }

  const colors = {
    match: { bg: "rgba(34, 197, 94, 0.08)", border: "rgba(34, 197, 94, 0.25)", text: "rgb(34, 197, 94)" },
    fixable: { bg: "rgba(245, 158, 11, 0.08)", border: "rgba(245, 158, 11, 0.25)", text: "rgb(245, 158, 11)" },
    mismatch: { bg: "rgba(239, 68, 68, 0.08)", border: "rgba(239, 68, 68, 0.25)", text: "rgb(239, 68, 68)" },
  };
  const c = colors[result.status];

  return (
    <div className="text-[10px] rounded-md p-2 flex items-start gap-1.5"
      style={{ background: c.bg, border: `1px solid ${c.border}`, color: c.text }}>
      <span className="shrink-0 mt-px">{result.status === "match" ? "\u2713" : result.status === "fixable" ? "\u26A0" : "\u2717"}</span>
      <div className="flex-1 min-w-0">
        <span>{result.message}</span>
        {result.status === "fixable" && result.suggestedInputShape && result.inputNodeId && (
          <button
            className="ml-2 px-2 py-1 rounded text-[10px] font-medium"
            style={{ background: "var(--accent)", color: "#fff" }}
            onClick={() => updateNodeParam(result.inputNodeId!, "shape", JSON.stringify(result.suggestedInputShape))}
          >
            Fix Input Shape
          </button>
        )}
        {result.suggestedOutputFeatures && result.finalLinearId && (
          <button
            className="ml-2 px-2 py-1 rounded text-[10px] font-medium"
            style={{ background: "var(--accent)", color: "#fff" }}
            onClick={() => updateNodeParam(result.finalLinearId!, "out_features", result.suggestedOutputFeatures!)}
          >
            Fix Output ({result.suggestedOutputFeatures} classes)
          </button>
        )}
      </div>
    </div>
  );
}

function DatasetInfo({ datasetType, config, updateConfig }: {
  datasetType: string;
  config: ReturnType<typeof useTrainingStore.getState>["config"];
  updateConfig: ReturnType<typeof useTrainingStore.getState>["updateConfig"];
}) {
  const meta = DATASET_METADATA[datasetType];

  if (datasetType === "HuggingFace") {
    return (
      <>
        <Field label="Dataset Name">
          <TextInput value={config.dataset.root}
            onChange={(v) => updateConfig({ dataset: { ...config.dataset, root: v } })} />
        </Field>
        <DatasetHint>
          Enter a HuggingFace dataset path, e.g. <span className="font-mono">mnist</span>, <span className="font-mono">cifar10</span>, <span className="font-mono">imdb</span>, <span className="font-mono">username/dataset</span>
        </DatasetHint>
      </>
    );
  }

  if (datasetType === "Custom CSV") {
    return (
      <DatasetHint>
        CSV format: feature columns first, last column = target label. First row = header (skipped).
        <br />Set Input shape to <span className="font-mono">[1, num_features]</span> and final Linear out_features = num_classes.
      </DatasetHint>
    );
  }

  if (!meta) return null;

  return (
    <DatasetHint>
      {meta.description} &middot; {meta.sizeHint}
      <br />
      <span className="font-mono">shape={JSON.stringify(meta.inputShape)}</span>
      {meta.numClasses > 0 && <>, <span className="font-mono">classes={meta.numClasses}</span></>}
      {meta.requiresPip && (
        <><br />Requires <span className="font-mono">pip install {meta.requiresPip}</span></>
      )}
    </DatasetHint>
  );
}

function DomainRecommendations({ currentDataset, updateConfig, config }: {
  currentDataset: string;
  updateConfig: ReturnType<typeof useTrainingStore.getState>["updateConfig"];
  config: ReturnType<typeof useTrainingStore.getState>["config"];
}) {
  const nodes = useGraphStore((s) => s.nodes);

  const recommendations = useMemo(() => {
    const domain = inferModelDomain(nodes);
    if (domain === "general") return [];
    return getRecommendedDatasets(domain).filter((d) => d !== currentDataset);
  }, [nodes, currentDataset]);

  if (recommendations.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>Recommended:</span>
      {recommendations.map((ds) => (
        <button
          key={ds}
          className="px-1.5 py-0.5 rounded text-[9px] font-medium"
          style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
          onClick={() => updateConfig({ dataset: { ...config.dataset, type: ds } })}
        >
          {ds}
        </button>
      ))}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <div className="px-4 pt-3 pb-1">
        <div className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
          {title}
        </div>
      </div>
      <div className="px-4 pb-3 space-y-2.5">
        {children}
      </div>
    </div>
  );
}

function CollapsibleSection({ title, defaultOpen, children }: { title: React.ReactNode; defaultOpen: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 pt-3 pb-1"
        style={{ color: "var(--text-muted)" }}
      >
        <div className="text-[11px] font-semibold uppercase tracking-wider">
          {title}
        </div>
        <svg
          width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: open ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.15s" }}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>
      {open && (
        <div className="px-4 pb-3 space-y-2.5">
          {children}
        </div>
      )}
    </div>
  );
}

function DatasetHint({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] leading-relaxed rounded-md p-2" style={{ color: "var(--text-muted)", background: "var(--bg-tertiary)" }}>
      {children}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[11px] font-medium mb-1" style={{ color: "var(--text-secondary)" }}>
        {label}
      </label>
      {children}
    </div>
  );
}

const inputClass = "w-full px-2.5 py-2 text-xs rounded-md";
const inputStyle = {
  background: "transparent",
  border: "1px solid var(--border)",
  color: "var(--text-primary)",
};

function Select({ value, options, onChange }: { value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={inputClass} style={inputStyle}>
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

function NumberInput({ value, step, min, max, onChange }: { value: number; step: number; min?: number; max?: number; onChange: (v: number) => void }) {
  return (
    <NumberField value={value} step={step} min={min} max={max} onChange={onChange}
      className={inputClass} style={inputStyle} />
  );
}

function TextInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <input type="text" value={value} onChange={(e) => onChange(e.target.value)}
      className={inputClass} style={inputStyle} />
  );
}

function AugToggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2.5 cursor-pointer py-0.5">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-[var(--accent)]" />
      <span className="text-xs" style={{ color: "var(--text-primary)" }}>{label}</span>
    </label>
  );
}
