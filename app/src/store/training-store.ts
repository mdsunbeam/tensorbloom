import { create } from "zustand";

export interface TrainingMetrics {
  epoch: number;
  step: number;
  trainLoss: number;
  valLoss?: number;
  trainAcc?: number;
  valAcc?: number;
  lr: number;
  timestamp: number;
  checkpoint?: string;
  gpu?: { allocated_mb: number; reserved_mb: number; max_allocated_mb: number };
}

export interface TrainingConfig {
  optimizer: { type: string; lr: number; weightDecay: number; momentum?: number };
  scheduler: { type: string; params: Record<string, unknown> } | null;
  loss: string;
  epochs: number;
  batchSize: number;
  valSplit: number;
  dataset: { type: string; root: string };
  device: string;
  mixedPrecision: boolean;
  gradientClipping: { enabled: boolean; maxNorm: number };
  earlyStopping: { enabled: boolean; patience: number; minDelta: number };
  resumeCheckpoint: string;
  augmentation: {
    horizontalFlip: boolean;
    verticalFlip: boolean;
    randomCrop: boolean;
    colorJitter: boolean;
    rotation: number;
  };
  weightInit: string;
  seed: number | null;
  dataloaderSeed: number | null;
  deterministicMode: boolean;
}

export interface EnvironmentInfo {
  torchVersion: string;
  cudaVersion: string | null;
  pythonVersion: string;
  platform: string;
  gpuName: string | null;
}

export interface TrainingResults {
  startedAt: string;
  completedAt: string;
  totalEpochs: number;
  finalTrainLoss: number;
  finalValLoss?: number;
  bestValLoss?: number;
  bestValEpoch?: number;
  finalTrainAcc?: number;
  finalValAcc?: number;
  bestValAcc?: number;
  bestValAccEpoch?: number;
  /** The run ended before its last epoch because validation stopped improving */
  earlyStopped?: boolean;
  /** The run ended before its last epoch because Stop was pressed */
  stoppedByUser?: boolean;
  /** Epochs the run was set up for */
  plannedEpochs?: number;
  durationMs: number;
  environment?: EnvironmentInfo;
}

export interface LogEntry {
  text: string;
  stream: "stdout" | "stderr";
  timestamp: number;
}

export type TrainingStatus = "idle" | "starting" | "running" | "paused" | "finished" | "error";

/** "check": the pre-flight check stopped the run before it started. "run": PyTorch raised during the run. */
export type TrainingErrorKind = "check" | "run";

interface TrainingState {
  config: TrainingConfig;
  status: TrainingStatus;
  currentEpoch: number;
  metrics: TrainingMetrics[];
  logs: LogEntry[];
  error: string | null;
  errorKind: TrainingErrorKind;
  /** Batches done in the current epoch, when the sidecar reports them */
  epochProgress: { step: number; total: number } | null;
  /** Name of the dataset being downloaded before the first epoch */
  downloading: string | null;
  stopRequested: boolean;
  startedAt: number | null;
  lastAppliedTemplate: string | null;
  trainingResults: TrainingResults | null;
  cachedEnvironment: EnvironmentInfo | null;
  lastCheckpointPath: string | null;
  updateConfig: (partial: Partial<TrainingConfig>) => void;
  updateOptimizer: (partial: Partial<TrainingConfig["optimizer"]>) => void;
  setStatus: (status: TrainingStatus) => void;
  appendMetric: (m: TrainingMetrics) => void;
  appendLog: (entry: LogEntry) => void;
  setError: (error: string | null, kind?: TrainingErrorKind) => void;
  setEpochProgress: (progress: { step: number; total: number } | null) => void;
  setDownloading: (dataset: string | null) => void;
  setStopRequested: (requested: boolean) => void;
  applyTemplateDefaults: (name: string, defaults: Partial<TrainingConfig>) => void;
  setTrainingResults: (results: TrainingResults | null) => void;
  setCachedEnvironment: (env: EnvironmentInfo | null) => void;
  setLastCheckpointPath: (path: string | null) => void;
  clearLogs: () => void;
  reset: () => void;
}

const DEFAULT_CONFIG: TrainingConfig = {
  optimizer: { type: "AdamW", lr: 0.001, weightDecay: 0.01 },
  scheduler: { type: "CosineAnnealingLR", params: { T_max: 10 } },
  loss: "CrossEntropyLoss",
  epochs: 10,
  batchSize: 32,
  valSplit: 0.2,
  dataset: { type: "CIFAR10", root: "./data" },
  device: "auto",
  mixedPrecision: false,
  gradientClipping: { enabled: false, maxNorm: 1.0 },
  earlyStopping: { enabled: false, patience: 5, minDelta: 0.001 },
  resumeCheckpoint: "",
  augmentation: {
    horizontalFlip: true,
    verticalFlip: false,
    randomCrop: true,
    colorJitter: false,
    rotation: 0,
  },
  weightInit: "default",
  seed: null,
  dataloaderSeed: null,
  deterministicMode: false,
};

export const useTrainingStore = create<TrainingState>((set, get) => ({
  config: { ...DEFAULT_CONFIG },
  status: "idle",
  currentEpoch: 0,
  metrics: [],
  logs: [],
  error: null,
  errorKind: "run",
  epochProgress: null,
  downloading: null,
  stopRequested: false,
  startedAt: null,
  lastAppliedTemplate: null,
  trainingResults: null,
  cachedEnvironment: null,
  lastCheckpointPath: null,

  updateConfig: (partial) =>
    set({ config: { ...get().config, ...partial }, lastAppliedTemplate: null }),

  updateOptimizer: (partial) =>
    set({ config: { ...get().config, optimizer: { ...get().config.optimizer, ...partial } }, lastAppliedTemplate: null }),

  setStatus: (status) => set({
    status,
    startedAt: status === "running" ? Date.now() : (status === "idle" ? null : get().startedAt),
  }),

  appendMetric: (m) => {
    set({ metrics: [...get().metrics, m], currentEpoch: m.epoch });
  },
  appendLog: (entry) => {
    const logs = get().logs;
    // Cap at 1000 lines to prevent memory bloat
    set({ logs: logs.length >= 1000 ? [...logs.slice(-999), entry] : [...logs, entry] });
  },

  setError: (error, kind = "run") => set({ error, errorKind: kind, status: error ? "error" : get().status, downloading: null }),
  setEpochProgress: (epochProgress) => set({ epochProgress }),
  setDownloading: (downloading) => set({ downloading }),
  setStopRequested: (stopRequested) => set({ stopRequested }),

  applyTemplateDefaults: (name, defaults) => {
    const current = get().config;
    set({
      config: { ...current, ...defaults },
      lastAppliedTemplate: name,
    });
  },

  setTrainingResults: (results) => set({ trainingResults: results }),

  setCachedEnvironment: (env) => set({ cachedEnvironment: env }),

  setLastCheckpointPath: (path) => set({ lastCheckpointPath: path }),

  clearLogs: () => set({ logs: [] }),
  reset: () => set({
    metrics: [], logs: [], currentEpoch: 0, error: null, errorKind: "run", status: "idle", startedAt: null, trainingResults: null,
    epochProgress: null, downloading: null, stopRequested: false,
  }),
}));

/** How far the run is, from 0 to 1, counting batches inside the current epoch when they're known */
export function runFraction(s: Pick<TrainingState, "currentEpoch" | "epochProgress" | "config">): number {
  const epochs = s.config.epochs;
  if (epochs <= 0 || s.currentEpoch <= 0) return 0;
  const within = s.epochProgress && s.epochProgress.total > 0 ? Math.min(1, s.epochProgress.step / s.epochProgress.total) : 1;
  return Math.min(1, (s.currentEpoch - 1 + within) / epochs);
}
