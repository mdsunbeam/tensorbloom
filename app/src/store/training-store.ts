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
  durationMs: number;
  environment?: EnvironmentInfo;
}

export interface LogEntry {
  text: string;
  stream: "stdout" | "stderr";
  timestamp: number;
}

type TrainingStatus = "idle" | "starting" | "running" | "paused" | "finished" | "error";

interface TrainingState {
  config: TrainingConfig;
  status: TrainingStatus;
  currentEpoch: number;
  metrics: TrainingMetrics[];
  logs: LogEntry[];
  error: string | null;
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
  setError: (error: string | null) => void;
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

  setError: (error) => set({ error, status: error ? "error" : get().status }),

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
  reset: () => set({ metrics: [], logs: [], currentEpoch: 0, error: null, status: "idle", startedAt: null, trainingResults: null }),
}));
