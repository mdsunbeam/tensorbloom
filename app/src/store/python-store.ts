import { create } from "zustand";
import { useTrainingStore } from "./training-store";
import { killSidecar, setCachedEnvironment } from "../lib/sidecar";

/**
 * The Python environment that training and code import run in. The editor works without it;
 * it's checked at startup and set up from a card on the canvas when someone wants to train.
 */
export type PythonStatus = "checking" | "ready" | "missing" | "installing" | "failed";

interface EnvStatus {
  ready: boolean;
  venv_path: string;
  error: string | null;
  stage: string;
  torch_version: string | null;
  torch_cuda: boolean;
  cuda_upgrade_supported: boolean;
}

type SetupUpdate =
  | { kind: "step"; step: number; total: number; label: string }
  | { kind: "download"; done: number; total: number }
  | { kind: "log"; line: string };

/** "setup": Python and the CPU build of PyTorch. "gpu": swapping in the CUDA build. */
export type InstallTask = "setup" | "gpu";

interface PythonState {
  status: PythonStatus;
  task: InstallTask;
  /** The setup step running now */
  step: { step: number; total: number; label: string } | null;
  /** Bytes downloaded so far of the packages the current step downloads */
  download: { done: number; total: number } | null;
  log: string[];
  error: string | null;
  venvPath: string;
  /** Whether the installed PyTorch can use an NVIDIA GPU */
  torchCuda: boolean;
  /** Whether the CUDA build is offered on this platform (Windows and Linux) */
  cudaUpgradeSupported: boolean;
  /** The NVIDIA GPU nvidia-smi found, if any */
  nvidiaGpu: string | null;
  /** Whether the card shows on the canvas */
  cardOpen: boolean;
  /** Why the card was opened, e.g. "Training needs Python first" */
  cardReason: string | null;
  /** Set after an install succeeds, so the card can say so */
  justInstalled: boolean;
  check: () => Promise<void>;
  install: () => Promise<void>;
  installGpu: () => Promise<void>;
  openCard: (reason?: string) => void;
  closeCard: () => void;
}

const LOG_LIMIT = 400;

/** Whether to offer the CUDA build: PyTorch is set up as the CPU build and there's an NVIDIA GPU to use. */
export function gpuUpgradeAvailable(s: Pick<PythonState, "status" | "torchCuda" | "cudaUpgradeSupported" | "nvidiaGpu">): boolean {
  return s.status === "ready" && !s.torchCuda && s.cudaUpgradeSupported && s.nvidiaGpu != null;
}

function envFields(env: EnvStatus) {
  return { venvPath: env.venv_path, torchCuda: env.torch_cuda, cudaUpgradeSupported: env.cuda_upgrade_supported };
}

export const usePythonStore = create<PythonState>((set, get) => {
  /** Runs one of the install commands, following its progress events. */
  const runInstall = async (task: InstallTask, command: string) => {
    if (get().status === "installing") return;
    set({ status: "installing", task, step: null, download: null, log: [], error: null, cardOpen: true, justInstalled: false });
    const { invoke } = await import("@tauri-apps/api/core");
    const { listen } = await import("@tauri-apps/api/event");
    const unlisten = await listen<SetupUpdate>("python-setup", ({ payload: u }) => {
      if (u.kind === "step") {
        set({ step: { step: u.step, total: u.total, label: u.label }, download: null });
      } else if (u.kind === "download") {
        set({ download: { done: u.done, total: u.total } });
      } else {
        const log = get().log;
        set({ log: log.length >= LOG_LIMIT ? [...log.slice(1 - LOG_LIMIT), u.line] : [...log, u.line] });
      }
    });
    try {
      const env = await invoke<EnvStatus>(command);
      if (env.ready) {
        set({ status: "ready", ...envFields(env), error: null, download: null, justInstalled: true, cardReason: null });
        // Stay open when there's a GPU to offer; otherwise say it worked, then get out of the way
        if (!gpuUpgradeAvailable(get())) setTimeout(() => set({ justInstalled: false, cardOpen: false }), 4000);
      } else {
        set({ status: "failed", error: env.error ?? "PyTorch didn't install correctly.", download: null });
      }
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      if (task === "gpu") {
        // The CPU build usually still works; keep training available if it does
        let env: EnvStatus | null = null;
        try { env = await invoke<EnvStatus>("check_python_env"); } catch { /* treat as broken */ }
        set({ status: env?.ready ? "ready" : "failed", ...(env ? envFields(env) : {}), error, download: null });
      } else {
        set({ status: "failed", error, download: null });
      }
    } finally {
      unlisten();
    }
  };

  return {
    status: "checking",
    task: "setup",
    step: null,
    download: null,
    log: [],
    error: null,
    venvPath: "",
    torchCuda: false,
    cudaUpgradeSupported: false,
    nvidiaGpu: null,
    cardOpen: false,
    cardReason: null,
    justInstalled: false,

    check: async () => {
      let invoke: typeof import("@tauri-apps/api/core").invoke;
      try {
        ({ invoke } = await import("@tauri-apps/api/core"));
      } catch {
        // Browser dev mode: there's no Python to set up
        set({ status: "ready" });
        return;
      }
      try {
        const env = await invoke<EnvStatus>("check_python_env");
        set({ status: env.ready ? "ready" : "missing", ...envFields(env), cardOpen: !env.ready });
      } catch {
        // Not running in Tauri
        set({ status: "ready" });
      }
    },

    install: () => runInstall("setup", "setup_python_env"),

    installGpu: async () => {
      if (["starting", "running", "paused"].includes(useTrainingStore.getState().status)) {
        set({ task: "gpu", cardOpen: true, error: "Stop training before adding GPU support." });
        return;
      }
      // The running sidecar has PyTorch loaded, and Windows won't replace files in use
      await killSidecar();
      setCachedEnvironment(null);
      await runInstall("gpu", "install_gpu_support");
    },

    openCard: (reason) => set({ cardOpen: true, cardReason: reason ?? null }),
    closeCard: () => set({ cardOpen: false, cardReason: null, justInstalled: false, ...(get().status === "ready" ? { error: null } : {}) }),
  };
});

/** True when Python is set up. Otherwise opens the setup card, explaining `reason`, and returns false. */
export function requirePython(reason: string): boolean {
  const s = usePythonStore.getState();
  if (s.status === "ready") return true;
  s.openCard(reason);
  return false;
}
