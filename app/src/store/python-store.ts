import { create } from "zustand";

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
}

type SetupUpdate =
  | { kind: "step"; step: number; total: number; label: string }
  | { kind: "download"; done: number; total: number }
  | { kind: "log"; line: string };

interface PythonState {
  status: PythonStatus;
  /** The setup step running now */
  step: { step: number; total: number; label: string } | null;
  /** The file pip is downloading, when it's big enough for pip to report progress */
  download: { name: string; done: number; total: number } | null;
  log: string[];
  error: string | null;
  venvPath: string;
  /** Whether the setup card shows on the canvas */
  cardOpen: boolean;
  /** Why the card was opened, e.g. "Training needs Python first" */
  cardReason: string | null;
  /** Set for a few seconds after setup succeeds, so the card can say so */
  justInstalled: boolean;
  check: () => Promise<void>;
  install: () => Promise<void>;
  openCard: (reason?: string) => void;
  closeCard: () => void;
}

const LOG_LIMIT = 400;

/** "https://…/torch-2.6.0%2Bcu126-cp312-….whl (2.5 GB)" → "torch" */
export function packageFromDownloadLine(line: string): string | null {
  const m = /^\s*Downloading (\S+)/.exec(line);
  if (!m || m[1].endsWith(".metadata")) return null;
  let file = m[1].split("/").pop() ?? m[1];
  try { file = decodeURIComponent(file); } catch { /* keep it as is */ }
  return file.split("-")[0] || null;
}

export const usePythonStore = create<PythonState>((set, get) => ({
  status: "checking",
  step: null,
  download: null,
  log: [],
  error: null,
  venvPath: "",
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
      set({ status: env.ready ? "ready" : "missing", venvPath: env.venv_path, cardOpen: !env.ready });
    } catch {
      // Not running in Tauri
      set({ status: "ready" });
    }
  },

  install: async () => {
    if (get().status === "installing") return;
    set({ status: "installing", step: null, download: null, log: [], error: null, cardOpen: true });
    const { invoke } = await import("@tauri-apps/api/core");
    const { listen } = await import("@tauri-apps/api/event");
    const unlisten = await listen<SetupUpdate>("python-setup", ({ payload: u }) => {
      if (u.kind === "step") {
        set({ step: { step: u.step, total: u.total, label: u.label }, download: null });
      } else if (u.kind === "download") {
        const d = get().download;
        if (d) set({ download: { ...d, done: u.done, total: u.total } });
      } else {
        const name = packageFromDownloadLine(u.line);
        const log = get().log;
        set({
          log: log.length >= LOG_LIMIT ? [...log.slice(1 - LOG_LIMIT), u.line] : [...log, u.line],
          ...(name ? { download: { name, done: 0, total: 0 } } : {}),
        });
      }
    });
    try {
      const env = await invoke<EnvStatus>("setup_python_env");
      if (env.ready) {
        set({ status: "ready", venvPath: env.venv_path, download: null, justInstalled: true, cardReason: null });
        setTimeout(() => set({ justInstalled: false, cardOpen: false }), 4000);
      } else {
        set({ status: "failed", error: env.error ?? "PyTorch didn't install correctly.", download: null });
      }
    } catch (e) {
      set({ status: "failed", error: e instanceof Error ? e.message : String(e), download: null });
    } finally {
      unlisten();
    }
  },

  openCard: (reason) => set({ cardOpen: true, cardReason: reason ?? null }),
  closeCard: () => set({ cardOpen: false, cardReason: null }),
}));

/** True when Python is set up. Otherwise opens the setup card, explaining `reason`, and returns false. */
export function requirePython(reason: string): boolean {
  const s = usePythonStore.getState();
  if (s.status === "ready") return true;
  s.openCard(reason);
  return false;
}
