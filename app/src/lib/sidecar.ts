/**
 * Python sidecar IPC layer.
 * The Rust backend spawns and manages the Python process.
 * Communication via Tauri commands + events.
 */

import type { JsonRpcResponse } from "./types";
import type { EnvironmentInfo } from "../store/training-store";
import { requirePython } from "../store/python-store";

let requestId = 0;
let sidecarStarted = false;
let sidecarReady = false;
let startupError: string | null = null;
let cachedEnvironment: EnvironmentInfo | null = null;
let pendingRequests = new Map<number, {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}>();

type ProgressCallback = (params: unknown) => void;
let onTrainProgress: ProgressCallback | null = null;

export function setTrainProgressCallback(cb: ProgressCallback | null) {
  onTrainProgress = cb;
}

type LogCallback = (params: { text: string; stream: "stdout" | "stderr" }) => void;
let onTrainLog: LogCallback | null = null;

export function setTrainLogCallback(cb: LogCallback | null) {
  onTrainLog = cb;
}

function handleStdoutLine(line: string) {
  if (!line.trim()) return;
  try {
    const msg = JSON.parse(line) as JsonRpcResponse & { method?: string; params?: unknown };

    if (!("id" in msg) || msg.id === undefined) {
      if (msg.method === "ready") {
        sidecarReady = true;
        const p = msg.params as Record<string, unknown> | undefined;
        if (p) {
          cachedEnvironment = {
            torchVersion: String(p.torch_version ?? ""),
            cudaVersion: p.cuda_version != null ? String(p.cuda_version) : null,
            pythonVersion: String(p.python_version ?? ""),
            platform: String(p.platform ?? ""),
            gpuName: p.gpu_name != null ? String(p.gpu_name) : null,
          };
        }
        // Sidecar ready
      }
      if (msg.method === "train.progress" && onTrainProgress) {
        onTrainProgress(msg.params);
      }
      if (msg.method === "train.log" && onTrainLog) {
        onTrainLog(msg.params as { text: string; stream: "stdout" | "stderr" });
      }
      return;
    }

    const pending = pendingRequests.get(msg.id);
    if (pending) {
      pendingRequests.delete(msg.id);
      if (msg.error) {
        pending.reject(new Error(msg.error.message));
      } else {
        pending.resolve(msg.result);
      }
    }
  } catch {
    // Non-JSON — could be Python print() or warnings
    // Non-JSON output from Python (print, warnings)
  }
}

async function ensureSidecar(): Promise<boolean> {
  if (sidecarReady) return true;

  // If already trying to start, wait
  if (sidecarStarted && !startupError) {
    return new Promise((resolve) => {
      const check = setInterval(() => {
        if (sidecarReady) { clearInterval(check); resolve(true); }
        if (startupError) { clearInterval(check); resolve(false); }
      }, 200);
      setTimeout(() => { clearInterval(check); resolve(sidecarReady); }, 20000);
    });
  }

  // Reset state
  sidecarStarted = true;
  startupError = null;

  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const { listen } = await import("@tauri-apps/api/event");

    // Listen for stdout/stderr events from Rust
    await listen<string>("sidecar-stdout", (event) => {
      handleStdoutLine(event.payload);
    });

    await listen<string>("sidecar-stderr", (event) => {
      const line = event.payload.trim();
      if (line) {
        console.warn("[sidecar stderr]", line);
        // Check for fatal errors
        if (line.includes("ModuleNotFoundError") || line.includes("No module named")) {
          startupError = `Python module error: ${line}`;
        }
      }
    });

    // Start the sidecar via Rust
    const result = await invoke<string>("start_sidecar");

    if (result === "already_running") {
      // Sidecar process is alive — if we missed the "ready" event, assume it's ready
      sidecarReady = true;
      return true;
    }

    // Wait for "ready" notification (torch import takes 5-15s)
    return new Promise((resolve) => {
      let elapsed = 0;
      const check = setInterval(() => {
        elapsed += 300;
        if (sidecarReady) {
          clearInterval(check);
          resolve(true);
        }
        if (startupError) {
          clearInterval(check);
          console.error("[sidecar] Startup error:", startupError);
          // Reset so next attempt can retry fresh
          sidecarStarted = false;
          resolve(false);
        }
        if (elapsed >= 20000) {
          clearInterval(check);
          startupError = "Sidecar startup timed out (20s). Check Python/PyTorch installation.";
          console.error("[sidecar]", startupError);
          // Reset so next attempt can retry fresh
          sidecarStarted = false;
          resolve(false);
        }
      }, 300);
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[sidecar] Failed to start:", msg);
    startupError = msg;
    sidecarStarted = false;
    return false;
  }
}

/** What to tell people when a call can't run because Python isn't set up */
function needsPythonReason(method: string): string {
  if (method === "compile") return "Turning code into a graph needs Python.";
  if (method.startsWith("data.")) return "Reading datasets needs Python.";
  if (method.startsWith("export.")) return "Exporting needs Python.";
  if (method.startsWith("train.")) return "Training needs Python.";
  return "This needs Python.";
}

export async function sidecarCall<T = unknown>(method: string, params: unknown = {}): Promise<T> {
  if (!requirePython(needsPythonReason(method))) {
    throw new Error("Python isn't set up yet. Set it up from the card at the top right of the canvas.");
  }
  const ready = await ensureSidecar();
  if (!ready) {
    const detail = startupError || "Unknown error";
    throw new Error(`Sidecar not available: ${detail}`);
  }

  const { invoke } = await import("@tauri-apps/api/core");
  const id = ++requestId;
  const request = JSON.stringify({ jsonrpc: "2.0", method, params, id });

  return new Promise<T>((resolve, reject) => {
    pendingRequests.set(id, { resolve: resolve as (v: unknown) => void, reject });

    invoke("sidecar_write", { data: request }).catch((e) => {
      pendingRequests.delete(id);
      reject(new Error(`Failed to write to sidecar: ${e}`));
    });

    setTimeout(() => {
      if (pendingRequests.has(id)) {
        pendingRequests.delete(id);
        reject(new Error(`Sidecar call "${method}" timed out after 60s`));
      }
    }, 60000);
  });
}

export async function killSidecar() {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("stop_sidecar");
  } catch { /* ignore */ }
  sidecarStarted = false;
  sidecarReady = false;
  startupError = null;
}

export function isSidecarReady(): boolean {
  return sidecarReady;
}

export function getSidecarError(): string | null {
  return startupError;
}

export function getCachedEnvironment(): EnvironmentInfo | null {
  return cachedEnvironment;
}

export function setCachedEnvironment(env: EnvironmentInfo | null) {
  cachedEnvironment = env;
}
