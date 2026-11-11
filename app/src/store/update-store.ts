/**
 * In-app updates through the Tauri updater plugin. The app checks the latest
 * GitHub release once at startup; the user decides when to install.
 */
import { create } from "zustand";
import type { Update } from "@tauri-apps/plugin-updater";
import { writeAutoSave } from "../lib/auto-save";
import { toast } from "../components/Toast";

export type UpdateStatus = "idle" | "checking" | "available" | "downloading" | "installing" | "error";

interface UpdateState {
  status: UpdateStatus;
  version: string | null;
  notes: string | null;
  /** Download progress from 0 to 1, or null while the size is unknown */
  progress: number | null;
  error: string | null;
  /** The user chose "Later" for this version */
  dismissed: boolean;
  check: (opts?: { manual?: boolean }) => Promise<void>;
  install: () => Promise<void>;
  dismiss: () => void;
}

let pending: Update | null = null;

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export const useUpdateStore = create<UpdateState>()((set, get) => ({
  status: "idle",
  version: null,
  notes: null,
  progress: null,
  error: null,
  dismissed: false,

  check: async ({ manual = false } = {}) => {
    if (!isTauri()) {
      if (manual) toast.info("Updates are only available in the desktop app.");
      return;
    }
    const { status } = get();
    if (status === "checking" || status === "downloading" || status === "installing") return;
    set({ status: "checking", error: null });
    try {
      const { check } = await import("@tauri-apps/plugin-updater");
      const update = await check();
      if (!update) {
        pending = null;
        set({ status: "idle", version: null, notes: null });
        if (manual) toast.success("TensorBloom is up to date.");
        return;
      }
      pending = update;
      set({ status: "available", version: update.version, notes: update.body ?? null, dismissed: false });
    } catch (e) {
      // A quiet check that fails (offline, rate-limited) shouldn't bother anyone
      set({ status: "idle", error: String(e) });
      if (manual) toast.error("Couldn't check for updates. Check your internet connection and try again.");
    }
  },

  install: async () => {
    const update = pending;
    if (!update || get().status !== "available") return;
    // The restart skips the unsaved-changes prompt, so keep a recovery copy
    writeAutoSave();
    set({ status: "downloading", progress: null });
    try {
      let total = 0;
      let received = 0;
      await update.downloadAndInstall((event) => {
        if (event.event === "Started") {
          total = event.data.contentLength ?? 0;
        } else if (event.event === "Progress") {
          received += event.data.chunkLength;
          set({ progress: total > 0 ? Math.min(1, received / total) : null });
        } else if (event.event === "Finished") {
          set({ status: "installing", progress: 1 });
        }
      });
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (e) {
      set({ status: "available", progress: null, error: String(e) });
      toast.error("The update didn't install. Try again, or download it from the releases page.");
    }
  },

  dismiss: () => set({ dismissed: true }),
}));
