/**
 * Opt-in, anonymous usage events. Nothing is sent until the user says yes,
 * and nothing is sent at all in builds without VITE_USAGE_ENDPOINT.
 *
 * An event carries only: its name, a random install id, the app version, the
 * OS family, minutes since this install first ran, and the few properties
 * listed in UsageEvent. Never graphs, code, file paths, datasets, or names.
 * docs/guides/privacy.md lists the same thing for users; keep the two in step.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { version } from "../../package.json";

export type UsageConsent = "unset" | "granted" | "denied";

export type UsageEvent =
  | { name: "app_opened"; props: { first_run: boolean } }
  | { name: "pytorch_installed"; props: Record<string, never> }
  | { name: "training_finished"; props: { first: boolean; stopped_early: boolean; epochs: number; seconds: number; device: string } }
  | { name: "try_it_opened"; props: { first: boolean } };

export interface UsagePayload {
  event: UsageEvent["name"];
  props: UsageEvent["props"];
  install_id: string;
  app_version: string;
  os: "windows" | "macos" | "linux" | "other";
  minutes_since_install: number;
}

const ENDPOINT: string = import.meta.env.VITE_USAGE_ENDPOINT ?? "";

/** Whether this build can send usage events at all */
export const usageAvailable = ENDPOINT.length > 0;

interface UsageState {
  consent: UsageConsent;
  installId: string;
  firstSeenAt: number;
  /** Once-only milestones already counted, so "first" is only ever true once */
  seen: Partial<Record<UsageEvent["name"], true>>;
  setConsent: (consent: UsageConsent) => void;
}

function randomId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return Math.random().toString(36).slice(2) + Date.now().toString(36);
  }
}

const STORAGE_KEY = "tensorbloom-usage";

// Nothing stored yet means this is the install's very first run
const firstRun: boolean = (() => {
  try {
    return localStorage.getItem(STORAGE_KEY) === null;
  } catch {
    return false;
  }
})();

export const useUsageStore = create<UsageState>()(
  persist(
    (set) => ({
      consent: "unset",
      installId: randomId(),
      firstSeenAt: Date.now(),
      seen: {},
      setConsent: (consent) => set({ consent }),
    }),
    { name: STORAGE_KEY },
  ),
);

export function osFamily(ua: string = typeof navigator !== "undefined" ? navigator.userAgent : ""): UsagePayload["os"] {
  if (/Windows/i.test(ua)) return "windows";
  if (/Mac/i.test(ua)) return "macos";
  if (/Linux|X11/i.test(ua)) return "linux";
  return "other";
}

export function buildPayload(e: UsageEvent, state: Pick<UsageState, "installId" | "firstSeenAt">, now = Date.now()): UsagePayload {
  return {
    event: e.name,
    props: e.props,
    install_id: state.installId,
    app_version: version,
    os: osFamily(),
    minutes_since_install: Math.max(0, Math.round((now - state.firstSeenAt) / 60_000)),
  };
}

/** Sends one event if the user opted in. Never throws and never blocks the UI. */
export function track(e: UsageEvent) {
  const state = useUsageStore.getState();
  if (!usageAvailable || state.consent !== "granted") return;
  const body = JSON.stringify(buildPayload(e, state));
  try {
    void fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    }).catch(() => {});
  } catch {
    // Offline or blocked: usage events are best-effort
  }
}

/** Whether this is the first time a milestone happens on this install, and records that it has */
export function firstTime(name: UsageEvent["name"]): boolean {
  const { seen } = useUsageStore.getState();
  if (seen[name]) return false;
  useUsageStore.setState({ seen: { ...seen, [name]: true } });
  return true;
}

export function trackAppOpened() {
  track({ name: "app_opened", props: { first_run: firstRun } });
}

/** Device family only (cpu, cuda, mps), never a GPU's name */
export function deviceFamily(device: string | undefined): string {
  const d = (device ?? "").toLowerCase();
  if (d.startsWith("cuda")) return "cuda";
  if (d.startsWith("mps")) return "mps";
  if (d === "cpu") return "cpu";
  return "auto";
}

/** The OS as people write it, for the feedback form */
export function osName(): string {
  return { windows: "Windows", macos: "macOS", linux: "Linux", other: "Other" }[osFamily()];
}
