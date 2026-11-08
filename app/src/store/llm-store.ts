/**
 * The pretrained LLM a project was imported from, if any (saved with the project), and the state of
 * running it in the sidecar: packages, the loaded model, and the chat (this session only).
 * The Hugging Face token is kept for this session only and never saved.
 */

import { create } from "zustand";
import type { LLMSpec, LLMDevice, Precision, FinetuneSettings } from "../lib/llm";
import { DEFAULT_FINETUNE } from "../lib/llm";

export interface LoadedModel {
  name: string;
  path: string;
  precision: Precision;
  device: "cuda" | "mps" | "cpu";
  device_name: string;
  memory_used: number;
  /** Path of the LoRA adapter applied on top, after fine-tuning */
  adapter: string | null;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
  /** For replies: speed and how it ended */
  stats?: { tokens: number; seconds: number; tokensPerSecond: number | null; stopped?: boolean };
  error?: string;
  /** Set when the reply came from a fine-tuned adapter */
  adapter?: string | null;
}

export interface LoadProgress {
  status: "downloading" | "loading";
  bytes?: number;
  total?: number | null;
}

interface LLMState {
  spec: LLMSpec | null;
  /** Token for private or gated Hugging Face models, this session only */
  hfToken: string;
  device: LLMDevice | null;
  /** Packages still to install, or null until checked */
  missing: string[] | null;
  install: { status: "idle" | "running" | "done" | "error"; lines: string[]; error?: string };
  loaded: LoadedModel | null;
  loading: LoadProgress | null;
  loadError: string | null;
  chat: ChatMessage[];
  generating: boolean;
  systemPrompt: string;
  maxNewTokens: number;
  temperature: number;
  /** Dataset and LoRA settings for fine-tuning, saved with the project */
  finetune: FinetuneSettings;
  /** Which part of the Language model tab is showing */
  view: "overview" | "chat" | "finetune";

  setSpec: (spec: LLMSpec | null) => void;
  setHfToken: (token: string) => void;
  setDevice: (device: LLMDevice | null) => void;
}

export const useLLMStore = create<LLMState>((set) => ({
  spec: null,
  hfToken: "",
  device: null,
  missing: null,
  install: { status: "idle", lines: [] },
  loaded: null,
  loading: null,
  loadError: null,
  chat: [],
  generating: false,
  systemPrompt: "",
  maxNewTokens: 512,
  temperature: 0.7,
  finetune: DEFAULT_FINETUNE,
  view: "overview",

  // A different model starts a fresh chat
  setSpec: (spec) => set((s) => (s.spec?.name === spec?.name ? { spec } : { spec, chat: [], loadError: null, finetune: DEFAULT_FINETUNE, view: "overview" })),
  setHfToken: (hfToken) => set({ hfToken }),
  setDevice: (device) => set({ device }),
}));
