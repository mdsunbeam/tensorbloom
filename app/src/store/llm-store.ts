/**
 * The pretrained LLM a project was imported from, if any. Saved with the project.
 * The Hugging Face token is kept for this session only and never saved.
 */

import { create } from "zustand";
import type { LLMSpec, LLMDevice } from "../lib/llm";

interface LLMState {
  spec: LLMSpec | null;
  /** Token for private or gated Hugging Face models, this session only */
  hfToken: string;
  device: LLMDevice | null;
  setSpec: (spec: LLMSpec | null) => void;
  setHfToken: (token: string) => void;
  setDevice: (device: LLMDevice | null) => void;
}

export const useLLMStore = create<LLMState>((set) => ({
  spec: null,
  hfToken: "",
  device: null,
  setSpec: (spec) => set({ spec }),
  setHfToken: (hfToken) => set({ hfToken }),
  setDevice: (device) => set({ device }),
}));
