import { create } from "zustand";
import { DEFAULT_MODEL, MODELS, keyStatus, type KeyStatus, type MessageParam } from "../lib/assistant/api";
import { runTurn, type TurnOutcome } from "../lib/assistant/agent";
import { READ_TOOLS } from "../lib/assistant/read-tools";
import { EDIT_TOOLS } from "../lib/assistant/edit-tools";
import { CODE_TOOLS } from "../lib/assistant/code-tools";
import { takeDecision } from "../lib/assistant/change-set";
import type { AssistantTool, ApprovalRequest } from "../lib/assistant/tool-types";

export interface ToolActivity {
  id: string;
  label: string;
  status: "running" | "done" | "error";
  detail?: string;
}

/** What the chat shows. The API transcript is kept separately, since it carries app snapshots and tool results. */
export type ChatItem =
  | { id: string; kind: "user"; text: string }
  | { id: string; kind: "assistant"; text: string }
  | { id: string; kind: "tools"; tools: ToolActivity[] }
  | { id: string; kind: "notice"; tone: "info" | "error"; text: string }
  | { id: string; kind: "approval"; request: ApprovalRequest; status: "pending" | "approved" | "declined" };

const MODEL_STORAGE = "tensorbloom.assistant.model";

function storedModel(): string {
  try {
    const m = localStorage.getItem(MODEL_STORAGE);
    if (m && MODELS.some((x) => x.id === m)) return m;
  } catch { /* storage unavailable */ }
  return DEFAULT_MODEL;
}

interface AssistantState {
  items: ChatItem[];
  transcript: MessageParam[];
  busy: boolean;
  model: string;
  key: KeyStatus | null;
  /** Text to put in the message box, e.g. from an "Ask the assistant" button elsewhere */
  draft: string;
  setDraft: (text: string) => void;
  setModel: (model: string) => void;
  refreshKey: () => Promise<void>;
  send: (text: string) => Promise<void>;
  stop: () => void;
  reset: () => void;
  /** The user's answer to an approval card */
  answer: (itemId: string, approved: boolean) => void;
}

let itemCounter = 0;
const nextId = () => `item_${++itemCounter}`;
// Bumped by stop() and reset(), so a turn in flight knows it was abandoned
let generation = 0;
// Approval cards waiting for an answer, by item id
const waiting = new Map<string, (approved: boolean) => void>();

/** Declines every approval still waiting, e.g. when the turn is stopped */
function declineWaiting() {
  for (const resolve of waiting.values()) resolve(false);
  waiting.clear();
}

/** The tools available to the assistant. Later phases add more. */
export function assistantTools(): AssistantTool[] {
  return [...READ_TOOLS, ...EDIT_TOOLS, ...CODE_TOOLS];
}

function outcomeNotice(outcome: TurnOutcome): ChatItem | null {
  switch (outcome.kind) {
    case "refused": return { id: nextId(), kind: "notice", tone: "error", text: `Claude declined to answer this.${outcome.explanation ? ` ${outcome.explanation}` : ""}` };
    case "truncated": return { id: nextId(), kind: "notice", tone: "info", text: "The answer was cut off because it got too long. Ask it to continue." };
    case "step_limit": return { id: nextId(), kind: "notice", tone: "info", text: "Stopped after many steps without an answer. Try a narrower question." };
    case "cancelled": return { id: nextId(), kind: "notice", tone: "info", text: "Stopped." };
    default: return null;
  }
}

export const useAssistantStore = create<AssistantState>((set, get) => ({
  items: [],
  transcript: [],
  busy: false,
  model: storedModel(),
  key: null,
  draft: "",

  setDraft: (draft) => set({ draft }),

  setModel: (model) => {
    try { localStorage.setItem(MODEL_STORAGE, model); } catch { /* storage unavailable */ }
    set({ model });
  },

  refreshKey: async () => set({ key: await keyStatus() }),

  send: async (text) => {
    const trimmed = text.trim();
    if (!trimmed || get().busy) return;
    const gen = ++generation;
    const transcript = [...get().transcript];
    const push = (item: ChatItem) => { if (gen === generation) set({ items: [...get().items, item] }); };
    set({ busy: true, items: [...get().items, { id: nextId(), kind: "user", text: trimmed }] });

    // Tool calls of one step share a row; a new step starts a new row
    let toolRow: string | null = null;
    try {
      const turn = runTurn(trimmed, { model: get().model, tools: assistantTools(), transcript }, {
        cancelled: () => gen !== generation,
        approve: (request) => new Promise<boolean>((resolve) => {
          if (gen !== generation) { resolve(false); return; }
          const id = nextId();
          waiting.set(id, resolve);
          toolRow = null;
          set({ items: [...get().items, { id, kind: "approval", request, status: "pending" }] });
        }),
        onText: (t) => { toolRow = null; push({ id: nextId(), kind: "assistant", text: t }); },
        onTool: (call) => {
          if (gen !== generation) return;
          const items = get().items;
          const row = toolRow ? items.find((i) => i.id === toolRow) : null;
          if (row && row.kind === "tools") {
            const tools = row.tools.some((t) => t.id === call.id)
              ? row.tools.map((t) => (t.id === call.id ? { ...t, status: call.status, detail: call.detail } : t))
              : [...row.tools, { id: call.id, label: call.label, status: call.status, detail: call.detail }];
            set({ items: items.map((i) => (i.id === row.id ? { ...row, tools } : i)) });
          } else {
            toolRow = nextId();
            set({ items: [...items, { id: toolRow, kind: "tools", tools: [{ id: call.id, label: call.label, status: call.status, detail: call.detail }] }] });
          }
        },
      });
      // The message above told the assistant what the user did with its last changes; say it only once
      takeDecision();
      const outcome = await turn;
      if (gen !== generation) return;
      const notice = outcomeNotice(outcome);
      set({ transcript, ...(notice ? { items: [...get().items, notice] } : {}) });
    } catch (e) {
      if (gen !== generation) return;
      // The request failed: drop this turn from the transcript so the next one starts clean
      push({ id: nextId(), kind: "notice", tone: "error", text: e instanceof Error ? e.message : String(e) });
      if (/API key/.test(String(e))) void get().refreshKey();
    } finally {
      if (gen === generation) set({ busy: false });
    }
  },

  answer: (itemId, approved) => {
    const resolve = waiting.get(itemId);
    if (!resolve) return;
    waiting.delete(itemId);
    set({ items: get().items.map((i) => (i.id === itemId && i.kind === "approval" ? { ...i, status: approved ? "approved" : "declined" } : i)) });
    resolve(approved);
  },

  stop: () => {
    if (!get().busy) return;
    generation++;
    declineWaiting();
    set({ busy: false, items: [...get().items, { id: nextId(), kind: "notice", tone: "info", text: "Stopped." }] });
  },

  reset: () => {
    generation++;
    declineWaiting();
    set({ items: [], transcript: [], busy: false });
  },
}));

/** Opens the assistant with a question ready to send, e.g. from a shape error */
export function askAssistant(text: string) {
  useAssistantStore.getState().setDraft(text);
  window.dispatchEvent(new CustomEvent("tensorbloom:open-assistant"));
}
