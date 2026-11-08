import { describe, it, expect, beforeEach, vi } from "vitest";

vi.mock("../sidecar", () => ({ sidecarCall: () => Promise.reject(new Error("no sidecar in tests")) }));

const { useGraphStore } = await import("../../store/graph-store");
const { useLessonStore } = await import("../../store/lesson-store");
const { LESSON_TOOLS } = await import("./lesson-tools");
const { EDIT_TOOLS } = await import("./edit-tools");
const { CODE_TOOLS } = await import("./code-tools");
const { runToolCall } = await import("./agent");
const { applyChanges, discardChanges, hasPendingChanges } = await import("./change-set");
const { appState } = await import("./prompt");
import type { ApprovalRequest } from "./tool-types";

const TOOLS = [...LESSON_TOOLS, ...EDIT_TOOLS, ...CODE_TOOLS];
let asked: ApprovalRequest[] = [];
const call = async (name: string, input: unknown = {}, approve = true) => {
  const r = await runToolCall(TOOLS, { id: "t", name, input }, async (req) => { asked.push(req); return approve; });
  let out: unknown = r.result.content;
  try { out = JSON.parse(r.result.content as string); } catch { /* plain text */ }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return { ok: r.ok, out: out as any };
};
const fc2 = () => useGraphStore.getState().nodes.find((n) => n.data.label === "fc2")!;

beforeEach(() => {
  if (hasPendingChanges()) discardChanges();
  useLessonStore.getState().exit();
  useGraphStore.getState().clearGraph();
  asked = [];
});

describe("lesson tools", () => {
  it("lists the lessons when none is open", async () => {
    const { ok, out } = await call("get_lesson");
    expect(ok).toBe(true);
    expect(out.open).toBe(false);
    expect(out.lessons.map((l: { id: string }) => l.id)).toContain("fix-shape-error");
  });

  it("starts a lesson only once the user approves", async () => {
    const declined = await call("start_lesson", { lesson_id: "fix-shape-error" }, false);
    expect(declined.ok).toBe(false);
    expect(useLessonStore.getState().active).toBeNull();

    const { ok, out } = await call("start_lesson", { lesson_id: "fix-shape-error" });
    expect(ok).toBe(true);
    expect(asked[asked.length - 1]?.title).toContain("Fix a shape error");
    expect(out.step).toMatchObject({ index: 0, passed: false, hintsSeen: [] });
    expect(fc2().data.params.in_features).toBe(500);
  });

  it("suggests the closest lesson for an unknown id", async () => {
    const { ok, out } = await call("start_lesson", { lesson_id: "fix-shape" });
    expect(ok).toBe(false);
    expect(String(out)).toContain("fix-shape-error");
  });

  it("reveals hints in the panel one at a time, then says when none are left", async () => {
    await call("start_lesson", { lesson_id: "fix-shape-error" });
    const first = await call("show_hint");
    expect(first.out).toMatchObject({ number: 1, of: 3, isTheAnswer: false });
    expect(useLessonStore.getState().active?.hintsShown).toBe(1);
    await call("show_hint");
    const last = await call("show_hint");
    expect(last.out).toMatchObject({ number: 3, isTheAnswer: true });
    const none = await call("show_hint");
    expect(none.ok).toBe(false);
    const view = await call("get_lesson");
    expect(view.out.step.hintsSeen).toHaveLength(3);
    expect(view.out.step.hintsLeft).toHaveLength(0);
  });

  it("checks the step against a preview, and says it only counts once applied", async () => {
    await call("start_lesson", { lesson_id: "fix-shape-error" });
    await call("set_params", { changes: [{ layer_id: fc2().id, params: { in_features: 512 } }] });
    const previewed = await call("get_lesson");
    expect(previewed.out.step.passed).toBe(true);
    expect(previewed.out.pendingChanges).toContain("only counts once the learner applies");
    expect(previewed.out.currentStepExplain).toContain("in_features");

    applyChanges();
    const applied = await call("get_lesson");
    expect(applied.out.pendingChanges).toBeUndefined();
    const hint = await call("show_hint");
    expect(hint.ok).toBe(false);
  });

  it("won't replace the lesson's model while a lesson is open", async () => {
    await call("start_lesson", { lesson_id: "fix-shape-error" });
    const replaced = await call("replace_graph", { template: "Simple MLP" });
    expect(replaced.ok).toBe(false);
    expect(String(replaced.out)).toContain("A lesson is open");
    const imported = await call("import_code", { code: "import torch" });
    expect(imported.ok).toBe(false);
    expect(asked.some((a) => a.title.includes("import"))).toBe(false);
  });

  it("puts the lesson's step in the app state without its hints", async () => {
    expect(appState()).not.toContain("lesson:");
    await call("start_lesson", { lesson_id: "fix-shape-error" });
    const state = appState();
    expect(state).toContain('lesson: "Fix a shape error", step 1 of 3');
    expect(state).toContain("hints shown: 0 of 3");
    expect(state).not.toContain("Set fc2's in_features to 512");
  });
});
