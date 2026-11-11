import { describe, it, expect } from "vitest";
import { LESSONS } from "./catalog";
import { buildLessonStart } from "./start";
import { applyEdits, type LessonGraph } from "./edits";
import { evaluate, describeCheck, isConnected, type CheckContext } from "./checks";
import { inferShapesClient } from "../shape-inference";
import { useTrainingStore, type TrainingConfig } from "../../store/training-store";
import type { RunRecord } from "../../store/run-store";
import type { LessonCheck } from "./types";

const BASE: TrainingConfig = useTrainingStore.getState().config;

function fakeRun(graph: LessonGraph, config: TrainingConfig, n: number): RunRecord {
  // Each run does a bit better than the one before, as a learner following the lesson would see
  const acc = 0.975 + 0.002 * n;
  const loss = 0.5 - 0.05 * n;
  return {
    id: `run_${n}`, number: n + 1, modelName: "test", startedAt: new Date(2026, 0, 1, 0, n).toISOString(), durationMs: 1000,
    status: "finished", config, epochs: [], model: { params: 0, layers: graph.nodes.length, key: "" }, graph,
    results: {
      startedAt: "", completedAt: "", totalEpochs: config.epochs, durationMs: 1000,
      finalTrainLoss: loss - 0.1 + 0.01 * n, finalValLoss: loss, bestValLoss: loss, finalTrainAcc: acc + 0.01, finalValAcc: acc, bestValAcc: acc,
    },
  };
}

function usesRuns(check: LessonCheck): boolean {
  if (check.kind === "run") return true;
  if (check.kind === "all" || check.kind === "any") return check.checks.some(usesRuns);
  return false;
}

describe("lesson catalog", () => {
  it("has 6 to 8 lessons with unique ids", () => {
    expect(LESSONS.length).toBeGreaterThanOrEqual(6);
    expect(LESSONS.length).toBeLessThanOrEqual(8);
    expect(new Set(LESSONS.map((l) => l.id)).size).toBe(LESSONS.length);
    for (const l of LESSONS) expect(new Set(l.steps.map((s) => s.id)).size).toBe(l.steps.length);
  });

  it("is plain data that survives JSON", () => {
    expect(JSON.parse(JSON.stringify(LESSONS))).toEqual(LESSONS);
  });

  it("describes every check in words", () => {
    for (const l of LESSONS) for (const s of l.steps) {
      expect(describeCheck(s.check).length).toBeGreaterThan(5);
      expect(s.hints.length).toBeGreaterThan(0);
    }
  });
});

describe.each(LESSONS.map((l) => [l.id, l] as const))("lesson %s", (_id, lesson) => {
  it("starts from a connected graph whose only shape errors are the lesson's", () => {
    const { graph } = buildLessonStart(lesson, BASE);
    expect(isConnected(graph)).toBe(true);
    const errors = inferShapesClient(graph.nodes, graph.edges).errors;
    if (lesson.id === "fix-shape-error") expect(Object.keys(errors).length).toBeGreaterThan(0);
    else expect(errors).toEqual({});
  });

  it("can be completed step by step, and no step passes before its work is done", () => {
    let { graph, config } = buildLessonStart(lesson, BASE);
    const runs: RunRecord[] = [];
    const confirmed: string[] = [];
    const ctx = (stepId: string): CheckContext => ({ graph, config, runs, confirmed, stepId });

    for (const step of lesson.steps) {
      const before = evaluate(step.check, ctx(step.id)).passed;
      if (step.check.kind === "manual") {
        expect(before).toBe(false);
        confirmed.push(step.id);
      } else if (!usesRuns(step.check)) {
        expect(before, `${step.id} passes before any change`).toBe(false);
        expect(step.solution, `${step.id} needs a solution`).toBeDefined();
      }
      if (step.solution) ({ graph, config } = applyEdits(graph, config, step.solution));
      if (usesRuns(step.check)) runs.push(fakeRun(graph, config, runs.length));
      expect(evaluate(step.check, ctx(step.id)).passed, `${step.id} passes once done`).toBe(true);
    }
  });
});

describe("checks", () => {
  const lesson = LESSONS.find((l) => l.id === "fix-shape-error")!;
  const { graph, config } = buildLessonStart(lesson, BASE);

  it("doesn't count a fix that deletes the broken layer", () => {
    const fc2 = graph.nodes.find((n) => n.data.label === "fc2")!;
    const removed = applyEdits(graph, config, [{ op: "remove_layer", target: { label: "fc2" } }]).graph;
    expect(removed.nodes.find((n) => n.id === fc2.id)).toBeUndefined();
    // Deleting fc2 leaves a 512-wide input to fc3, which expects 256: still an error
    expect(evaluate(lesson.steps[0].check, { graph: removed, config, runs: [] }).passed).toBe(false);
    // A dangling layer isn't connected
    const dangling = { nodes: graph.nodes, edges: graph.edges.filter((e) => e.target !== fc2.id) };
    expect(isConnected(dangling)).toBe(false);
  });

  it("compares a run with the lesson's earlier runs", () => {
    const check: LessonCheck = { kind: "run", where: [{ metric: "bestValAcc", op: "gt", than: "first" }] };
    const a = fakeRun(graph, config, 0);
    const worse = { ...fakeRun(graph, config, 1), results: { ...a.results!, bestValAcc: 0.5 } };
    expect(evaluate(check, { graph, config, runs: [a] }).passed).toBe(false);
    expect(evaluate(check, { graph, config, runs: [a, worse] }).passed).toBe(false);
    expect(evaluate(check, { graph, config, runs: [a, worse, fakeRun(graph, config, 2)] }).passed).toBe(true);
  });

  it("ignores failed runs", () => {
    const failed: RunRecord = { ...fakeRun(graph, config, 0), status: "failed", results: undefined };
    expect(evaluate({ kind: "run" }, { graph, config, runs: [failed] }).passed).toBe(false);
  });

  it("lists labeled parts for the checklist", () => {
    const step = lesson.steps[1];
    const result = evaluate(step.check, { graph, config, runs: [] });
    expect(result.parts.map((p) => p.label)).toEqual(["fc1 has 1024 outputs", "No shape errors", "Every layer is connected"]);
    expect(result.parts[0].passed).toBe(false);
  });
});

describe("lessonRuns", () => {
  it("counts only runs started since the lesson began", async () => {
    const { lessonRuns } = await import("../../store/lesson-store");
    const lesson = LESSONS[0];
    const { graph, config } = buildLessonStart(lesson, BASE);
    const runs = [0, 1, 2].map((n) => fakeRun(graph, config, n));
    expect(lessonRuns(runs, runs[1].startedAt).map((r) => r.id)).toEqual(["run_1", "run_2"]);
  });
});
