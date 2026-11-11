import { create } from "zustand";
import { useGraphStore } from "./graph-store";
import { useTrainingStore } from "./training-store";
import { useRunStore, type RunRecord } from "./run-store";
import { useDocumentStore } from "./document-store";
import { confirmDiscardChanges, replaceGraph } from "../lib/document-actions";
import { LESSONS, findLesson } from "../lib/lessons/catalog";
import { buildLessonStart } from "../lib/lessons/start";
import { evaluate, describeCheck, type CheckResult } from "../lib/lessons/checks";
import type { Lesson, LessonStep } from "../lib/lessons/types";
import { toast } from "../components/Toast";

const COMPLETED_KEY = "tensorbloom.lessons.completed";

function readCompleted(): string[] {
  try {
    const raw = localStorage.getItem(COMPLETED_KEY);
    const ids = raw ? JSON.parse(raw) : [];
    return Array.isArray(ids) ? ids.filter((id) => typeof id === "string") : [];
  } catch {
    return [];
  }
}

function writeCompleted(ids: string[]) {
  try { localStorage.setItem(COMPLETED_KEY, JSON.stringify(ids)); } catch { /* storage unavailable: progress lasts for this session */ }
}

export interface ActiveLesson {
  lessonId: string;
  /** When the lesson started. Only runs started after this count toward its checks. */
  startedAt: string;
  stepIndex: number;
  /** Steps that have passed. A step stays done even if later edits would make its check fail again. */
  done: string[];
  /** Manual steps the learner marked done */
  confirmed: string[];
  /** How many hints of the current step are showing */
  hintsShown: number;
  /** Every step is done */
  finished: boolean;
}

interface LessonState {
  active: ActiveLesson | null;
  /** Ids of lessons finished on this machine */
  completed: string[];
  collapsed: boolean;
  start: (lessonId: string) => Promise<boolean>;
  exit: () => void;
  next: () => void;
  confirmStep: () => void;
  showHint: () => void;
  markDone: (stepId: string) => void;
  setCollapsed: (collapsed: boolean) => void;
}

export const useLessonStore = create<LessonState>((set, get) => ({
  active: null,
  completed: readCompleted(),
  collapsed: false,

  start: async (lessonId) => {
    const lesson = findLesson(lessonId);
    if (!lesson) return false;
    const status = useTrainingStore.getState().status;
    if (status === "running" || status === "starting") {
      toast.info("Stop the current training run before starting a lesson.");
      return false;
    }
    if (!(await confirmDiscardChanges("starting a lesson"))) return false;
    const { graph, config, projectName } = buildLessonStart(lesson, useTrainingStore.getState().config);
    replaceGraph(graph.nodes, graph.edges, projectName);
    useTrainingStore.getState().reset();
    useTrainingStore.getState().applyTemplateDefaults(lesson.start.template, config);
    // A lesson is a fresh project: earlier runs would muddle the comparisons it asks for
    useRunStore.getState().setRuns([]);
    useDocumentStore.getState().markSaved(null);
    set({
      active: { lessonId, startedAt: new Date().toISOString(), stepIndex: 0, done: [], confirmed: [], hintsShown: 0, finished: false },
      collapsed: false,
    });
    return true;
  },

  exit: () => set({ active: null }),

  next: () => {
    const a = get().active;
    const lesson = a && findLesson(a.lessonId);
    if (!a || !lesson || !a.done.includes(lesson.steps[a.stepIndex]?.id)) return;
    if (a.stepIndex + 1 >= lesson.steps.length) {
      const completed = get().completed.includes(lesson.id) ? get().completed : [...get().completed, lesson.id];
      writeCompleted(completed);
      set({ active: { ...a, finished: true }, completed });
      return;
    }
    set({ active: { ...a, stepIndex: a.stepIndex + 1, hintsShown: 0 } });
  },

  confirmStep: () => {
    const a = get().active;
    const step = a && currentStep(a);
    if (!a || !step || a.confirmed.includes(step.id)) return;
    set({ active: { ...a, confirmed: [...a.confirmed, step.id] } });
  },

  showHint: () => {
    const a = get().active;
    const step = a && currentStep(a);
    if (!a || !step) return;
    set({ active: { ...a, hintsShown: Math.min(step.hints.length, a.hintsShown + 1) } });
  },

  markDone: (stepId) => {
    const a = get().active;
    if (!a || a.done.includes(stepId)) return;
    set({ active: { ...a, done: [...a.done, stepId] } });
  },

  setCollapsed: (collapsed) => set({ collapsed }),
}));

function currentStep(a: ActiveLesson): LessonStep | undefined {
  return findLesson(a.lessonId)?.steps[a.stepIndex];
}

/** Runs that count toward the lesson: those started since it began */
export function lessonRuns(runs: RunRecord[], startedAt: string): RunRecord[] {
  const since = Date.parse(startedAt);
  return runs.filter((r) => Date.parse(r.startedAt) >= since);
}

/** Checks the current step against the app's state right now */
export function checkCurrentStep(a: ActiveLesson): CheckResult | null {
  const step = currentStep(a);
  if (!step) return null;
  const g = useGraphStore.getState();
  return evaluate(step.check, {
    graph: { nodes: g.nodes, edges: g.edges },
    config: useTrainingStore.getState().config,
    runs: lessonRuns(useRunStore.getState().runs, a.startedAt),
    confirmed: a.confirmed,
    stepId: step.id,
  });
}

/**
 * The lesson in progress, as plain data for the assistant when it tutors: the lesson, where the
 * learner is, what the current step checks, and which hints they've already seen.
 */
export function lessonContext(): null | {
  lesson: Pick<Lesson, "id" | "title" | "goal" | "concepts">;
  step: { index: number; count: number; id: string; instruction: string; checks: string; passed: boolean; hintsSeen: string[]; hintsLeft: string[] };
  finished: boolean;
} {
  const a = useLessonStore.getState().active;
  const lesson = a && findLesson(a.lessonId);
  const step = a && currentStep(a);
  if (!a || !lesson || !step) return null;
  return {
    lesson: { id: lesson.id, title: lesson.title, goal: lesson.goal, concepts: lesson.concepts },
    step: {
      index: a.stepIndex,
      count: lesson.steps.length,
      id: step.id,
      instruction: step.instruction,
      checks: describeCheck(step.check),
      passed: a.done.includes(step.id) || !!checkCurrentStep(a)?.passed,
      hintsSeen: step.hints.slice(0, a.hintsShown),
      hintsLeft: step.hints.slice(a.hintsShown),
    },
    finished: a.finished,
  };
}

export { LESSONS };
