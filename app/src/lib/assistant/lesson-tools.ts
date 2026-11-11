/**
 * Tools for tutoring the guided lessons: what the learner is working on, the lesson's own hints
 * (revealed in the lesson panel so the two stay in step), and starting a lesson.
 */

import { useLessonStore, lessonContext, checkCurrentStep } from "../../store/lesson-store";
import { useTrainingStore } from "../../store/training-store";
import { LESSONS, findLesson } from "../lessons/catalog";
import { hasPendingChanges } from "./change-set";
import { ToolError, closestNames, type AssistantTool } from "./tool-types";

const noInput = { type: "object" as const, properties: {}, additionalProperties: false };

/** The lesson list, for when no lesson is open or the learner asks what else there is */
function lessonList() {
  const completed = useLessonStore.getState().completed;
  return LESSONS.map((l) => ({
    id: l.id,
    title: l.title,
    summary: l.summary,
    level: l.level,
    minutes: l.minutes,
    concepts: l.concepts,
    finished: completed.includes(l.id),
  }));
}

/** Everything the tutor needs about the lesson in progress, or null when none is open */
export function tutorView() {
  const ctx = lessonContext();
  const active = useLessonStore.getState().active;
  const lesson = active && findLesson(active.lessonId);
  if (!ctx || !active || !lesson) return null;
  const result = checkCurrentStep(active);
  return {
    ...ctx,
    checklist: result?.parts ?? [],
    // What earlier steps taught, so the tutor can build on it. Later steps stay hidden.
    earlierSteps: lesson.steps.slice(0, active.stepIndex).map((s) => ({ instruction: s.instruction, explain: s.explain })),
    currentStepExplain: ctx.step.passed ? lesson.steps[active.stepIndex].explain : undefined,
    pendingChanges: hasPendingChanges()
      ? "Your changes are still a preview. The step is checked against the canvas as it is, and only counts once the learner applies them."
      : undefined,
  };
}

const getLesson: AssistantTool = {
  kind: "read",
  definition: {
    name: "get_lesson",
    description:
      "The guided lesson the learner has open: its goal and concepts, the current step's instruction, what the step checks (this can give the answer away, so don't quote it), whether it passes, its checklist, the hints already shown and the ones left, and what earlier steps explained. With no lesson open, lists the lessons there are.",
    input_schema: noInput,
  },
  describe: () => "Reading the lesson",
  run: async () => tutorView() ?? { open: false, lessons: lessonList() },
};

const showHint: AssistantTool = {
  kind: "read",
  definition: {
    name: "show_hint",
    description:
      "Shows the learner the current step's next hint in the lesson panel and returns it. The hints go from a nudge to the answer, so use one only when your own nudges haven't been enough or the learner asks for a hint. Build on the hint in your reply rather than repeating it word for word.",
    input_schema: noInput,
  },
  describe: () => "Showing a hint",
  run: async () => {
    const view = tutorView();
    if (!view) throw new ToolError("No lesson is open.");
    if (view.finished) throw new ToolError("The lesson is finished.");
    if (view.step.passed) throw new ToolError("This step already passes; the learner can go on to the next one.");
    const next = view.step.hintsLeft[0];
    if (!next) throw new ToolError(`All ${view.step.hintsSeen.length} hints are showing already. Help the learner from there yourself.`);
    useLessonStore.getState().showHint();
    const shown = view.step.hintsSeen.length + 1;
    return { hint: next, number: shown, of: shown + view.step.hintsLeft.length - 1, isTheAnswer: view.step.hintsLeft.length === 1 };
  },
};

const startLesson: AssistantTool<{ lesson_id: string }> = {
  kind: "confirm",
  definition: {
    name: "start_lesson",
    description:
      "Opens a guided lesson. It replaces the model on the canvas with the lesson's starting model and clears the run history, so the user approves it first, and is asked to save unsaved work. Use it when the user wants to learn something a lesson covers and agrees to start it.",
    input_schema: {
      type: "object",
      properties: { lesson_id: { type: "string", description: "The lesson's id, from get_lesson" } },
      required: ["lesson_id"],
      additionalProperties: false,
    },
  },
  confirm: ({ lesson_id }) => {
    const lesson = findLesson(lesson_id);
    if (!lesson) throw new ToolError(`There's no lesson "${lesson_id}". Closest: ${closestNames(lesson_id, LESSONS.map((l) => l.id)).join(", ")}.`);
    if (hasPendingChanges()) throw new ToolError("There are pending changes. Ask the user to apply or discard them first.");
    const status = useTrainingStore.getState().status;
    if (status === "running" || status === "starting") throw new ToolError("Training is running. It has to stop before a lesson can start.");
    return {
      title: `Start the lesson "${lesson.title}"?`,
      summary: `${lesson.summary} About ${lesson.minutes} minutes. The lesson's starting model replaces what's on the canvas, and the run history is cleared.`,
      approveLabel: "Start lesson",
    };
  },
  describe: ({ lesson_id }) => `Starting the lesson ${findLesson(lesson_id)?.title ?? lesson_id}`,
  run: async ({ lesson_id }) => {
    const started = await useLessonStore.getState().start(lesson_id);
    if (!started) throw new ToolError("The lesson didn't start: the user kept their current project.");
    return tutorView();
  },
};

export const LESSON_TOOLS: AssistantTool[] = [getLesson, showHint, startLesson] as AssistantTool[];

/** Refuses a tool that would replace the lesson's model while a lesson is open */
export function refuseDuringLesson(what: string) {
  const active = useLessonStore.getState().active;
  if (active && !active.finished) {
    throw new ToolError(`A lesson is open, and ${what} would replace the model it checks. Help with the lesson's model instead, or ask the learner to leave the lesson first (the × on the lesson panel).`);
  }
}
