/**
 * The lesson format. A lesson is plain data (it survives JSON.stringify unchanged), so the same
 * definition drives the lesson panel, the tests, and the assistant when it tutors a lesson.
 *
 * A lesson is a starting graph, a goal, and a list of steps. Each step has an instruction, a check
 * that decides when it's done, hints from gentle to the answer, and an explanation shown once it passes.
 * See docs/guides/lessons.md for the format with examples.
 */

/** Picks layers by their label on the canvas, their module type, or both */
export interface LayerSelector {
  label?: string;
  moduleType?: string;
}

export type CompareOp = "eq" | "ne" | "lt" | "lte" | "gt" | "gte";

/** Training settings a lesson can set or check, as dotted paths into the Training panel's settings */
export type TrainingField =
  | "optimizer.type"
  | "optimizer.lr"
  | "optimizer.weightDecay"
  | "optimizer.momentum"
  | "epochs"
  | "batchSize"
  | "loss"
  | "dataset.type";

/** A change to the graph or the training settings. Lessons use these to build the starting graph, and to describe a solution. */
export type LessonEdit =
  /** Sets a parameter on every matching layer */
  | { op: "set_param"; target: LayerSelector; param: string; value: number | string | boolean }
  /** Removes every matching layer, connecting its input straight to what it fed */
  | { op: "remove_layer"; target: LayerSelector }
  /** Swaps the module of every matching layer, keeping its connections. Parameters reset to the new module's defaults, then `params` apply. */
  | { op: "replace_layer"; target: LayerSelector; module: string; label?: string; params?: Record<string, unknown> }
  /** Adds a layer right after the one layer that matches, taking over everything it fed */
  | { op: "insert_after"; target: LayerSelector; module: string; label?: string; params?: Record<string, unknown> }
  | { op: "set_training"; field: TrainingField; value: number | string };

/** The numbers a finished run is judged by. Accuracies are fractions (0.97 is 97%). */
export type RunMetric =
  | "bestValAcc"
  | "finalValAcc"
  | "finalTrainAcc"
  | "bestValLoss"
  | "finalValLoss"
  | "finalTrainLoss"
  /** finalTrainAcc − finalValAcc: how much better it does on data it trained on */
  | "trainValAccGap"
  /** finalValLoss − finalTrainLoss */
  | "trainValLossGap";

/**
 * Something a single run must satisfy. Every condition in a `run` check must hold for the same run.
 * Relative conditions compare with the lesson's other runs: `first` is the first run since the lesson
 * started, `previous` the one just before, `every_previous` all of the earlier ones.
 */
export type RunCondition =
  | { metric: RunMetric; op: CompareOp; value: number }
  | { metric: RunMetric; op: "lt" | "gt"; than: "first" | "previous" | "every_previous" }
  | { training: TrainingField; op: CompareOp; value: number | string }
  /** The graph the run trained passes this check */
  | { graph: LessonCheck }
  | { any: RunCondition[] };

/** Decides when a step is done. Graph checks look at the canvas as it is now. */
export type LessonCheck = (
  /** No layer has a shape error */
  | { kind: "no_shape_errors" }
  /** Data flows from a Data node through every layer to the loss: nothing is left dangling */
  | { kind: "connected" }
  /** How many layers match */
  | { kind: "layer_count"; select: LayerSelector; min?: number; max?: number }
  /** A parameter on every matching layer (and at least one must match) */
  | { kind: "param"; select: LayerSelector; param: string; op: CompareOp; value: number | string | boolean }
  /** Every layer matching `select` feeds straight into one matching `next` */
  | { kind: "followed_by"; select: LayerSelector; next: LayerSelector }
  /** A setting in the Training panel, as it is now */
  | { kind: "training"; field: TrainingField; op: CompareOp; value: number | string }
  /** Some run since the lesson started meets every condition */
  | { kind: "run"; where?: RunCondition[] }
  /** The learner says they've done it, for steps about looking at something */
  | { kind: "manual"; button?: string }
  | { kind: "all"; checks: LessonCheck[] }
  | { kind: "any"; checks: LessonCheck[] }
) & {
  /** Shown to the learner as a line of the goal's checklist. Leave it out to keep a part hidden, e.g. when it would give the answer away. */
  label?: string;
};

export interface LessonStep {
  id: string;
  /** What to do, in a sentence or two */
  instruction: string;
  check: LessonCheck;
  /** From a nudge to the answer. The panel shows one at a time; the tutor works through them in order. */
  hints: string[];
  /** Shown when the step passes: what just happened and why */
  explain: string;
  /** Edits that complete the step, for steps about the graph or settings. The tests apply them to prove the step can pass. */
  solution?: LessonEdit[];
}

export interface Lesson {
  id: string;
  title: string;
  /** One sentence for the lesson list */
  summary: string;
  /** What the learner is trying to do, shown at the top of the lesson panel */
  goal: string;
  level: "beginner" | "intermediate";
  /** Rough time including training on a laptop CPU */
  minutes: number;
  /** Ideas the lesson teaches, for the list and for the tutor */
  concepts: string[];
  start: {
    /** Name of a template in templates.ts */
    template: string;
    /** Replaces the template's dataset, on the Data node and in the training settings */
    dataset?: string;
    /** Applied after the template's training defaults */
    training?: Partial<Record<TrainingField, number | string>>;
    /** Applied in order to the template's graph */
    edits?: LessonEdit[];
    /** The project's name while the lesson is open */
    projectName?: string;
  };
  steps: LessonStep[];
}
