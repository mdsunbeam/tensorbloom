/** Builds a lesson's starting graph and training settings. Pure: the lesson store loads the result. */

import { TEMPLATES, templateGraph, type Template } from "../templates";
import { autoLayout } from "../auto-layout";
import type { TrainingConfig } from "../../store/training-store";
import { applyEdits, type LessonGraph } from "./edits";
import type { Lesson, LessonEdit, TrainingField } from "./types";

/** A template's training defaults laid over the current settings, as opening the template does */
export function templateTrainingConfig(tpl: Template, base: TrainingConfig): TrainingConfig {
  const d = tpl.trainingDefaults;
  if (!d) return base;
  return {
    ...base,
    ...(d.optimizer && { optimizer: { ...base.optimizer, ...d.optimizer } }),
    ...(d.loss && { loss: d.loss }),
    ...(d.epochs != null && { epochs: d.epochs }),
    ...(d.batchSize != null && { batchSize: d.batchSize }),
    ...(d.dataset && { dataset: d.dataset }),
  };
}

export function lessonTemplate(lesson: Lesson): Template {
  const tpl = TEMPLATES.find((t) => t.name === lesson.start.template);
  if (!tpl) throw new Error(`Lesson "${lesson.id}" starts from template "${lesson.start.template}", which doesn't exist`);
  const dataset = lesson.start.dataset;
  if (!dataset) return tpl;
  const defaults = tpl.trainingDefaults ?? {};
  return { ...tpl, trainingDefaults: { ...defaults, dataset: { root: "./data", ...defaults.dataset, type: dataset } } };
}

export function buildLessonStart(lesson: Lesson, base: TrainingConfig): { graph: LessonGraph; config: TrainingConfig; projectName: string } {
  const tpl = lessonTemplate(lesson);
  const raw = templateGraph(tpl);
  const trainingEdits: LessonEdit[] = Object.entries(lesson.start.training ?? {}).map(([field, value]) => ({
    op: "set_training", field: field as TrainingField, value: value as number | string,
  }));
  const { graph, config } = applyEdits(raw, templateTrainingConfig(tpl, base), [...(lesson.start.edits ?? []), ...trainingEdits]);
  const laidOut = lesson.start.edits?.length ? { nodes: autoLayout(graph.nodes, graph.edges), edges: graph.edges } : graph;
  return { graph: laidOut, config, projectName: lesson.start.projectName ?? `Lesson: ${lesson.title}` };
}
