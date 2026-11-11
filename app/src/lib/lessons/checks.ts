/** Evaluates lesson checks against the canvas, the training settings and the lesson's runs. Pure, so tests and the tutor can use it. */

import type { RunRecord } from "../../store/run-store";
import type { TrainingConfig } from "../../store/training-store";
import type { ShapeError } from "../shape-inference";
import { inferShapesClient } from "../shape-inference";
import { isDatasetNode } from "../dataset-metadata";
import { getTrainingField, matches, selectorText, type LessonGraph } from "./edits";
import type { CompareOp, LessonCheck, RunCondition, RunMetric } from "./types";

export interface CheckContext {
  graph: LessonGraph;
  /** The canvas's shape errors; worked out from the graph when left out */
  shapeErrors?: Record<string, ShapeError>;
  config: TrainingConfig;
  /** Runs since the lesson started, oldest first */
  runs: RunRecord[];
  /** Ids of manual steps the learner has marked done */
  confirmed?: string[];
  /** The step being checked, for manual checks */
  stepId?: string;
}

export interface CheckResult {
  passed: boolean;
  /** The labeled parts of the check, for the goal's checklist */
  parts: { label: string; passed: boolean }[];
}

export function compare(a: unknown, op: CompareOp, b: unknown): boolean {
  if (op === "eq") return a === b;
  if (op === "ne") return a !== b;
  if (typeof a !== "number" || typeof b !== "number" || Number.isNaN(a)) return false;
  if (op === "lt") return a < b;
  if (op === "lte") return a <= b;
  if (op === "gt") return a > b;
  return a >= b;
}

export function runMetric(run: RunRecord, metric: RunMetric): number | undefined {
  const r = run.results;
  if (!r) return undefined;
  if (metric === "trainValAccGap") return r.finalTrainAcc != null && r.finalValAcc != null ? r.finalTrainAcc - r.finalValAcc : undefined;
  if (metric === "trainValLossGap") return r.finalValLoss != null ? r.finalValLoss - r.finalTrainLoss : undefined;
  return r[metric];
}

/** Runs that finished with results: a failed run, or one stopped before its first epoch, has nothing to judge */
export function usableRuns(runs: RunRecord[]): RunRecord[] {
  return runs.filter((r) => r.status !== "failed" && r.results);
}

function shapeErrorsOf(ctx: CheckContext): Record<string, ShapeError> {
  return ctx.shapeErrors ?? inferShapesClient(ctx.graph.nodes, ctx.graph.edges).errors;
}

/** Every layer has an input (Data nodes aside) and feeds something (losses aside), and a loss is reachable from data */
export function isConnected(graph: LessonGraph): boolean {
  const isData = (mt: string) => mt === "Data" || isDatasetNode(mt);
  const isLoss = (cat: string) => cat === "loss";
  const hasIn = new Set(graph.edges.map((e) => e.target));
  const hasOut = new Set(graph.edges.map((e) => e.source));
  for (const n of graph.nodes) {
    if (!isData(n.data.moduleType) && !hasIn.has(n.id)) return false;
    if (!isLoss(n.data.category) && !hasOut.has(n.id)) return false;
  }
  const next = new Map<string, string[]>();
  for (const e of graph.edges) next.set(e.source, [...(next.get(e.source) ?? []), e.target]);
  const seen = new Set<string>();
  const queue = graph.nodes.filter((n) => isData(n.data.moduleType)).map((n) => n.id);
  while (queue.length) {
    const id = queue.pop()!;
    if (seen.has(id)) continue;
    seen.add(id);
    queue.push(...(next.get(id) ?? []));
  }
  return graph.nodes.some((n) => isLoss(n.data.category) && seen.has(n.id));
}

function runMeets(run: RunRecord, cond: RunCondition, earlier: RunRecord[], ctx: CheckContext): boolean {
  if ("any" in cond) return cond.any.some((c) => runMeets(run, c, earlier, ctx));
  if ("graph" in cond) {
    return evaluate(cond.graph, { ...ctx, graph: run.graph, shapeErrors: undefined, config: run.config }).passed;
  }
  if ("training" in cond) return compare(getTrainingField(run.config, cond.training), cond.op, cond.value);
  const value = runMetric(run, cond.metric);
  if (value == null) return false;
  if ("value" in cond) return compare(value, cond.op, cond.value);
  const others = cond.than === "first" ? earlier.slice(0, 1) : cond.than === "previous" ? earlier.slice(-1) : earlier;
  if (others.length === 0) return false;
  return others.every((o) => {
    const ov = runMetric(o, cond.metric);
    return ov != null && compare(value, cond.op, ov);
  });
}

function evaluateOne(check: LessonCheck, ctx: CheckContext): CheckResult {
  const nodes = ctx.graph.nodes;
  switch (check.kind) {
    case "no_shape_errors":
      return leaf(Object.keys(shapeErrorsOf(ctx)).length === 0);
    case "connected":
      return leaf(isConnected(ctx.graph));
    case "layer_count": {
      const n = nodes.filter((x) => matches(x, check.select)).length;
      return leaf((check.min == null || n >= check.min) && (check.max == null || n <= check.max));
    }
    case "param": {
      const hits = nodes.filter((x) => matches(x, check.select));
      return leaf(hits.length > 0 && hits.every((x) => compare(x.data.params[check.param], check.op, check.value)));
    }
    case "followed_by": {
      const hits = nodes.filter((x) => matches(x, check.select));
      const byId = new Map(nodes.map((x) => [x.id, x]));
      return leaf(hits.length > 0 && hits.every((x) =>
        ctx.graph.edges.some((e) => e.source === x.id && byId.get(e.target) && matches(byId.get(e.target)!, check.next))));
    }
    case "training":
      return leaf(compare(getTrainingField(ctx.config, check.field), check.op, check.value));
    case "run": {
      const runs = usableRuns(ctx.runs);
      return leaf(runs.some((run, i) => (check.where ?? []).every((c) => runMeets(run, c, runs.slice(0, i), ctx))));
    }
    case "manual":
      return leaf(!!ctx.stepId && (ctx.confirmed ?? []).includes(ctx.stepId));
    case "all":
    case "any": {
      const results = check.checks.map((c) => evaluate(c, ctx));
      const passed = check.kind === "all" ? results.every((r) => r.passed) : results.some((r) => r.passed);
      return { passed, parts: results.flatMap((r) => r.parts) };
    }
  }
}

function leaf(passed: boolean): CheckResult {
  return { passed, parts: [] };
}

export function evaluate(check: LessonCheck, ctx: CheckContext): CheckResult {
  const result = evaluateOne(check, ctx);
  return check.label ? { passed: result.passed, parts: [{ label: check.label, passed: result.passed }, ...result.parts] } : result;
}

const OP_TEXT: Record<CompareOp, string> = { eq: "is", ne: "is not", lt: "is below", lte: "is at most", gt: "is above", gte: "is at least" };

/** The check in words, for the tutor and for the docs. It can give the answer away, so the panel shows labels instead. */
export function describeCheck(check: LessonCheck): string {
  switch (check.kind) {
    case "no_shape_errors": return "no layer has a shape error";
    case "connected": return "every layer is connected, from the Data node to the loss";
    case "layer_count": {
      const what = selectorText(check.select);
      if (check.min != null && check.max != null) return check.min === check.max ? `exactly ${check.min} × ${what}` : `between ${check.min} and ${check.max} × ${what}`;
      if (check.min != null) return `at least ${check.min} × ${what}`;
      if (check.max != null) return check.max === 0 ? `no ${what} layers` : `at most ${check.max} × ${what}`;
      return `any number of ${what}`;
    }
    case "param": return `${check.param} on ${selectorText(check.select)} ${OP_TEXT[check.op]} ${JSON.stringify(check.value)}`;
    case "followed_by": return `every ${selectorText(check.select)} feeds straight into a ${selectorText(check.next)}`;
    case "training": return `the training setting ${check.field} ${OP_TEXT[check.op]} ${JSON.stringify(check.value)}`;
    case "run": {
      const conds = (check.where ?? []).map(describeCondition);
      return conds.length ? `a run since the lesson started where ${conds.join(", and ")}` : "a training run has finished since the lesson started";
    }
    case "manual": return "the learner marks it done";
    case "all": return check.checks.map(describeCheck).join("; and ");
    case "any": return `one of: ${check.checks.map(describeCheck).join("; or ")}`;
  }
}

function describeCondition(c: RunCondition): string {
  if ("any" in c) return `(${c.any.map(describeCondition).join(" or ")})`;
  if ("graph" in c) return `its graph has ${describeCheck(c.graph)}`;
  if ("training" in c) return `${c.training} ${OP_TEXT[c.op]} ${JSON.stringify(c.value)}`;
  if ("value" in c) return `${c.metric} ${OP_TEXT[c.op]} ${c.value}`;
  const whom = c.than === "first" ? "the lesson's first run" : c.than === "previous" ? "the run before it" : "every earlier run in the lesson";
  return `${c.metric} ${c.op === "lt" ? "is lower than" : "is higher than"} ${whom}'s`;
}
