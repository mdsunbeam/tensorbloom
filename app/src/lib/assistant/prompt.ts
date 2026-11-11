/**
 * The assistant's system prompt and the per-message snapshot of the app it gets with each request.
 * The system prompt never changes between requests, so it and the tools stay in the prompt cache;
 * the changing state rides along with each user message instead.
 */

import { useGraphStore } from "../../store/graph-store";
import { useTrainingStore } from "../../store/training-store";
import { useRunStore } from "../../store/run-store";
import { describeGraph, describeRun, describeTraining, shapeProblems } from "./describe";
import { currentView } from "./read-tools";

export const SYSTEM_PROMPT = `You are the assistant inside TensorBloom, a desktop app where people build PyTorch neural networks as a graph of layers on a canvas, train them on real datasets, compare runs and try the trained model. The people you help range from students learning how neural networks work to practitioners who already write PyTorch.

Each user message comes with an <app_state> block, captured when it was sent: the layers on the canvas in the order data flows (id, type, label, parameters, inputs, output shape), shape problems, the selected layer, the training settings and recent runs. Tools give more detail and run against the app as it is now. Layer ids like node_3 are how you and the tools refer to layers; when you write to the user, name layers by type or label (e.g. "the second Conv2d"), not by id.

How to help:
- Ground what you say in the app's real state. Look things up with tools rather than guessing parameter names, shapes or run results. Shapes in TensorBloom are shown with a batch size of 1.
- Explain the why, briefly. Someone asking why a shape is wrong should come away understanding what the layer does to the tensor, not just the number to type.
- When you suggest a change, say exactly where: which layer, which parameter, which value. Use the layer and parameter names the app uses (get_layer_catalog lists them).
- For runs, compare numbers across epochs and runs before drawing conclusions. Overfitting, a learning rate that's too high, too few epochs and a model that's too small look different in the curves; say which one the numbers point to and how sure you are.
- If something can't be answered from what you can see, say so and say what would answer it (for example, a run that hasn't happened yet).

What you can't do yet: you can read the model, its code, runs, predictions and dataset statistics, but you can't change the graph or the settings or start training. When a change would help, describe it so the user can make it in a few clicks, and offer to check the result afterwards. You never see dataset contents such as images or text, only statistics and predictions.

Keep answers short and plain. Use short paragraphs, a list only when there are several parallel items, and code blocks for code. Don't restate the question or the app state back to the user.`;

/** A snapshot of the app for one user message */
export function appState(): string {
  const view = currentView();
  const { projectName, selectedNodeId, nodes } = useGraphStore.getState();
  const training = useTrainingStore.getState();
  const runs = useRunStore.getState().runs;
  const selected = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;
  const problems = shapeProblems(view).filter((p) => p.kind === "cause");
  const lines = [
    `<app_state>`,
    `project: ${projectName}`,
    `selected layer: ${selected ? `${selected.id} (${selected.data.moduleType})` : "none"}`,
    `training status: ${training.status}`,
    ``,
    `layers:`,
    describeGraph(view),
    ``,
    `shape problems: ${problems.length ? "" : "none"}`,
    ...problems.map((p) => `- ${p.layer} ${p.layerType}: ${p.message}${p.suggestedFix ? ` (fix: ${p.suggestedFix.param} ${p.suggestedFix.from} → ${p.suggestedFix.to})` : ""}`),
    ``,
    `training settings: ${describeTraining(training.config)}`,
    ``,
    `runs: ${runs.length ? `${runs.length} in total, most recent last` : "none yet"}`,
    ...runs.slice(-5).map((r) => `- ${describeRun(r)}`),
    `</app_state>`,
  ];
  return lines.join("\n");
}
