/**
 * Training execution — bridge between frontend stores and Python sidecar.
 */

import { sidecarCall, setTrainProgressCallback, setTrainLogCallback, getCachedEnvironment } from "./sidecar";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore, type TrainingResults, type TrainingConfig } from "../store/training-store";
import type { Node } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import { preflightCheck } from "./preflight-check";
import { toast } from "../components/Toast";
import { reviewPreflightFixes } from "../components/PreflightDialog";
import { useHistoryStore } from "../store/history-store";
import { isDatasetNode, extractDatasetConfig } from "./dataset-metadata";
import { buildSidecarGraph } from "./sidecar-graph";
import { requirePython } from "../store/python-store";
import { useRunStore, epochSummary, architectureKey, graphSnapshot, type RunRecord } from "../store/run-store";
import { computeModelSummary } from "./model-summary";

export async function startTraining() {
  const store = useGraphStore.getState();
  let { nodes } = store;
  const { edges } = store;
  const config = useTrainingStore.getState().config;
  const previousStatus = useTrainingStore.getState().status;
  if (previousStatus === "running" || previousStatus === "starting") return;

  if (nodes.length === 0) {
    useTrainingStore.getState().setError("The graph is empty. Add a Data node and some layers first.", "check");
    return;
  }

  if (!requirePython("Training needs Python.")) return;

  useTrainingStore.getState().setStatus("starting");

  // Pre-flight validation
  const preflight = preflightCheck(nodes, edges, config);

  if (!preflight.valid && preflight.errors.length > 0) {
    useTrainingStore.getState().setError(preflight.errors.join("\n"), "check");
    return;
  }

  if (preflight.autoFixes.length > 0) {
    // Show the suggested changes instead of making them silently
    const labelOf = (id: string) => nodes.find((n) => n.id === id)?.data.label ?? id;
    const choice = await reviewPreflightFixes(
      preflight.autoFixes.map((f) => ({ ...f, nodeLabel: labelOf(f.nodeId) })),
      preflight.warnings,
    );
    if (choice === "cancel") {
      useTrainingStore.getState().setStatus(previousStatus);
      return;
    }
    if (choice === "apply") {
      useHistoryStore.getState().pushSnapshot();
      for (const fix of preflight.autoFixes) {
        useGraphStore.getState().updateNodeParam(fix.nodeId, fix.param, fix.newValue);
      }
      nodes = useGraphStore.getState().nodes;
    }
  } else {
    for (const w of preflight.warnings) {
      toast.warning(w);
    }
  }

  // Auto-generate seeds if null (random-but-recorded)
  if (config.seed === null) {
    const seed = Math.floor(Math.random() * 2147483647);
    useTrainingStore.getState().updateConfig({ seed });
  }
  if (config.dataloaderSeed === null) {
    const dataloaderSeed = Math.floor(Math.random() * 2147483647);
    useTrainingStore.getState().updateConfig({ dataloaderSeed });
  }
  // Re-read config after seed updates
  const finalConfig = useTrainingStore.getState().config;

  const graphJson = buildSidecarGraph(nodes, edges);

  // What this run trains, recorded when it ends so runs can be compared later
  const run: RunStart = {
    modelName: useGraphStore.getState().projectName,
    startedAt: new Date().toISOString(),
    config: finalConfig,
    model: {
      params: computeModelSummary(nodes).totalParams,
      layers: nodes.filter((n) => n.data.moduleType !== "Input" && n.data.moduleType !== "Output" && !isDatasetNode(n.data.moduleType)).length,
      key: architectureKey(nodes, edges),
    },
    graph: graphSnapshot(nodes, edges),
  };
  // The run's latest checkpoint, so its model can be tried again after the app restarts
  let checkpoint: RunRecord["checkpoint"];

  // Set up log callback — streams stdout/stderr from training thread
  setTrainLogCallback((params) => {
    useTrainingStore.getState().appendLog({
      text: params.text,
      stream: params.stream,
      timestamp: Date.now(),
    });
  });

  // Set up progress callback
  let earlyStopped = false;
  setTrainProgressCallback((params: unknown) => {
    const p = params as {
      epoch?: number;
      step?: number;
      total_steps?: number;
      dataset?: string;
      train_loss?: number;
      val_loss?: number;
      train_acc?: number;
      val_acc?: number;
      lr?: number;
      status?: string;
      error?: string;
      path?: string;
      gpu?: { allocated_mb: number; reserved_mb: number; max_allocated_mb: number };
    };

    if (p.status === "downloading") {
      useTrainingStore.getState().setDownloading(p.dataset ?? "dataset");
      return;
    }
    if (p.status === "early_stop") {
      earlyStopped = true;
      return;
    }
    if (p.status === "complete") {
      // Compute TrainingResults from accumulated metrics
      const store = useTrainingStore.getState();
      const metrics = store.metrics;
      if (metrics.length > 0) {
        const last = metrics[metrics.length - 1];
        let bestValLoss: number | undefined;
        let bestValEpoch: number | undefined;
        let bestValAcc: number | undefined;
        let bestValAccEpoch: number | undefined;
        for (const m of metrics) {
          if (m.valLoss != null) {
            if (bestValLoss === undefined || m.valLoss < bestValLoss) {
              bestValLoss = m.valLoss;
              bestValEpoch = m.epoch;
            }
          }
          if (m.valAcc != null) {
            if (bestValAcc === undefined || m.valAcc > bestValAcc) {
              bestValAcc = m.valAcc;
              bestValAccEpoch = m.epoch;
            }
          }
        }
        const results: TrainingResults = {
          startedAt: store.startedAt ? new Date(store.startedAt).toISOString() : new Date().toISOString(),
          completedAt: new Date().toISOString(),
          totalEpochs: last.epoch,
          finalTrainLoss: last.trainLoss,
          finalValLoss: last.valLoss,
          bestValLoss,
          bestValEpoch,
          finalTrainAcc: last.trainAcc,
          finalValAcc: last.valAcc,
          bestValAcc,
          bestValAccEpoch,
          earlyStopped: earlyStopped || undefined,
          stoppedByUser: store.stopRequested || undefined,
          plannedEpochs: finalConfig.epochs,
          durationMs: store.startedAt ? Date.now() - store.startedAt : 0,
          environment: getCachedEnvironment() ?? undefined,
        };
        store.setTrainingResults(results);
      }
      recordRun({ ...run, checkpoint }, useTrainingStore.getState().stopRequested ? "stopped" : "finished");
      useTrainingStore.getState().setStatus("finished");
      return;
    }
    if (p.status === "checkpoint") {
      if (p.path) {
        useTrainingStore.getState().setLastCheckpointPath(p.path);
        checkpoint = { path: p.path, epoch: p.epoch ?? 0 };
      }
      return;
    }
    if (p.status === "error" || p.error) {
      recordRun({ ...run, checkpoint }, "failed", p.error ?? "Training error");
      useTrainingStore.getState().setError(p.error ?? "Training error");
      return;
    }
    if (p.epoch != null && p.train_loss != null) {
      const ts = useTrainingStore.getState();
      if (ts.downloading) ts.setDownloading(null);
      if (p.total_steps) {
        ts.setEpochProgress({ step: p.step ?? 0, total: p.total_steps });
      } else if (ts.epochProgress) {
        // End-of-epoch report: the epoch's batches are all done
        ts.setEpochProgress({ step: ts.epochProgress.total, total: ts.epochProgress.total });
      }
      ts.appendMetric({
        epoch: p.epoch,
        step: p.step ?? 0,
        trainLoss: p.train_loss,
        valLoss: p.val_loss,
        trainAcc: p.train_acc,
        valAcc: p.val_acc,
        lr: p.lr ?? finalConfig.optimizer.lr,
        timestamp: Date.now(),
        gpu: p.gpu,
      });
    }
  });

  useTrainingStore.getState().reset();
  useTrainingStore.getState().setStatus("running");

  try {
    await sidecarCall("train.start", {
      graph: graphJson,
      config: sidecarTrainConfig(nodes, finalConfig),
    });
  } catch (e) {
    useTrainingStore.getState().setError(
      e instanceof Error ? e.message : "Failed to start training"
    );
  }
}

/** The training settings as the sidecar takes them. The Data node's settings take precedence over the panel's. */
export function sidecarTrainConfig(nodes: Node<NodeData>[], config: TrainingConfig) {
  const datasetNode = nodes.find((n) => isDatasetNode(n.data.moduleType));
  const datasetOverride = datasetNode
    ? extractDatasetConfig(datasetNode.data.moduleType, datasetNode.data.params)
    : null;
  return {
    optimizer: {
      type: config.optimizer.type,
      lr: config.optimizer.lr,
      weight_decay: config.optimizer.weightDecay,
      momentum: config.optimizer.momentum,
    },
    scheduler: config.scheduler ? {
      type: config.scheduler.type,
      ...config.scheduler.params,
    } : null,
    loss: config.loss,
    epochs: config.epochs,
    batch_size: config.batchSize,
    val_split: datasetOverride?.valSplit ?? config.valSplit,
    dataset: datasetOverride
      ? {
          ...datasetOverride.dataset,
          val_split: datasetOverride.valSplit,
          ...datasetOverride.datasetExtra,
          ...(datasetNode?.data.dataOutputs ? { data_outputs: datasetNode.data.dataOutputs } : {}),
        }
      : config.dataset,
    device: config.device,
    mixed_precision: config.mixedPrecision,
    gradient_clipping: config.gradientClipping.enabled ? { max_norm: config.gradientClipping.maxNorm } : null,
    early_stopping: config.earlyStopping.enabled ? { patience: config.earlyStopping.patience, min_delta: config.earlyStopping.minDelta } : null,
    resume_checkpoint: config.resumeCheckpoint || null,
    augmentation: datasetOverride?.augmentation ?? config.augmentation,
    weight_init: config.weightInit,
    seed: config.seed,
    dataloader_seed: config.dataloaderSeed,
    deterministic: config.deterministicMode,
  };
}

type RunStart = Pick<RunRecord, "modelName" | "startedAt" | "config" | "model" | "graph" | "checkpoint">;

/** Adds the run that just ended to the project's run history. A run stopped before its first epoch has nothing to keep. */
function recordRun(start: RunStart, status: RunRecord["status"], error?: string) {
  const ts = useTrainingStore.getState();
  const epochs = epochSummary(ts.metrics);
  if (status !== "failed" && epochs.length === 0) return;
  const { runs, addRun } = useRunStore.getState();
  addRun({
    ...start,
    id: `run_${Date.now().toString(36)}`,
    number: (runs[runs.length - 1]?.number ?? 0) + 1,
    durationMs: ts.startedAt ? Date.now() - ts.startedAt : 0,
    status,
    error,
    results: status === "failed" ? undefined : ts.trainingResults ?? undefined,
    epochs,
  });
}

export async function stopTraining() {
  useTrainingStore.getState().setStopRequested(true);
  setTrainLogCallback(null);
  try {
    await sidecarCall("train.stop", {});
  } catch {
    // No sidecar, so nothing will report back: end the run here
    useTrainingStore.setState({ status: "idle" });
    return;
  }
  // The sidecar finishes the batch it's on and then reports "complete", which ends the run.
  // If that never arrives, don't leave the app looking busy.
  setTimeout(() => {
    const s = useTrainingStore.getState();
    if (s.status === "running" && s.stopRequested) useTrainingStore.setState({ status: "idle" });
  }, 15000);
}
