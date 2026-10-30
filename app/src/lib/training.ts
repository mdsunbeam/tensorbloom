/**
 * Training execution — bridge between frontend stores and Python sidecar.
 */

import { sidecarCall, setTrainProgressCallback, setTrainLogCallback, getCachedEnvironment } from "./sidecar";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore, type TrainingResults } from "../store/training-store";
import { preflightCheck } from "./preflight-check";
import { toast } from "../components/Toast";
import { reviewPreflightFixes } from "../components/PreflightDialog";
import { useHistoryStore } from "../store/history-store";
import { isDatasetNode, extractDatasetConfig } from "./dataset-metadata";
import { buildSidecarGraph } from "./sidecar-graph";

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

  // Extract dataset config from Dataset node if one exists in the graph
  const datasetNode = nodes.find((n) => isDatasetNode(n.data.moduleType));
  const datasetOverride = datasetNode
    ? extractDatasetConfig(datasetNode.data.moduleType, datasetNode.data.params)
    : null;

  const graphJson = buildSidecarGraph(nodes, edges);

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
      useTrainingStore.getState().setStatus("finished");
      return;
    }
    if (p.status === "checkpoint") {
      if (p.path) useTrainingStore.getState().setLastCheckpointPath(p.path);
      return;
    }
    if (p.status === "error" || p.error) {
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
      config: {
        optimizer: {
          type: finalConfig.optimizer.type,
          lr: finalConfig.optimizer.lr,
          weight_decay: finalConfig.optimizer.weightDecay,
          momentum: finalConfig.optimizer.momentum,
        },
        scheduler: finalConfig.scheduler ? {
          type: finalConfig.scheduler.type,
          ...finalConfig.scheduler.params,
        } : null,
        loss: finalConfig.loss,
        epochs: finalConfig.epochs,
        batch_size: finalConfig.batchSize,
        val_split: datasetOverride?.valSplit ?? finalConfig.valSplit,
        dataset: datasetOverride
          ? {
              ...datasetOverride.dataset,
              val_split: datasetOverride.valSplit,
              ...datasetOverride.datasetExtra,
              ...(datasetNode?.data.dataOutputs ? { data_outputs: datasetNode.data.dataOutputs } : {}),
            }
          : finalConfig.dataset,
        device: finalConfig.device,
        mixed_precision: finalConfig.mixedPrecision,
        gradient_clipping: finalConfig.gradientClipping.enabled ? { max_norm: finalConfig.gradientClipping.maxNorm } : null,
        early_stopping: finalConfig.earlyStopping.enabled ? { patience: finalConfig.earlyStopping.patience, min_delta: finalConfig.earlyStopping.minDelta } : null,
        resume_checkpoint: finalConfig.resumeCheckpoint || null,
        augmentation: datasetOverride?.augmentation ?? finalConfig.augmentation,
        weight_init: finalConfig.weightInit,
        seed: finalConfig.seed,
        dataloader_seed: finalConfig.dataloaderSeed,
        deterministic: finalConfig.deterministicMode,
      },
    });
  } catch (e) {
    useTrainingStore.getState().setError(
      e instanceof Error ? e.message : "Failed to start training"
    );
  }
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
