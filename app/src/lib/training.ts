/**
 * Training execution — bridge between frontend stores and Python sidecar.
 */

import { sidecarCall, setTrainProgressCallback, setTrainLogCallback, getCachedEnvironment } from "./sidecar";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore, type TrainingResults } from "../store/training-store";
import { preflightCheck } from "./preflight-check";
import { toast } from "../components/Toast";
import { isDatasetNode, extractDatasetConfig } from "./dataset-metadata";
import { buildSidecarGraph } from "./sidecar-graph";

export async function startTraining() {
  const store = useGraphStore.getState();
  let { nodes } = store;
  const { edges } = store;
  const config = useTrainingStore.getState().config;

  if (nodes.length === 0) {
    useTrainingStore.getState().setError("No nodes in graph");
    return;
  }

  useTrainingStore.getState().setStatus("starting");

  // Pre-flight validation and auto-fix
  const preflight = preflightCheck(nodes, edges, config);

  if (preflight.autoFixes.length > 0) {
    // Apply auto-fixes to the graph store
    // Apply fixes to the graph store
    for (const fix of preflight.autoFixes) {
      useGraphStore.getState().updateNodeParam(fix.nodeId, fix.param, fix.newValue);
    }
    // Re-read nodes after fixes
    nodes = useGraphStore.getState().nodes;
    const fixSummary = preflight.autoFixes.map((f) => `${f.param}: ${f.oldValue} → ${f.newValue}`).join(", ");
    toast.info(`Auto-fixed: ${fixSummary}`);
  }

  // Show warnings as toasts
  for (const w of preflight.warnings) {
    toast.warning(w);
  }

  if (!preflight.valid && preflight.errors.length > 0) {
    useTrainingStore.getState().setError(preflight.errors.join("; "));
    return;
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
  setTrainProgressCallback((params: unknown) => {
    const p = params as {
      epoch?: number;
      step?: number;
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

    if (p.status === "complete" || p.status === "early_stop") {
      // Compute TrainingResults from accumulated metrics
      const store = useTrainingStore.getState();
      const metrics = store.metrics;
      if (metrics.length > 0) {
        const last = metrics[metrics.length - 1];
        let bestValLoss: number | undefined;
        let bestValEpoch: number | undefined;
        let bestValAcc: number | undefined;
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
      useTrainingStore.getState().appendMetric({
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
  setTrainLogCallback(null);
  try {
    await sidecarCall("train.stop", {});
  } catch {
    // Ignore if sidecar is not running
  }
  useTrainingStore.getState().setStatus("idle");
}
