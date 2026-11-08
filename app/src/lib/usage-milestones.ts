/** Turns app milestones into usage events (see usage.ts). Started once by App. */
import { usePythonStore } from "../store/python-store";
import { useTrainingStore } from "../store/training-store";
import { deviceFamily, firstTime, track, trackAppOpened, useUsageStore } from "./usage";

export function startUsageMilestones(): () => void {
  trackAppOpened();

  const stops = [
    // Saying yes mid-session still counts this launch
    useUsageStore.subscribe((s, prev) => {
      if (s.consent === "granted" && prev.consent !== "granted") trackAppOpened();
    }),
    usePythonStore.subscribe((s, prev) => {
      if (s.justInstalled && !prev.justInstalled && firstTime("pytorch_installed")) {
        track({ name: "pytorch_installed", props: {} });
      }
    }),
    useTrainingStore.subscribe((s, prev) => {
      const r = s.trainingResults;
      if (!r || r === prev.trainingResults) return;
      const env = r.environment;
      const configured = deviceFamily(s.config.device);
      track({
        name: "training_finished",
        props: {
          first: firstTime("training_finished"),
          stopped_early: Boolean(r.stoppedByUser || r.earlyStopped),
          epochs: r.totalEpochs,
          seconds: Math.round(r.durationMs / 1000),
          device: configured === "auto" && env ? (env.gpuName ? "gpu" : "cpu") : configured,
        },
      });
    }),
  ];
  return () => stops.forEach((stop) => stop());
}

export function trackTryItOpened() {
  track({ name: "try_it_opened", props: { first: firstTime("try_it_opened") } });
}
