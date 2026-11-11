import { useLayerStatsStore } from "../store/layer-stats-store";
import {
  layerFlags,
  overlayLabel,
  overlayPosition,
  overlayRange,
  overlayValue,
  typicalGradient,
  type LayerFlag,
  type LayerOverlay,
  type LayerSnapshot,
  type LayerStat,
} from "./layer-health";

interface Scale {
  range: { min: number; max: number } | null;
  typicalGrad: number | null;
}

// Every node reads the same scale, so it's worked out once per snapshot and overlay
const scales = new WeakMap<LayerSnapshot, Partial<Record<LayerOverlay, Scale>>>();

export function overlayScale(snapshot: LayerSnapshot | null, overlay: LayerOverlay): Scale {
  if (!snapshot) return { range: null, typicalGrad: null };
  const bySnapshot = scales.get(snapshot) ?? {};
  let scale = bySnapshot[overlay];
  if (!scale) {
    scale = { range: overlayRange(snapshot, overlay), typicalGrad: typicalGradient(snapshot) };
    bySnapshot[overlay] = scale;
    scales.set(snapshot, bySnapshot);
  }
  return scale;
}

export interface NodeOverlay {
  overlay: LayerOverlay;
  stat: LayerStat | undefined;
  flags: LayerFlag[];
  /** Place on the color scale, 0 to 1, or null when the layer has no value to color by */
  position: number | null;
  label: string | null;
}

/** What the training overlay shows on one layer. Units only count as dead on activation layers. */
export function useNodeOverlay(nodeId: string, unitsCanDie: boolean): NodeOverlay {
  const overlay = useLayerStatsStore((s) => s.overlay);
  const snapshot = useLayerStatsStore((s) => s.snapshot);
  const stat = snapshot?.layers[nodeId];
  if (overlay === "off" || !stat) return { overlay, stat: undefined, flags: [], position: null, label: null };
  const { range, typicalGrad } = overlayScale(snapshot, overlay);
  const value = overlayValue(stat, overlay);
  return {
    overlay,
    stat,
    flags: layerFlags(stat, typicalGrad, unitsCanDie),
    position: value != null && range ? overlayPosition(value, range) : null,
    label: overlayLabel(stat, overlay, unitsCanDie),
  };
}
