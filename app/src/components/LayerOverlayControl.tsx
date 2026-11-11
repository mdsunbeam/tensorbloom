import { useMemo, useState } from "react";
import { useLayerStatsStore } from "../store/layer-stats-store";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore } from "../store/training-store";
import { architectureKey } from "../store/run-store";
import { formatStat, layerFlags, overlayColor, type LayerOverlay } from "../lib/layer-health";
import { overlayScale } from "../lib/use-layer-overlay";
import { focusNode } from "../lib/canvas-focus";
import Segmented from "./ui/Segmented";

const OPTIONS: { value: LayerOverlay; label: string; title: string }[] = [
  { value: "off", label: "Off", title: "Hide training stats on the layers" },
  { value: "gradients", label: "Gradients", title: "Color each layer by how big the gradient reaching it is" },
  { value: "activations", label: "Activations", title: "Color each layer by how spread out its outputs are" },
];

const DESCRIBE: Record<Exclude<LayerOverlay, "off">, string> = {
  gradients: "Gradient reaching each layer",
  activations: "Spread (σ) of each layer's outputs",
};

/**
 * Shown on the canvas once a run has layer stats: picks what the layers are colored by,
 * explains the scale, and lists the layers whose stats look wrong.
 */
export default function LayerOverlayControl() {
  const overlay = useLayerStatsStore((s) => s.overlay);
  const setOverlay = useLayerStatsStore((s) => s.setOverlay);
  const snapshot = useLayerStatsStore((s) => s.snapshot);
  const source = useLayerStatsStore((s) => s.source);
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const training = useTrainingStore((s) => s.status === "running" || s.status === "starting");
  const [listOpen, setListOpen] = useState(false);

  const changed = useMemo(
    () => source?.kind === "run" && architectureKey(nodes, edges) !== source.modelKey,
    [source, nodes, edges],
  );

  const flagged = useMemo(() => {
    if (!snapshot || overlay === "off") return [];
    const { typicalGrad } = overlayScale(snapshot, overlay);
    return nodes
      .map((n) => ({ node: n, flags: layerFlags(snapshot.layers[n.id], typicalGrad, n.data.category === "activation") }))
      .filter((x) => x.flags.length > 0);
  }, [snapshot, overlay, nodes]);

  if (!snapshot && !(training && source?.kind === "live")) return null;

  const range = snapshot && overlay !== "off" ? overlayScale(snapshot, overlay).range : null;
  const from = source?.kind === "run"
    ? `Run ${source.runNumber}${snapshot ? `, epoch ${snapshot.epoch}` : ""}${changed ? ". The model has changed since." : ""}`
    : snapshot ? `Training now, epoch ${snapshot.epoch}` : "Waiting for the first batch";

  return (
    <div className="absolute top-3 left-3 z-10 w-[236px] rounded-lg border border-line bg-surface p-2.5 space-y-2 text-[11px]" style={{ boxShadow: "var(--node-shadow)" }}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-semibold text-fg-secondary">Training on the graph</span>
      </div>
      <Segmented label="Color layers by" size="sm" value={overlay} onChange={setOverlay} options={OPTIONS} />
      {overlay !== "off" && (
        <>
          <div>
            <div className="text-fg-secondary">{DESCRIBE[overlay]}</div>
            {range ? (
              <div className="mt-1">
                <div className="h-2 rounded-sm" style={{ background: `linear-gradient(to right, ${overlayColor(0)}, ${overlayColor(0.5)}, ${overlayColor(1)})` }} aria-hidden="true" />
                <div className="flex justify-between font-mono tabular-nums text-fg-muted mt-0.5">
                  <span>{formatStat(range.min)}</span>
                  <span>{range.max > range.min ? formatStat(range.max) : ""}</span>
                </div>
              </div>
            ) : (
              <div className="text-fg-muted mt-0.5">No values yet</div>
            )}
          </div>
          {flagged.length > 0 && (
            <div>
              <button type="button" className="w-full text-left font-medium text-warning hover:underline" aria-expanded={listOpen} onClick={() => setListOpen(!listOpen)}>
                {flagged.length} layer{flagged.length === 1 ? "" : "s"} to look at {listOpen ? "▴" : "▾"}
              </button>
              {listOpen && (
                <ul className="mt-1 space-y-0.5 max-h-40 overflow-y-auto">
                  {flagged.map(({ node, flags }) => (
                    <li key={node.id}>
                      <button type="button" className="w-full flex justify-between gap-2 px-1 py-0.5 rounded hover:bg-surface-raised text-left" onClick={() => focusNode(node.id)}>
                        <span className="truncate text-fg">{node.data.label}</span>
                        <span className={`flex-shrink-0 ${flags[0].severity === "bad" ? "text-danger" : "text-warning"}`}>{flags[0].title}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
          <div className="text-fg-muted">{from}. Hover a layer for details.</div>
        </>
      )}
    </div>
  );
}
