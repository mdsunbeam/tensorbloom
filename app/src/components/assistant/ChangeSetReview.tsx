import { useMemo } from "react";
import { useChangeSetStore, pendingSummary, applyChanges, discardChanges } from "../../lib/assistant/change-set";
import { useGraphStore } from "../../store/graph-store";
import { useTrainingStore } from "../../store/training-store";
import Button from "../ui/Button";

/** The assistant's pending changes and the Apply / Discard choice, in a compact or a detailed form */
export default function ChangeSetReview({ variant }: { variant: "canvas" | "panel" }) {
  const pending = useChangeSetStore((s) => s.base !== null);
  const version = useChangeSetStore((s) => s.version);
  // The summary depends on the graph and settings as they are now
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const shapeErrors = useGraphStore((s) => s.shapeErrors);
  const config = useTrainingStore((s) => s.config);
  const summary = useMemo(() => (pending ? pendingSummary() : null), [pending, version, nodes, edges, shapeErrors, config]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!summary) return null;
  const lines = [...summary.readable, ...summary.training.map((t) => `Training ${t}`)];
  const count = lines.length;
  const problems = summary.newProblems.length;

  if (variant === "canvas") {
    return (
      <div role="region" aria-label="Assistant's changes"
        className="flex items-center gap-3 pl-4 pr-2 py-2 rounded-xl border border-accent bg-surface text-xs text-fg-secondary"
        style={{ boxShadow: "var(--node-shadow-hover)" }}>
        <span>
          <span className="font-semibold text-fg">Previewing the assistant's changes</span>
          {" · "}{count} {count === 1 ? "change" : "changes"}
          {problems > 0 && <span className="text-danger">{" · "}{problems} new shape {problems === 1 ? "error" : "errors"}</span>}
        </span>
        <Button size="sm" onClick={discardChanges}>Discard</Button>
        <Button size="sm" variant="primary" onClick={applyChanges}>Apply</Button>
      </div>
    );
  }

  return (
    <div className="mx-3 mb-2 rounded-lg border border-accent bg-accent-subtle text-[11px] leading-relaxed">
      <div className="px-3 pt-2.5 font-semibold text-fg">{count} {count === 1 ? "change" : "changes"} to review</div>
      <ul className="px-3 pt-1 max-h-32 overflow-y-auto text-fg-secondary space-y-0.5">
        {lines.map((line, i) => <li key={i}>{line}</li>)}
      </ul>
      {problems > 0 && (
        <p className="px-3 pt-1 text-danger">
          {problems} new shape {problems === 1 ? "error" : "errors"}: {summary.newProblems.map((p) => layerName(p.layer)).join(", ")}
        </p>
      )}
      <div className="flex items-center justify-end gap-2 px-3 py-2">
        <span className="flex-1 text-fg-muted">Apply is one undo step.</span>
        <Button size="sm" onClick={discardChanges}>Discard</Button>
        <Button size="sm" variant="primary" onClick={applyChanges}>Apply</Button>
      </div>
    </div>
  );
}

function layerName(id: string): string {
  return useGraphStore.getState().nodes.find((n) => n.id === id)?.data.label ?? id;
}
