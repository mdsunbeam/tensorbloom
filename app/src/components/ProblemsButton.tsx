import { useEffect, useRef, useState } from "react";
import { useGraphStore } from "../store/graph-store";
import { formatShape } from "../lib/shape-inference";
import { shapeErrorFix, shapeErrorSummary } from "../lib/shape-fixes";
import { focusNode } from "../lib/canvas-focus";
import { useEscapeKey } from "../lib/use-escape-key";
import { CategoryBadge } from "../lib/category-icons";
import { CATEGORY_COLORS } from "../lib/registry";
import Button from "./ui/Button";

/**
 * "2 problems" in the toolbar when layers can't take the shapes they're given. Opens a list
 * that jumps to each layer and offers the one-click fix when there is one.
 */
export default function ProblemsButton() {
  const shapeErrors = useGraphStore((s) => s.shapeErrors);
  const nodes = useGraphStore((s) => s.nodes);
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Only the layers where a problem starts; the ones after them follow from it
  const problems = nodes
    .filter((n) => shapeErrors[n.id]?.type === "root")
    .map((n) => ({ node: n, error: shapeErrors[n.id] }));

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as HTMLElement)) setOpen(false); };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);
  useEscapeKey(() => setOpen(false), { enabled: open });
  useEffect(() => { if (problems.length === 0) setOpen(false); }, [problems.length]);

  if (problems.length === 0) return null;

  return (
    <div ref={ref} className="relative flex-shrink-0">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-haspopup="dialog"
        className="flex items-center gap-1.5 h-6 px-2 rounded-md text-[11px] font-medium whitespace-nowrap text-danger bg-danger-subtle hover:brightness-110 transition"
      >
        <AlertIcon />
        {problems.length} {problems.length === 1 ? "problem" : "problems"}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Problems"
          className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-[380px] max-h-[60vh] overflow-y-auto rounded-xl bg-surface border border-line z-50"
          style={{ boxShadow: "0 8px 24px rgba(0,0,0,0.25), 0 2px 6px rgba(0,0,0,0.12)" }}
        >
          <div className="px-3.5 pt-3 pb-2 text-[11px] text-fg-muted border-b border-line-subtle">
            These layers can't take the shape they're given. Layers after them show "shape unknown" until they're fixed.
          </div>
          <ul className="py-1">
            {problems.map(({ node, error }) => {
              const fix = shapeErrorFix(error, node.data.params);
              return (
                <li key={node.id} className="px-2">
                  <div className="flex items-start gap-2.5 px-1.5 py-2 rounded-lg hover:bg-surface-raised">
                    <CategoryBadge category={node.data.category} color={CATEGORY_COLORS[node.data.category]} size={20} />
                    <div className="flex-1 min-w-0">
                      <button
                        type="button"
                        onClick={() => { focusNode(node.id); setOpen(false); }}
                        className="text-xs font-semibold text-fg hover:underline underline-offset-2 text-left"
                      >
                        {node.data.label}
                      </button>
                      <div className="text-[11px] font-mono text-danger mt-0.5 break-words">
                        {shapeErrorSummary(error, formatShape)}
                      </div>
                      {error.receivedShape && error.expectedShape && (
                        <div className="text-[11px] text-fg-muted mt-0.5">{error.message}</div>
                      )}
                      {fix && (
                        <Button size="sm" className="mt-1.5" onClick={() => updateNodeParam(node.id, fix.param, fix.to)}>
                          Set {fix.param} to {fix.to}
                        </Button>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

function AlertIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}
