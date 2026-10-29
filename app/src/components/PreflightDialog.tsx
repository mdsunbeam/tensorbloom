import { useRef } from "react";
import { create } from "zustand";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";

export interface SuggestedFix {
  nodeLabel: string;
  param: string;
  oldValue: unknown;
  newValue: unknown;
  reason: string;
}

export type PreflightChoice = "apply" | "skip" | "cancel";

interface Request {
  fixes: SuggestedFix[];
  warnings: string[];
  resolve: (choice: PreflightChoice) => void;
}

const usePreflightStore = create<{ request: Request | null }>(() => ({ request: null }));

/** Shows the changes the pre-flight check suggests and asks whether to make them before training */
export function reviewPreflightFixes(fixes: SuggestedFix[], warnings: string[]): Promise<PreflightChoice> {
  return new Promise((resolve) => {
    usePreflightStore.getState().request?.resolve("cancel");
    usePreflightStore.setState({ request: { fixes, warnings, resolve } });
  });
}

const show = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v));

export default function PreflightDialog() {
  const request = usePreflightStore((s) => s.request);
  const applyRef = useRef<HTMLButtonElement>(null);
  const shown = useRef(request);
  if (request) shown.current = request;
  const fixes = shown.current?.fixes ?? [];
  const warnings = shown.current?.warnings ?? [];

  const answer = (choice: PreflightChoice) => {
    request?.resolve(choice);
    usePreflightStore.setState({ request: null });
  };

  return (
    <Dialog
      open={request !== null}
      onClose={() => answer("cancel")}
      role="alertdialog"
      width={480}
      initialFocus={applyRef}
      title={fixes.length === 1 ? "One change before training" : `${fixes.length} changes before training`}
      description={fixes.length === 1
        ? "This doesn't match the data. TensorBloom can fix it for you, and you can undo the change afterward."
        : "These don't match the data. TensorBloom can fix them for you, and you can undo the changes afterward."}
      footer={
        <>
          <Button onClick={() => answer("cancel")} className="mr-auto">Cancel</Button>
          <Button onClick={() => answer("skip")}>Train without changes</Button>
          <Button ref={applyRef} variant="primary" onClick={() => answer("apply")}>Fix and train</Button>
        </>
      }
    >
      <ul className="px-5 py-4 space-y-3">
        {fixes.map((f, i) => (
          <li key={i} className="text-xs">
            <div className="flex items-baseline gap-2 flex-wrap">
              <span className="font-semibold text-fg">{f.nodeLabel}</span>
              <span className="font-mono text-fg-secondary">
                {f.param}: <span className="line-through text-fg-muted">{show(f.oldValue)}</span> → <span className="text-fg">{show(f.newValue)}</span>
              </span>
            </div>
            <div className="mt-0.5 text-[11px] leading-relaxed text-fg-muted">{f.reason}</div>
          </li>
        ))}
      </ul>
      {warnings.length > 0 && (
        <div className="mx-5 mb-4 px-3 py-2 rounded-lg text-[11px] leading-relaxed bg-warning-subtle text-fg-secondary">
          <div className="font-semibold text-warning mb-0.5">Also worth checking</div>
          <ul className="list-disc pl-4 space-y-0.5">
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}
    </Dialog>
  );
}
