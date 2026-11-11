import LLMMemoryTable from "./LLMMemoryTable";
import { LLMFacts, useLLMDevice } from "./ImportLLMDialog";
import { useLLMStore } from "../store/llm-store";

/** Bottom panel for a project imported from a pretrained language model */
export default function LLMPanel() {
  const spec = useLLMStore((s) => s.spec);
  const device = useLLMDevice();
  if (!spec) return null;
  return (
    <div className="h-full overflow-auto p-4">
      <div className="grid gap-6 lg:grid-cols-2 max-w-6xl">
        <section aria-label="Architecture">
          <div className="flex items-baseline justify-between gap-2 mb-3">
            <h2 className="text-sm font-medium text-fg truncate">{spec.name}</h2>
            <span className="text-[11px] text-fg-muted shrink-0">{spec.family}</span>
          </div>
          <LLMFacts spec={spec} />
          {spec.warnings.length > 0 && (
            <ul className="mt-3 px-3 py-2 rounded-lg text-xs bg-warning-subtle text-fg-secondary space-y-1 list-none">
              {spec.warnings.map((w) => <li key={w}>{w}</li>)}
            </ul>
          )}
        </section>
        <section aria-label="Memory needed">
          <h2 className="text-[11px] font-medium mb-2 text-fg-secondary">Memory needed</h2>
          <LLMMemoryTable spec={spec} device={device} />
        </section>
      </div>
    </div>
  );
}
