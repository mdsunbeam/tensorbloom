import LLMMemoryTable from "./LLMMemoryTable";
import LLMChat from "./LLMChat";
import LLMFinetune from "./LLMFinetune";
import Segmented from "./ui/Segmented";
import { LLMFacts, useLLMDevice } from "./ImportLLMDialog";
import { useLLMStore } from "../store/llm-store";

/** Bottom panel for a project imported from a pretrained language model */
export default function LLMPanel({ onShowCharts }: { onShowCharts: () => void }) {
  const spec = useLLMStore((s) => s.spec);
  const device = useLLMDevice();
  const view = useLLMStore((s) => s.view);
  const setView = (v: typeof view) => useLLMStore.setState({ view: v });
  if (!spec) return null;
  return (
    <div className="h-full flex flex-col min-h-0 p-4 gap-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex items-baseline gap-2">
          <h2 className="text-sm font-medium text-fg truncate">{spec.name}</h2>
          <span className="text-[11px] text-fg-muted shrink-0">{spec.family}</span>
        </div>
        <Segmented label="Language model view" size="sm" value={view} onChange={setView}
          options={[{ value: "overview", label: "Overview" }, { value: "chat", label: "Chat" }, { value: "finetune", label: <span className="whitespace-nowrap">Fine-tune</span> }]} />
      </div>
      {view === "overview" ? (
        <div className="flex-1 min-h-0 overflow-auto">
          <div className="grid gap-6 lg:grid-cols-2 max-w-6xl">
            <section aria-label="Architecture">
              <LLMFacts spec={spec} />
              {spec.warnings.length > 0 && (
                <ul className="mt-3 px-3 py-2 rounded-lg text-xs bg-warning-subtle text-fg-secondary space-y-1 list-none">
                  {spec.warnings.map((w) => <li key={w}>{w}</li>)}
                </ul>
              )}
            </section>
            <section aria-label="Memory needed">
              <h3 className="text-[11px] font-medium mb-2 text-fg-secondary">Memory needed</h3>
              <LLMMemoryTable spec={spec} device={device} />
            </section>
          </div>
        </div>
      ) : view === "chat" ? (
        <div className="flex-1 min-h-0 max-w-4xl w-full"><LLMChat /></div>
      ) : (
        <div className="flex-1 min-h-0 overflow-auto"><LLMFinetune onShowCharts={onShowCharts} /></div>
      )}
    </div>
  );
}
