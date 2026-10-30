import { TEMPLATES } from "../lib/templates";
import { TemplateCard } from "./TemplatesDialog";
import { LogoMark } from "./ui/Logo";
import Button from "./ui/Button";

/** Small, quick-to-train models across the main kinds of data, for a first run */
const FEATURED = ["Simple MLP", "LeNet-5", "nanoGPT"];

/** What an empty canvas shows: ways to get a first model on it */
export default function WelcomeCard({ onSelectTemplate, onShowTemplates, onShowImport, onAddData, onShowGettingStarted }: {
  onSelectTemplate: (index: number) => void;
  onShowTemplates: () => void;
  onShowImport: () => void;
  onAddData: () => void;
  onShowGettingStarted: () => void;
}) {
  const featured = FEATURED
    .map((name) => TEMPLATES.findIndex((t) => t.name === name))
    .filter((i) => i >= 0);

  return (
    <section aria-labelledby="welcome-title" className="pointer-events-none [&_button]:pointer-events-auto w-full max-w-[600px] text-center">
      <LogoMark size={36} className="mx-auto mb-3 text-accent-text" />
      <h2 id="welcome-title" className="text-base font-semibold text-fg mb-1">Build a neural network</h2>
      <p className="text-xs text-fg-secondary mb-5">
        Start from one of these models, or drag layers in from the left.{" "}
        <button type="button" onClick={onShowGettingStarted} className="text-accent-text hover:underline underline-offset-2">
          New here? Getting started
        </button>
      </p>

      <div className="grid grid-cols-3 gap-2.5 mb-3">
        {featured.map((i) => (
          <TemplateCard key={i} template={TEMPLATES[i]} compact onClick={() => onSelectTemplate(i)} />
        ))}
      </div>
      <button type="button" onClick={onShowTemplates}
        className="text-xs font-medium text-accent-text hover:underline underline-offset-2 mb-6">
        Browse all {TEMPLATES.length} templates
      </button>

      <div className="flex items-center justify-center gap-2 mb-4">
        <Button size="sm" onClick={onAddData}>Start with a Data node</Button>
        <Button size="sm" onClick={onShowImport}>Import PyTorch code</Button>
      </div>
      <div className="flex items-center justify-center gap-4 text-[11px] text-fg-muted">
        <span><Kbd>Ctrl+K</Kbd> search layers</span>
        <span><Kbd>Ctrl+I</Kbd> import code</span>
        <span><Kbd>Ctrl+/</Kbd> all shortcuts</span>
      </div>
    </section>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return <kbd className="px-1.5 py-0.5 rounded bg-surface-raised border border-line font-sans">{children}</kbd>;
}
