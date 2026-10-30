import { TEMPLATES, TEMPLATE_CATEGORY_ORDER, TEMPLATE_CATEGORY_LABELS } from "../lib/templates";
import type { Template, TemplateCategory } from "../lib/templates";
import { templatePreview } from "./TemplatePreview";
import TemplatePreview from "./TemplatePreview";
import Dialog from "./ui/Dialog";

interface TemplatesDialogProps {
  open: boolean;
  onClose: () => void;
  onSelect: (index: number) => void;
}

/** Group templates by category, preserving original indices for onSelect. */
function groupedTemplates(): { category: TemplateCategory; items: { index: number; template: Template }[] }[] {
  const groups: Record<string, { index: number; template: Template }[]> = {};
  TEMPLATES.forEach((template, index) => {
    (groups[template.category] ??= []).push({ index, template });
  });
  return TEMPLATE_CATEGORY_ORDER
    .filter((cat) => groups[cat]?.length)
    .map((cat) => ({ category: cat, items: groups[cat] }));
}

export default function TemplatesDialog({ open, onClose, onSelect }: TemplatesDialogProps) {
  const sections = groupedTemplates();

  return (
    <Dialog open={open} onClose={onClose} width={680} title="Architecture templates" description="Well-known models from papers and courses, ready to train">
      <div className="p-4 space-y-5">
        {sections.map(({ category, items }) => (
          <section key={category} aria-label={TEMPLATE_CATEGORY_LABELS[category]}>
            <h3 className="text-[11px] font-semibold uppercase tracking-wider pb-2 text-fg-muted">
              {TEMPLATE_CATEGORY_LABELS[category]}
            </h3>
            <div className="grid grid-cols-2 gap-2.5">
              {items.map(({ index, template }) => (
                <TemplateCard key={index} template={template} onClick={() => { onSelect(index); onClose(); }} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </Dialog>
  );
}

/** A template as a card: a preview of its layers, its name, and what it trains on */
export function TemplateCard({ template, onClick, compact = false }: { template: Template; onClick: () => void; compact?: boolean }) {
  const layers = templatePreview(template).layers;
  const dataset = template.trainingDefaults?.dataset?.type;
  return (
    <button
      type="button"
      onClick={onClick}
      className="group text-left rounded-lg border border-line bg-surface overflow-hidden transition-colors hover:border-accent focus-visible:border-accent"
    >
      <div className="px-3 py-1 bg-surface-sunken border-b border-line-subtle group-hover:bg-surface-raised transition-colors">
        <TemplatePreview template={template} height={compact ? 40 : 48} />
      </div>
      <div className="px-3 py-2.5">
        <div className="flex items-baseline gap-2">
          <span className="text-xs font-semibold text-fg truncate">{template.name}</span>
          {template.year && <span className="text-[11px] text-fg-muted">{template.year}</span>}
        </div>
        {!compact && <div className="text-[11px] mt-0.5 text-fg-secondary line-clamp-2">{template.description}</div>}
        <div className="flex items-center gap-1.5 mt-1.5 text-[11px] text-fg-muted">
          <span>{layers} layers</span>
          {dataset && <><span aria-hidden="true">·</span><span>{dataset}</span></>}
        </div>
        {!compact && template.paper && <div className="text-[11px] mt-0.5 italic text-fg-muted truncate" title={template.paper}>{template.paper}</div>}
      </div>
    </button>
  );
}
