import { TEMPLATES, TEMPLATE_CATEGORY_ORDER, TEMPLATE_CATEGORY_LABELS } from "../lib/templates";
import type { TemplateCategory } from "../lib/templates";
import Dialog from "./ui/Dialog";

interface TemplatesDialogProps {
  open: boolean;
  onClose: () => void;
  onSelect: (index: number) => void;
}

/** Group templates by category, preserving original indices for onSelect. */
function groupedTemplates(): { category: TemplateCategory; items: { index: number; name: string; description: string; year?: number; paper?: string }[] }[] {
  const groups: Record<string, { index: number; name: string; description: string; year?: number; paper?: string }[]> = {};
  TEMPLATES.forEach((tpl, i) => {
    const cat = tpl.category;
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push({ index: i, name: tpl.name, description: tpl.description, year: tpl.year, paper: tpl.paper });
  });
  return TEMPLATE_CATEGORY_ORDER
    .filter((cat) => groups[cat]?.length)
    .map((cat) => ({ category: cat, items: groups[cat] }));
}

export default function TemplatesDialog({ open, onClose, onSelect }: TemplatesDialogProps) {
  const sections = groupedTemplates();

  return (
    <Dialog open={open} onClose={onClose} width={520} title="Architecture templates" description="Well-known models from papers and courses">
      <div className="p-3 space-y-4">
        {sections.map(({ category, items }) => (
          <section key={category} aria-label={TEMPLATE_CATEGORY_LABELS[category]}>
            <h3 className="text-[11px] font-semibold uppercase tracking-wider px-2 pb-1.5 text-fg-muted">
              {TEMPLATE_CATEGORY_LABELS[category]}
            </h3>
            <div className="space-y-1.5">
              {items.map(({ index, name, description, year, paper }) => (
                <button
                  key={index}
                  onClick={() => { onSelect(index); onClose(); }}
                  className="w-full text-left px-4 py-2.5 rounded-lg transition-colors border border-line hover:bg-surface-raised hover:border-accent focus-visible:border-accent"
                >
                  <div className="flex items-baseline gap-2">
                    <span className="text-xs font-semibold text-fg">{name}</span>
                    {year && <span className="text-[11px] text-fg-muted">{year}</span>}
                  </div>
                  <div className="text-[11px] mt-0.5 text-fg-secondary">{description}</div>
                  {paper && <div className="text-[11px] mt-0.5 italic text-fg-muted">{paper}</div>}
                </button>
              ))}
            </div>
          </section>
        ))}
      </div>
    </Dialog>
  );
}
