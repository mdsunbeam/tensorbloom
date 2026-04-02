import { TEMPLATES, TEMPLATE_CATEGORY_ORDER, TEMPLATE_CATEGORY_LABELS } from "../lib/templates";
import type { TemplateCategory } from "../lib/templates";

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
  if (!open) return null;

  const sections = groupedTemplates();

  return (
    <>
      <div
        className="fixed inset-0 z-50"
        style={{ background: "rgba(0,0,0,0.5)" }}
        onClick={onClose}
      />

      <div
        className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[520px] max-h-[75vh] rounded-xl shadow-2xl flex flex-col overflow-hidden"
        style={{
          background: "var(--bg-secondary)",
          border: "1px solid var(--border)",
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-3"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div>
            <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Architecture Templates
            </div>
            <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
              Gold-standard models from papers and courses
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded-md transition-colors"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-tertiary)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Template list grouped by category */}
        <div className="flex-1 overflow-y-auto p-3 space-y-4">
          {sections.map(({ category, items }) => (
            <div key={category}>
              {/* Category header */}
              <div
                className="text-[10px] font-bold uppercase tracking-wider px-2 pb-1.5"
                style={{ color: "var(--text-muted)" }}
              >
                {TEMPLATE_CATEGORY_LABELS[category]}
              </div>
              <div className="space-y-1.5">
                {items.map(({ index, name, description, year, paper }) => (
                  <button
                    key={index}
                    onClick={() => { onSelect(index); onClose(); }}
                    className="w-full text-left px-4 py-2.5 rounded-lg transition-colors"
                    style={{ border: "1px solid var(--border)" }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = "var(--bg-tertiary)";
                      e.currentTarget.style.borderColor = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "transparent";
                      e.currentTarget.style.borderColor = "var(--border)";
                    }}
                  >
                    <div className="flex items-baseline gap-2">
                      <span className="text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
                        {name}
                      </span>
                      {year && (
                        <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                          {year}
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] mt-0.5" style={{ color: "var(--text-secondary)" }}>
                      {description}
                    </div>
                    {paper && (
                      <div className="text-[10px] mt-0.5 italic" style={{ color: "var(--text-muted)" }}>
                        {paper}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
