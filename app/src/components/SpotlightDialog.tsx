import { useState, useRef, useEffect, useCallback } from "react";
import { MODULE_REGISTRY, CATEGORY_COLORS, CATEGORY_LABELS } from "../lib/registry";
import type { ModuleDef } from "../lib/types";
import { CategoryBadge, tint } from "../lib/category-icons";

export interface ConnectFromContext {
  sourceNodeId: string;
  sourceHandleId: string;
  sourceLabel: string;
}

interface SpotlightProps {
  open: boolean;
  onClose: () => void;
  onSelect: (moduleType: string) => void;
  connectFrom?: ConnectFromContext | null;
}

const ALL_MODULES: ModuleDef[] = Object.values(MODULE_REGISTRY);

export default function SpotlightDialog({ open, onClose, onSelect, connectFrom }: SpotlightProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const results = query.trim()
    ? scoreAndFilter(query.trim().toLowerCase())
    : ALL_MODULES.slice(0, 12);

  // Auto-focus and reset on open
  useEffect(() => {
    if (open) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  // Scroll selected into view
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const item = list.children[selectedIndex] as HTMLElement | undefined;
    item?.scrollIntoView({ block: "nearest" });
  }, [selectedIndex]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, results.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        const selected = results[selectedIndex];
        if (selected) {
          onSelect(selected.type);
          onClose();
        }
      } else if (e.key === "Escape") {
        onClose();
      }
    },
    [results, selectedIndex, onSelect, onClose],
  );

  if (!open) return null;

  const placeholder = connectFrom
    ? `Add layer after ${connectFrom.sourceLabel}...`
    : "Search layers... (e.g. Conv2d, Linear, GELU)";

  return (
    <>
      <div className="fixed inset-0 z-50" style={{ background: "rgba(0,0,0,0.4)" }} onClick={onClose} />

      <div
        className="fixed z-50 top-[15%] left-1/2 -translate-x-1/2 w-[440px] rounded-xl shadow-2xl overflow-hidden"
        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}
      >
        {/* Context banner when connecting */}
        {connectFrom && (
          <div
            className="px-4 py-2 text-[11px] font-medium"
            style={{ background: "var(--accent-subtle)", color: "var(--accent-text)", borderBottom: "1px solid var(--border)" }}
          >
            Adding connected layer after <strong>{connectFrom.sourceLabel}</strong>
          </div>
        )}

        {/* Search input */}
        <div className="flex items-center px-4 gap-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => { setQuery(e.target.value); setSelectedIndex(0); }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="flex-1 py-3 text-sm bg-transparent outline-none"
            style={{ color: "var(--text-primary)" }}
          />
        </div>

        {/* Results */}
        <div ref={listRef} className="max-h-[320px] overflow-y-auto py-1">
          {results.length === 0 ? (
            <div className="px-4 py-6 text-center text-xs" style={{ color: "var(--text-muted)" }}>
              No matching layers found
            </div>
          ) : (
            results.map((mod, i) => {
              const color = CATEGORY_COLORS[mod.category];
              const isSelected = i === selectedIndex;
              return (
                <button
                  key={mod.type}
                  className="flex items-center w-full px-4 py-2 gap-3 text-left transition-colors"
                  style={{
                    background: isSelected ? "var(--accent-subtle)" : "transparent",
                  }}
                  onClick={() => { onSelect(mod.type); onClose(); }}
                  onMouseEnter={() => setSelectedIndex(i)}
                >
                  <CategoryBadge category={mod.category} color={color} size={24} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
                        {mod.label}
                      </span>
                      <span
                        className="text-[11px] px-1.5 py-0.5 rounded font-medium"
                        style={{ background: tint(color, 12), color }}
                      >
                        {CATEGORY_LABELS[mod.category]}
                      </span>
                    </div>
                    <div className="text-[11px] truncate mt-0.5" style={{ color: "var(--text-muted)" }}>
                      {mod.description}
                    </div>
                  </div>
                </button>
              );
            })
          )}
        </div>

        {/* Footer hint */}
        <div
          className="px-4 py-2 text-[11px] flex items-center gap-3"
          style={{ borderTop: "1px solid var(--border)", color: "var(--text-muted)" }}
        >
          <span><kbd className="px-1 py-0.5 rounded text-[11px]" style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)" }}>Enter</kbd> to add</span>
          <span><kbd className="px-1 py-0.5 rounded text-[11px]" style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)" }}>&uarr;&darr;</kbd> to navigate</span>
          <span><kbd className="px-1 py-0.5 rounded text-[11px]" style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)" }}>Esc</kbd> to close</span>
        </div>
      </div>
    </>
  );
}

function scoreAndFilter(query: string): ModuleDef[] {
  const scored = ALL_MODULES.map((mod) => {
    let score = 0;
    const label = mod.label.toLowerCase();
    const type = mod.type.toLowerCase();
    const desc = mod.description.toLowerCase();

    if (type === query) score += 100;
    else if (type.startsWith(query)) score += 80;
    else if (label.startsWith(query)) score += 70;
    else if (type.includes(query)) score += 50;
    else if (label.includes(query)) score += 40;
    else if (desc.includes(query)) score += 20;
    else if (mod.category.includes(query)) score += 15;
    else return null;

    return { mod, score };
  }).filter(Boolean) as Array<{ mod: ModuleDef; score: number }>;

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 12).map((s) => s.mod);
}
