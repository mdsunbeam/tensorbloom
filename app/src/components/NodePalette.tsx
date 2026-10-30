import { useState, useCallback } from "react";
import { useGraphStore } from "../store/graph-store";
import {
  getModulesByCategory,
  CATEGORY_LABELS,
  CATEGORY_COLORS,
  MODULE_REGISTRY,
} from "../lib/registry";
import type { NodeCategory, ModuleDef } from "../lib/types";
import { startDrag, endDrag } from "./DragOverlay";
import { TEMPLATES } from "../lib/templates";
import { CategoryIcon, tint } from "../lib/category-icons";

const modulesByCategory = getModulesByCategory();
const layerCategories = (Object.keys(modulesByCategory) as NodeCategory[]).filter((c) => c !== "data" && c !== "io");
const dataModules = modulesByCategory["data"] ?? [];
// Open at start: the data source and the layers most models begin with
const OPEN_AT_START = ["data", "linear", "conv", "activation"];

/** Adds a layer below the others, for a click or the keyboard instead of a drag */
function addAtDefaultSpot(moduleType: string) {
  const { addNode, nodes } = useGraphStore.getState();
  const snap = (v: number) => Math.round(v / 20) * 20;
  addNode(moduleType, { x: snap(260), y: snap(40 + nodes.length * 100) });
}

export default function NodePalette() {
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(OPEN_AT_START));

  const toggle = (cat: string) => {
    const next = new Set(expanded);
    if (next.has(cat)) next.delete(cat);
    else next.add(cat);
    setExpanded(next);
  };

  const onMouseDown = useCallback((e: React.MouseEvent, moduleType: string) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const def = MODULE_REGISTRY[moduleType];
    if (!def) return;

    const startX = e.clientX;
    const startY = e.clientY;
    let dragging = false;
    const THRESHOLD = 5;

    const onMove = (me: MouseEvent) => {
      const dx = me.clientX - startX;
      const dy = me.clientY - startY;
      if (!dragging && Math.sqrt(dx * dx + dy * dy) > THRESHOLD) {
        dragging = true;
        startDrag(moduleType, def.label, def.category, me.clientX, me.clientY);
      }
      if (dragging) {
        // updateDrag is called by the DragOverlay's own mousemove listener
      }
    };

    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      if (!dragging) {
        // Didn't drag far enough — treat as click, place node snapped to grid
        addAtDefaultSpot(moduleType);
      } else {
        // Clean up drag state if GraphEditor's onMouseUp didn't handle the drop
        // (e.g., mouse released outside the graph area)
        endDrag();
      }
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, []);

  const lowerSearch = search.toLowerCase();

  const filterModules = (modules: ModuleDef[]) =>
    modules.filter(
      (m) =>
        !lowerSearch ||
        m.label.toLowerCase().includes(lowerSearch) ||
        m.type.toLowerCase().includes(lowerSearch) ||
        m.description.toLowerCase().includes(lowerSearch),
    );

  const filteredDataModules = filterModules(dataModules);
  const dataOpen = expanded.has("data") || !!search;

  // Count only layer modules (exclude data); while searching, count only matches
  const layerCount = layerCategories.reduce(
    (acc, k) => acc + filterModules(modulesByCategory[k] ?? []).length,
    0,
  );
  const noMatches = !!search && layerCount === 0 && filteredDataModules.length === 0;
  const matchingTemplates = noMatches
    ? TEMPLATES.filter((t) => t.name.toLowerCase().includes(lowerSearch)).map((t) => t.name)
    : [];

  return (
    <div
      className="flex flex-col h-full overflow-hidden"
      style={{
        background: "var(--bg-secondary)",
      }}
    >
      {/* Header */}
      <div className="flex items-center px-3 pt-3 pb-2">
        <div className="relative flex-1">
          <svg
            className="absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none"
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--text-muted)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            placeholder="Search layers..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-8 pr-3 py-2 text-xs rounded-lg"
            style={{
              background: "var(--bg-primary)",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-4">
        {/* ── Data Sources section ── */}
        {filteredDataModules.length > 0 && (
          <>
            <div className="px-2 pt-1 pb-1.5 flex items-center justify-between">
              <h2 className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                Data Sources
              </h2>
              <span className="text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>
                {filteredDataModules.length}
              </span>
            </div>
            <CategoryGroup
              category="data"
              label="Datasets"
              modules={filteredDataModules}
              open={dataOpen}
              onToggle={() => toggle("data")}
              onItemMouseDown={onMouseDown}
            />

            {/* Divider */}
            <div className="mx-2 my-2" style={{ borderTop: "1px solid var(--border)" }} />
          </>
        )}

        {noMatches && (
          <div className="px-2 pt-2 text-[11px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            No layers match &ldquo;{search}&rdquo;.
            {matchingTemplates.length > 0 && (
              <div className="mt-1.5">
                {matchingTemplates.join(", ")} {matchingTemplates.length === 1 ? "is a template" : "are templates"}.
                Open {matchingTemplates.length === 1 ? "it" : "them"} from Insert &gt; From Template.
              </div>
            )}
          </div>
        )}

        {/* ── Layers section header ── */}
        {!noMatches && (
          <div className="px-2 pt-1 pb-1.5 flex items-center justify-between">
            <h2 className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
              Layers
            </h2>
            <span className="text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>
              {layerCount}
            </span>
          </div>
        )}

        {/* Layer categories */}
        {layerCategories.map((cat) => {
          const modules = filterModules(modulesByCategory[cat] ?? []);
          if (!modules.length) return null;

          const isOpen = expanded.has(cat) || !!search;

          return (
            <CategoryGroup
              key={cat}
              category={cat}
              label={CATEGORY_LABELS[cat]}
              modules={modules}
              open={isOpen}
              onToggle={() => toggle(cat)}
              onItemMouseDown={onMouseDown}
            />
          );
        })}
      </div>
    </div>
  );
}

/** A category heading that opens to its layers. Drag a layer onto the canvas, or click it (or press Enter) to add it. */
function CategoryGroup({ category, label, modules, open, onToggle, onItemMouseDown }: {
  category: NodeCategory;
  label: string;
  modules: ModuleDef[];
  open: boolean;
  onToggle: () => void;
  onItemMouseDown: (e: React.MouseEvent, moduleType: string) => void;
}) {
  const color = CATEGORY_COLORS[category];
  return (
    <div className="mb-0.5">
      <button
        onClick={onToggle}
        aria-expanded={open}
        className="flex items-center w-full px-2 py-[7px] text-xs font-semibold rounded-lg transition-colors text-fg-secondary hover:bg-surface-raised hover:text-fg"
      >
        <svg
          width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          className="mr-1.5 flex-shrink-0 transition-transform text-fg-muted" aria-hidden="true"
          style={{ transform: open ? "rotate(90deg)" : "rotate(0deg)" }}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        <CategoryIcon category={category} size={14} className="mr-2 flex-shrink-0" style={{ color }} />
        {label}
        <span className="ml-auto text-[11px] font-normal tabular-nums text-fg-muted">{modules.length}</span>
      </button>
      {open && (
        <div className="ml-3 pl-2.5 mt-0.5 mb-1 space-y-px" style={{ borderLeft: `2px solid ${tint(color, 25)}` }}>
          {modules.map((mod) => (
            <button
              key={mod.type}
              type="button"
              onMouseDown={(e) => onItemMouseDown(e, mod.type)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") { e.preventDefault(); addAtDefaultSpot(mod.type); }
              }}
              className="flex items-center w-full px-2.5 py-[6px] rounded-md cursor-grab active:cursor-grabbing text-xs text-left transition-colors select-none text-fg hover:bg-surface-raised"
              title={`${mod.description}\nDrag onto the canvas, or click to add`}
            >
              <span className="truncate">{mod.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
