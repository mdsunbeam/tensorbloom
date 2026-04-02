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

const modulesByCategory = getModulesByCategory();
const layerCategories = (Object.keys(modulesByCategory) as NodeCategory[]).filter((c) => c !== "data" && c !== "io");
const dataModules = modulesByCategory["data"] ?? [];

export default function NodePalette() {
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

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
        const { addNode, nodes } = useGraphStore.getState();
        const snap = (v: number) => Math.round(v / 20) * 20;
        addNode(moduleType, {
          x: snap(260),
          y: snap(40 + nodes.length * 100),
        });
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
  const dataColor = CATEGORY_COLORS["data"];
  const dataOpen = expanded.has("data") || !!search;

  // Count only layer modules (exclude data)
  const layerCount = layerCategories.reduce(
    (acc, k) => acc + (modulesByCategory[k]?.length ?? 0),
    0,
  );

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
            <div className="mb-0.5">
              <button
                onClick={() => toggle("data")}
                className="flex items-center w-full px-2 py-[7px] text-[11px] font-semibold rounded-lg transition-colors"
                style={{ color: "var(--text-secondary)" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-tertiary)")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <svg
                  width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
                  className="mr-1.5 flex-shrink-0 transition-transform"
                  style={{ transform: dataOpen ? "rotate(90deg)" : "rotate(0deg)" }}
                >
                  <polyline points="9 18 15 12 9 6" />
                </svg>
                <span className="w-2.5 h-2.5 rounded flex-shrink-0 mr-2" style={{ background: dataColor }} />
                Datasets
                <span className="ml-auto text-[10px] font-normal tabular-nums" style={{ color: "var(--text-muted)" }}>
                  {filteredDataModules.length}
                </span>
              </button>
              {dataOpen && (
                <div className="ml-3 pl-2.5 mt-0.5 mb-1 space-y-px" style={{ borderLeft: `2px solid ${dataColor}20` }}>
                  {filteredDataModules.map((mod) => (
                    <div
                      key={mod.type}
                      onMouseDown={(e) => onMouseDown(e, mod.type)}
                      className="flex items-center px-2.5 py-[6px] rounded-md cursor-grab active:cursor-grabbing text-xs transition-colors select-none"
                      style={{ color: "var(--text-primary)", background: "transparent" }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-tertiary)")}
                      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                      title={mod.description}
                    >
                      <span className="w-[5px] h-[5px] rounded-full mr-2.5 flex-shrink-0 opacity-60" style={{ background: dataColor }} />
                      <span className="truncate">{mod.label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Divider */}
            <div className="mx-2 my-2" style={{ borderTop: "1px solid var(--border)" }} />
          </>
        )}

        {/* ── Layers section header ── */}
        <div className="px-2 pt-1 pb-1.5 flex items-center justify-between">
          <h2 className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
            Layers
          </h2>
          <span className="text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>
            {layerCount}
          </span>
        </div>

        {/* Layer categories */}
        {layerCategories.map((cat) => {
          const modules = filterModules(modulesByCategory[cat] ?? []);
          if (!modules.length) return null;

          const isOpen = expanded.has(cat) || !!search;
          const color = CATEGORY_COLORS[cat];

          return (
            <div key={cat} className="mb-0.5">
              {/* Category header */}
              <button
                onClick={() => toggle(cat)}
                className="flex items-center w-full px-2 py-[7px] text-[11px] font-semibold rounded-lg transition-colors"
                style={{ color: "var(--text-secondary)" }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.background = "var(--bg-tertiary)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.background = "transparent")
                }
              >
                <svg
                  width="10"
                  height="10"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="mr-1.5 flex-shrink-0 transition-transform"
                  style={{
                    transform: isOpen ? "rotate(90deg)" : "rotate(0deg)",
                  }}
                >
                  <polyline points="9 18 15 12 9 6" />
                </svg>
                <span
                  className="w-2.5 h-2.5 rounded flex-shrink-0 mr-2"
                  style={{ background: color }}
                />
                {CATEGORY_LABELS[cat]}
                <span
                  className="ml-auto text-[10px] font-normal tabular-nums"
                  style={{ color: "var(--text-muted)" }}
                >
                  {modules.length}
                </span>
              </button>

              {/* Module items */}
              {isOpen && (
                <div className="ml-3 pl-2.5 mt-0.5 mb-1 space-y-px" style={{ borderLeft: `2px solid ${color}20` }}>
                  {modules.map((mod) => (
                    <div
                      key={mod.type}
                      onMouseDown={(e) => onMouseDown(e, mod.type)}
                      className="flex items-center px-2.5 py-[6px] rounded-md cursor-grab active:cursor-grabbing text-xs transition-colors select-none"
                      style={{
                        color: "var(--text-primary)",
                        background: "transparent",
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.background = "var(--bg-tertiary)")
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.background = "transparent")
                      }
                      title={mod.description}
                    >
                      <span
                        className="w-[5px] h-[5px] rounded-full mr-2.5 flex-shrink-0 opacity-60"
                        style={{ background: color }}
                      />
                      <span className="truncate">{mod.label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
