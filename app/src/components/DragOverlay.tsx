import { useState, useEffect } from "react";
import { CATEGORY_COLORS } from "../lib/registry";
import type { NodeCategory } from "../lib/types";
import { CategoryBadge } from "../lib/category-icons";

interface DragState {
  active: boolean;
  moduleType: string;
  label: string;
  category: NodeCategory;
  x: number;
  y: number;
}

// Global drag state — shared between NodePalette and GraphEditor
let _dragState: DragState | null = null;
let _listeners: Set<() => void> = new Set();

export function startDrag(moduleType: string, label: string, category: NodeCategory, x: number, y: number) {
  _dragState = { active: true, moduleType, label, category, x, y };
  _listeners.forEach((l) => l());
}

export function updateDrag(x: number, y: number) {
  if (_dragState) {
    _dragState = { ..._dragState, x, y };
    _listeners.forEach((l) => l());
  }
}

export function endDrag(): DragState | null {
  const state = _dragState;
  _dragState = null;
  _listeners.forEach((l) => l());
  return state;
}

export function getDragState(): DragState | null {
  return _dragState;
}

export function useDragState(): DragState | null {
  const [state, setState] = useState<DragState | null>(null);
  useEffect(() => {
    const listener = () => setState(_dragState ? { ..._dragState } : null);
    _listeners.add(listener);
    return () => { _listeners.delete(listener); };
  }, []);
  return state;
}

export default function DragOverlay() {
  const drag = useDragState();

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (_dragState) updateDrag(e.clientX, e.clientY);
    };
    const onMouseUp = () => {
      // Drop is handled by the GraphEditor's onMouseUp
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  if (!drag || !drag.active) return null;

  const color = CATEGORY_COLORS[drag.category];

  return (
    <div
      className="fixed pointer-events-none z-[200]"
      style={{
        left: drag.x - 85,
        top: drag.y - 20,
      }}
    >
      <div
        className="min-w-[170px] rounded-lg shadow-2xl opacity-90"
        style={{
          background: "var(--bg-secondary)",
          border: `1.5px solid ${color}`,
        }}
      >
        <div className="h-[3px] rounded-t-lg" style={{ background: color }} />
        <div className="flex items-center gap-2 px-3 py-2 text-xs font-semibold">
          <CategoryBadge category={drag.category} color={color} size={18} />
          <span style={{ color: "var(--text-primary)" }}>{drag.label}</span>
        </div>
      </div>
    </div>
  );
}
