import { useEffect } from "react";

/**
 * Calls onEscape when Escape is pressed (used to close dialogs and menus).
 * Pass ignoreInTextarea for dialogs holding edited code, so Escape in the editor doesn't discard it.
 */
export function useEscapeKey(onEscape: () => void, { enabled = true, ignoreInTextarea = false } = {}) {
  useEffect(() => {
    if (!enabled) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (ignoreInTextarea && (e.target as HTMLElement)?.tagName === "TEXTAREA") return;
      onEscape();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onEscape, enabled, ignoreInTextarea]);
}
