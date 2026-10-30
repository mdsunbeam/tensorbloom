import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

/**
 * A small "?" that explains the field next to it on hover or keyboard focus.
 * Give the field aria-describedby={id} so screen readers read the same text.
 */
export default function HelpTip({ id, text, label = "What is this?" }: { id: string; text: string; label?: string }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const ref = useRef<HTMLButtonElement>(null);

  // Place the bubble under the icon, kept inside the window
  useLayoutEffect(() => {
    if (!open || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const width = 260;
    setPos({ left: Math.max(8, Math.min(window.innerWidth - width - 8, r.left + r.width / 2 - width / 2)), top: r.bottom + 6 });
  }, [open]);

  return (
    <>
      <button
        ref={ref}
        type="button"
        aria-label={label}
        aria-describedby={id}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}
        className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full text-fg-muted hover:text-fg focus-visible:text-fg align-middle"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="10" /><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
      </button>
      {/* Always in the page, so aria-describedby works while the bubble is closed */}
      <span id={id} hidden>{text}</span>
      {open && pos && createPortal(
        <div
          role="tooltip"
          className="fixed z-[200] w-[260px] px-3 py-2 rounded-lg text-[11px] leading-relaxed font-normal normal-case tracking-normal bg-surface-elevated text-fg border border-line pointer-events-none"
          style={{ left: pos.left, top: pos.top, boxShadow: "0 8px 24px rgba(0,0,0,0.25)" }}
        >
          {text}
        </div>,
        document.body,
      )}
    </>
  );
}
