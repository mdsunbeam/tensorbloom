import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface SelectOption {
  value: string;
  label?: string;
  /** Options with the same group are listed together under that heading */
  group?: string;
  disabled?: boolean;
}

export interface SelectProps {
  value: string;
  onChange: (value: string) => void;
  options: Array<string | SelectOption>;
  /** Shown when no option matches `value` */
  placeholder?: string;
  className?: string;
  style?: React.CSSProperties;
  disabled?: boolean;
  "aria-label"?: string;
  id?: string;
}

const MAX_LIST_HEIGHT = 280;

/**
 * Drop-down list drawn by the app, so it matches the theme on every OS (a native <select> opens an
 * OS-drawn menu). Keyboard: arrows, Home/End, Enter or Space to pick, Escape to close, letters to jump.
 */
export default function Select({ value, onChange, options, placeholder = "Select…", className = "", style, disabled, id, ...aria }: SelectProps) {
  const opts: SelectOption[] = options.map((o) => (typeof o === "string" ? { value: o } : o));
  const selected = opts.find((o) => o.value === value);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [pos, setPos] = useState<{ left: number; top: number; width: number; above: boolean } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const listId = useId();

  const enabledIndex = (from: number, step: number) => {
    for (let i = from, n = 0; n < opts.length; i += step, n++) {
      const j = (i + opts.length) % opts.length;
      if (!opts[j].disabled) return j;
    }
    return from;
  };

  const openList = () => {
    if (disabled) return;
    const i = opts.findIndex((o) => o.value === value);
    setActive(i >= 0 ? i : enabledIndex(0, 1));
    setOpen(true);
  };
  const close = (refocus = true) => {
    setOpen(false);
    if (refocus) triggerRef.current?.focus();
  };
  const pick = (i: number) => {
    const o = opts[i];
    if (!o || o.disabled) return;
    if (o.value !== value) onChange(o.value);
    close();
  };

  // Place the list under the field, or above it when there's no room below
  useLayoutEffect(() => {
    if (!open || !triggerRef.current) return;
    const r = triggerRef.current.getBoundingClientRect();
    const above = window.innerHeight - r.bottom < Math.min(MAX_LIST_HEIGHT, opts.length * 30) + 12 && r.top > window.innerHeight - r.bottom;
    setPos({ left: r.left, top: above ? r.top - 4 : r.bottom + 4, width: r.width, above });
  }, [open, opts.length]);

  // Close on outside click, or when anything scrolls or resizes (the list would drift from the field)
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!listRef.current?.contains(t) && !triggerRef.current?.contains(t)) close(false);
    };
    const onScroll = (e: Event) => { if (!listRef.current?.contains(e.target as Node)) close(false); };
    const onResize = () => close(false);
    document.addEventListener("mousedown", onDown);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onResize);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onResize);
    };
  }, [open]);

  useEffect(() => {
    if (open) listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) { e.preventDefault(); openList(); }
      return;
    }
    // Keep keys away from the app's shortcuts and from dialogs listening for Escape
    e.stopPropagation();
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => enabledIndex(a + 1, 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => enabledIndex(a - 1, -1)); }
    else if (e.key === "Home") { e.preventDefault(); setActive(enabledIndex(0, 1)); }
    else if (e.key === "End") { e.preventDefault(); setActive(enabledIndex(opts.length - 1, -1)); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(active); }
    else if (e.key === "Escape") { e.preventDefault(); close(); }
    else if (e.key === "Tab") { close(false); }
    else if (e.key.length === 1) {
      const ch = e.key.toLowerCase();
      const start = active + 1;
      for (let n = 0; n < opts.length; n++) {
        const j = (start + n) % opts.length;
        if (!opts[j].disabled && (opts[j].label ?? opts[j].value).toLowerCase().startsWith(ch)) { setActive(j); break; }
      }
    }
  };

  let lastGroup: string | undefined;

  return (
    <>
      <button
        ref={triggerRef}
        id={id}
        type="button"
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-activedescendant={open ? `${listId}-${active}` : undefined}
        aria-label={aria["aria-label"]}
        disabled={disabled}
        onClick={() => (open ? close() : openList())}
        onKeyDown={onKeyDown}
        className={`flex items-center justify-between gap-2 text-left disabled:opacity-50 ${className}`}
        style={style}
      >
        <span className={`truncate ${selected ? "" : "text-fg-muted"}`}>{selected ? selected.label ?? selected.value : placeholder}</span>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          className="flex-shrink-0 text-fg-muted" aria-hidden="true" style={{ transform: open ? "rotate(180deg)" : undefined }}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {open && pos && createPortal(
        <div
          ref={listRef}
          id={listId}
          role="listbox"
          className="fixed z-[70] py-1 rounded-lg overflow-y-auto bg-surface border border-line text-xs"
          style={{
            left: pos.left,
            top: pos.above ? undefined : pos.top,
            bottom: pos.above ? window.innerHeight - pos.top : undefined,
            minWidth: pos.width,
            maxHeight: MAX_LIST_HEIGHT,
            boxShadow: "0 8px 24px rgba(0,0,0,0.25), 0 2px 6px rgba(0,0,0,0.15)",
          }}
        >
          {opts.map((o, i) => {
            const heading = o.group && o.group !== lastGroup ? o.group : null;
            lastGroup = o.group;
            const isSelected = o.value === value;
            return (
              <div key={`${o.group ?? ""}:${o.value}`}>
                {heading && <div className="px-3 pt-2 pb-1 text-[11px] font-semibold uppercase tracking-wider text-fg-muted">{heading}</div>}
                <div
                  id={`${listId}-${i}`}
                  data-index={i}
                  role="option"
                  aria-selected={isSelected}
                  aria-disabled={o.disabled || undefined}
                  onMouseEnter={() => !o.disabled && setActive(i)}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => pick(i)}
                  className={`flex items-center justify-between gap-3 px-3 py-1.5 cursor-default ${
                    o.disabled ? "text-fg-muted opacity-60" : i === active ? "bg-surface-raised text-fg" : "text-fg-secondary"
                  }`}
                >
                  <span className="truncate">{o.label ?? o.value}</span>
                  {isSelected && (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent-text)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  )}
                </div>
              </div>
            );
          })}
        </div>,
        document.body,
      )}
    </>
  );
}
