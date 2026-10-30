import { useRef } from "react";

export interface TabItem<T extends string> {
  id: T;
  label: React.ReactNode;
  /** Shown after the label in a quieter color, e.g. a count */
  badge?: React.ReactNode;
  /** Draws attention to the tab, e.g. while training is running */
  live?: boolean;
}

/** Underlined tab bar. Arrow keys move between tabs. */
export default function Tabs<T extends string>({ items, value, onChange, label, className = "", stretch = false }: {
  items: TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  /** Accessible name for the tab list */
  label: string;
  className?: string;
  /** Share the width equally between tabs */
  stretch?: boolean;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (e: React.KeyboardEvent, i: number) => {
    const next = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? items.length - 1 : null;
    if (next === null) return;
    e.preventDefault();
    const j = (next + items.length) % items.length;
    onChange(items[j].id);
    refs.current[j]?.focus();
  };

  return (
    <div role="tablist" aria-label={label} className={`flex items-stretch ${className}`}>
      {items.map((t, i) => {
        const active = t.id === value;
        return (
          <button
            key={t.id}
            ref={(el) => { refs.current[i] = el; }}
            role="tab"
            aria-selected={active}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(t.id)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={`relative flex items-center justify-center gap-1.5 px-3 text-xs font-medium transition-colors ${stretch ? "flex-1" : ""} ${
              active ? "text-fg" : "text-fg-muted hover:text-fg-secondary"
            }`}
          >
            {t.live && <><span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" aria-hidden="true" /><span className="sr-only">(live)</span></>}
            {t.label}
            {t.badge != null && <span className="tabular-nums text-fg-muted font-normal">{t.badge}</span>}
            <span className={`absolute left-2 right-2 bottom-0 h-0.5 rounded-full ${active ? "bg-accent" : "bg-transparent"}`} aria-hidden="true" />
          </button>
        );
      })}
    </div>
  );
}
