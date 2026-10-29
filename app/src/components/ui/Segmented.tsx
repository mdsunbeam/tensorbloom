import { useRef } from "react";

export interface SegmentOption<T extends string> {
  value: T;
  label: React.ReactNode;
  disabled?: boolean;
  /** Tooltip, e.g. why an option is unavailable */
  title?: string;
}

/** A row of mutually exclusive choices (a radio group drawn as buttons). Arrow keys move the choice. */
export default function Segmented<T extends string>({ value, onChange, options, label, size = "md", className = "" }: {
  value: T;
  onChange: (value: T) => void;
  options: SegmentOption<T>[];
  /** Accessible name for the group */
  label: string;
  size?: "sm" | "md";
  className?: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (e: React.KeyboardEvent, i: number) => {
    const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    for (let n = 1; n <= options.length; n++) {
      const j = (i + step * n + options.length) % options.length;
      if (!options[j].disabled) {
        onChange(options[j].value);
        refs.current[j]?.focus();
        return;
      }
    }
  };

  return (
    <div role="radiogroup" aria-label={label} className={`flex gap-2 ${className}`}>
      {options.map((o, i) => {
        const checked = o.value === value;
        return (
          <button
            key={o.value}
            ref={(el) => { refs.current[i] = el; }}
            type="button"
            role="radio"
            aria-checked={checked}
            tabIndex={checked ? 0 : -1}
            disabled={o.disabled}
            title={o.title}
            onClick={() => onChange(o.value)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={`flex-1 ${size === "sm" ? "h-7 text-[11px]" : "h-8 text-xs"} px-2 font-medium rounded-lg border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              checked ? "bg-accent-subtle border-accent text-accent-text" : "bg-transparent border-line text-fg-secondary hover:bg-surface-raised hover:text-fg"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
