import { useEffect, useRef, useState } from "react";

type NumberFieldProps = Omit<React.InputHTMLAttributes<HTMLInputElement>, "type" | "value" | "onChange"> & {
  value: unknown;
  onChange: (v: number) => void;
  integer?: boolean;
};

/**
 * Number input that can be cleared and retyped. The typed text lives locally and only parseable
 * values are reported; on blur the field shows the last committed value again.
 */
export default function NumberField({ value, onChange, integer, onFocus, onBlur, ...rest }: NumberFieldProps) {
  const shown = value == null ? "" : String(value);
  const [draft, setDraft] = useState(shown);
  const focused = useRef(false);

  // Follow outside changes (auto-fix, undo, loading a template) unless the user is mid-edit
  useEffect(() => {
    if (!focused.current) setDraft(shown);
  }, [shown]);

  return (
    <input
      {...rest}
      type="number"
      value={draft}
      onFocus={(e) => { focused.current = true; onFocus?.(e); }}
      onBlur={(e) => { focused.current = false; setDraft(shown); onBlur?.(e); }}
      onChange={(e) => {
        setDraft(e.target.value);
        const v = integer ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
        if (!isNaN(v)) onChange(v);
      }}
    />
  );
}
