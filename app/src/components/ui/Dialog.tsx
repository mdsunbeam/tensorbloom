import { useEffect, useId, useRef } from "react";
import { useEscapeKey } from "../../lib/use-escape-key";
import { IconButton } from "./Button";

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

// Dialogs that are open, oldest first. Only the newest one answers Escape, so closing a
// confirmation on top of another dialog doesn't close both.
const openDialogs: string[] = [];

export interface DialogProps {
  open: boolean;
  onClose: () => void;
  title: React.ReactNode;
  description?: React.ReactNode;
  /** Panel width in px */
  width?: number;
  /** Buttons for the bottom bar, right-aligned */
  footer?: React.ReactNode;
  /** Ignore Escape inside a textarea, for dialogs where people edit code */
  ignoreEscapeInTextarea?: boolean;
  /** Element to focus when the dialog opens; defaults to the first focusable element in the body */
  initialFocus?: React.RefObject<HTMLElement | null>;
  /** "alertdialog" for a question that needs an answer before going on */
  role?: "dialog" | "alertdialog";
  /** Draw above other dialogs, for a question asked while one is open */
  onTop?: boolean;
  className?: string;
  children: React.ReactNode;
}

/**
 * Modal dialog frame: overlay, title bar with a close button, body and footer.
 * Escape closes it, Tab stays inside it, and focus goes back where it was when it closes.
 */
export default function Dialog({
  open, onClose, title, description, width = 440, footer, ignoreEscapeInTextarea = false, initialFocus, role = "dialog", onTop = false, className = "", children,
}: DialogProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const descId = useId();

  useEffect(() => {
    if (!open) return;
    openDialogs.push(titleId);
    return () => { openDialogs.splice(openDialogs.indexOf(titleId), 1); };
  }, [open, titleId]);

  useEscapeKey(() => { if (openDialogs[openDialogs.length - 1] === titleId) onClose(); }, { enabled: open, ignoreInTextarea: ignoreEscapeInTextarea });

  useEffect(() => {
    if (!open) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const target = initialFocus?.current ?? bodyRef.current?.querySelector<HTMLElement>(FOCUSABLE) ?? panelRef.current;
    target?.focus();
    return () => { previouslyFocused?.focus?.(); };
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps -- only on open and close

  if (!open) return null;

  const trapTab = (e: React.KeyboardEvent) => {
    if (e.key !== "Tab" || !panelRef.current) return;
    const items = Array.from(panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => el.offsetParent !== null);
    if (items.length === 0) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };

  return (
    <>
      <div className={`fixed inset-0 ${onTop ? "z-[60]" : "z-50"}`} style={{ background: "rgba(0,0,0,0.5)" }} onClick={onClose} />
      <div
        ref={panelRef}
        role={role}
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descId : undefined}
        tabIndex={-1}
        onKeyDown={trapTab}
        className={`fixed ${onTop ? "z-[60]" : "z-50"} top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 max-w-[calc(100vw-32px)] max-h-[85vh] rounded-xl shadow-2xl flex flex-col overflow-hidden outline-none bg-surface border border-line ${className}`}
        style={{ width }}
      >
        <div className="flex items-start justify-between gap-3 px-5 py-3 flex-shrink-0 border-b border-line">
          <div className="min-w-0">
            <h2 id={titleId} className="text-sm font-semibold text-fg">{title}</h2>
            {description && <p id={descId} className="text-[11px] mt-0.5 text-fg-muted">{description}</p>}
          </div>
          <IconButton label="Close" onClick={onClose} className="-mr-1.5 flex-shrink-0">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </IconButton>
        </div>
        <div ref={bodyRef} className="flex-1 min-h-0 overflow-y-auto">
          {children}
        </div>
        {footer && (
          <div className="flex items-center justify-end gap-2 px-5 py-3 flex-shrink-0 border-t border-line">
            {footer}
          </div>
        )}
      </div>
    </>
  );
}
