import { create } from "zustand";

interface ToastItem {
  id: number;
  message: string;
  type: "info" | "success" | "error" | "warning";
  duration: number;
}

interface ToastState {
  toasts: ToastItem[];
  add: (message: string, type?: ToastItem["type"], duration?: number) => void;
  remove: (id: number) => void;
}

let _toastId = 0;

export const useToastStore = create<ToastState>((set, get) => ({
  toasts: [],
  add: (message, type = "info", duration = 3000) => {
    const id = ++_toastId;
    set({ toasts: [...get().toasts, { id, message, type, duration }] });
    setTimeout(() => get().remove(id), duration);
  },
  remove: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
}));

// Convenience exports
export const toast = {
  info: (msg: string) => useToastStore.getState().add(msg, "info"),
  success: (msg: string) => useToastStore.getState().add(msg, "success"),
  error: (msg: string) => useToastStore.getState().add(msg, "error", 5000),
  warning: (msg: string) => useToastStore.getState().add(msg, "warning", 4000),
};

/** rightOffset keeps toasts over the canvas, clear of the right panel (px from the window edge). */
export default function ToastContainer({ rightOffset = 304 }: { rightOffset?: number }) {
  const toasts = useToastStore((s) => s.toasts);
  const remove = useToastStore((s) => s.remove);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 z-50 flex flex-col gap-2 pointer-events-none" style={{ right: rightOffset }}>
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={() => remove(t.id)} />
      ))}
    </div>
  );
}

const TOAST_COLORS: Record<ToastItem["type"], string> = {
  info: "var(--accent)",
  success: "var(--success)",
  error: "var(--danger)",
  warning: "var(--warning)",
};

/** A distinct shape per kind, so the kind doesn't rest on color alone */
function ToastIcon({ type }: { type: ToastItem["type"] }) {
  const common = { width: 14, height: 14, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2.2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true, className: "flex-shrink-0" };
  if (type === "success") return <svg {...common}><polyline points="20 6 9 17 4 12" /></svg>;
  if (type === "error") return <svg {...common}><circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" /></svg>;
  if (type === "warning") return <svg {...common}><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>;
  return <svg {...common}><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></svg>;
}

function ToastItem({ toast: t, onDismiss }: { toast: ToastItem; onDismiss: () => void }) {
  const color = TOAST_COLORS[t.type];
  return (
    <div
      role={t.type === "error" ? "alert" : "status"}
      className="flex items-center gap-2.5 px-4 py-2.5 rounded-lg shadow-lg pointer-events-auto animate-in"
      style={{
        background: "var(--bg-secondary)",
        border: `1px solid ${color}`,
        minWidth: 220,
        maxWidth: 360,
      }}
      onClick={onDismiss}
    >
      <span style={{ color }}><ToastIcon type={t.type} /></span>
      <span className="text-xs" style={{ color: "var(--text-primary)" }}>{t.message}</span>
    </div>
  );
}
