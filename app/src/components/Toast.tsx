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

function ToastItem({ toast: t, onDismiss }: { toast: ToastItem; onDismiss: () => void }) {
  const colors = {
    info: { bg: "var(--bg-secondary)", border: "var(--accent)", icon: "var(--accent)" },
    success: { bg: "var(--bg-secondary)", border: "#10b981", icon: "#10b981" },
    error: { bg: "var(--bg-secondary)", border: "var(--danger)", icon: "var(--danger)" },
    warning: { bg: "var(--bg-secondary)", border: "#f59e0b", icon: "#f59e0b" },
  };
  const c = colors[t.type];

  return (
    <div
      className="flex items-center gap-2.5 px-4 py-2.5 rounded-lg shadow-lg pointer-events-auto animate-in"
      style={{
        background: c.bg,
        border: `1px solid ${c.border}`,
        minWidth: 220,
        maxWidth: 360,
      }}
      onClick={onDismiss}
    >
      <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: c.icon }} />
      <span className="text-xs" style={{ color: "var(--text-primary)" }}>{t.message}</span>
    </div>
  );
}
