import { useEscapeKey } from "../lib/use-escape-key";

interface ShortcutsDialogProps {
  open: boolean;
  onClose: () => void;
}

const SHORTCUTS = [
  { section: "File" },
  { key: "Ctrl+N", action: "New project" },
  { key: "Ctrl+O", action: "Open project" },
  { key: "Ctrl+S", action: "Save project" },
  { key: "Ctrl+E", action: "Export model (ONNX/TorchScript)" },

  { section: "Edit" },
  { key: "Ctrl+Z", action: "Undo" },
  { key: "Ctrl+Shift+Z", action: "Redo (also Ctrl+Y)" },
  { key: "Ctrl+A", action: "Select all nodes" },
  { key: "Ctrl+C", action: "Copy selected nodes" },
  { key: "Ctrl+V", action: "Paste copied nodes" },
  { key: "Delete", action: "Delete selected nodes" },
  { key: "Ctrl+D", action: "Duplicate selected nodes" },
  { key: "Double-click", action: "Rename node" },
  { key: "Right-click", action: "Context menu" },

  { section: "Tools" },
  { key: "Ctrl+K", action: "Spotlight — quick add layer" },
  { key: "Ctrl+I", action: "Import PyTorch code" },
  { key: "Ctrl+L", action: "Auto-layout graph" },

  { section: "View" },
  { key: "Ctrl+B", action: "Toggle left panel" },
  { key: "Ctrl+.", action: "Toggle right panel" },
  { key: "Ctrl+M", action: "Toggle minimap" },
  { key: "Ctrl+?", action: "Keyboard shortcuts" },
  { key: "Scroll", action: "Pan canvas" },
  { key: "Ctrl+Scroll", action: "Zoom in/out" },
  { key: "Drag empty canvas", action: "Pan canvas" },
];

export default function ShortcutsDialog({ open, onClose }: ShortcutsDialogProps) {
  useEscapeKey(onClose, { enabled: open });
  if (!open) return null;

  return (
    <>
      <div className="fixed inset-0 z-50" style={{ background: "rgba(0,0,0,0.4)" }} onClick={onClose} />
      <div
        className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[420px] max-h-[70vh] rounded-xl shadow-2xl flex flex-col overflow-hidden"
        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}
      >
        <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Keyboard Shortcuts
          </div>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-md"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-tertiary)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-1">
          {SHORTCUTS.map((item, i) =>
            "section" in item && item.section ? (
              <div
                key={i}
                className="text-[10px] font-semibold uppercase tracking-wider pt-3 pb-1 first:pt-0"
                style={{ color: "var(--text-muted)" }}
              >
                {item.section}
              </div>
            ) : (
              <div key={i} className="flex items-center justify-between py-1">
                <span className="text-xs" style={{ color: "var(--text-primary)" }}>
                  {"action" in item ? item.action : ""}
                </span>
                <kbd
                  className="px-2 py-0.5 text-[10px] font-mono rounded"
                  style={{
                    background: "var(--bg-tertiary)",
                    border: "1px solid var(--border)",
                    color: "var(--text-secondary)",
                  }}
                >
                  {"key" in item ? item.key : ""}
                </kbd>
              </div>
            ),
          )}
        </div>
      </div>
    </>
  );
}
