import Dialog from "./ui/Dialog";

interface ShortcutsDialogProps {
  open: boolean;
  onClose: () => void;
}

const SHORTCUTS = [
  { section: "File" },
  { key: "Ctrl+N", action: "New project" },
  { key: "Ctrl+O", action: "Open project" },
  { key: "Ctrl+S", action: "Save project" },
  { key: "Ctrl+Shift+S", action: "Save as" },
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

  { section: "Training" },
  { key: "Ctrl+Enter", action: "Start training" },

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
  return (
    <Dialog open={open} onClose={onClose} title="Keyboard shortcuts" width={420}>
      <div className="p-4 space-y-1">
        {SHORTCUTS.map((item, i) =>
          "section" in item && item.section ? (
            <h3 key={i} className="text-[11px] font-semibold uppercase tracking-wider pt-3 pb-1 first:pt-0 text-fg-muted">
              {item.section}
            </h3>
          ) : (
            <div key={i} className="flex items-center justify-between py-1">
              <span className="text-xs text-fg">{"action" in item ? item.action : ""}</span>
              <kbd className="px-2 py-0.5 text-[11px] font-mono rounded bg-surface-raised border border-line text-fg-secondary">
                {"key" in item ? item.key : ""}
              </kbd>
            </div>
          ),
        )}
      </div>
    </Dialog>
  );
}
