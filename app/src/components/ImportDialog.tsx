import { useState } from "react";

interface ImportDialogProps {
  open: boolean;
  onClose: () => void;
  onImport: (code: string) => void;
}

const PLACEHOLDER = `import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyModel()`;

export default function ImportDialog({ open, onClose, onImport }: ImportDialogProps) {
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  if (!open) return null;

  const handleImport = async () => {
    const trimmed = code.trim();
    if (!trimmed) {
      setError("Paste some PyTorch code first");
      return;
    }
    setError("");
    setLoading(true);
    try {
      await onImport(trimmed);
      setCode("");
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50"
        style={{ background: "rgba(0,0,0,0.5)" }}
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] max-h-[80vh] rounded-xl shadow-2xl flex flex-col overflow-hidden"
        style={{
          background: "var(--bg-secondary)",
          border: "1px solid var(--border)",
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-3"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div>
            <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Import PyTorch Code
            </div>
            <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
              Paste a model definition — it will be traced via torch.fx
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded-md transition-colors"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-tertiary)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Code input */}
        <div className="flex-1 p-4 min-h-0">
          <textarea
            value={code}
            onChange={(e) => { setCode(e.target.value); setError(""); }}
            placeholder={PLACEHOLDER}
            spellCheck={false}
            className="w-full h-[300px] p-3 text-xs font-mono leading-relaxed rounded-lg resize-none"
            style={{
              background: "var(--bg-tertiary)",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
              outline: "none",
            }}
            onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
            onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
          />
        </div>

        {/* Error */}
        {error && (
          <div
            className="mx-4 mb-2 px-3 py-2 rounded-lg text-xs"
            style={{
              background: "var(--danger-subtle)",
              color: "var(--danger)",
            }}
          >
            {error}
          </div>
        )}

        {/* Footer */}
        <div
          className="flex items-center justify-end gap-2 px-5 py-3"
          style={{ borderTop: "1px solid var(--border)" }}
        >
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-medium rounded-lg transition-colors"
            style={{
              background: "var(--bg-tertiary)",
              color: "var(--text-secondary)",
              border: "1px solid var(--border)",
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleImport}
            disabled={loading}
            className="px-4 py-1.5 text-xs font-medium rounded-lg text-white transition-colors"
            style={{
              background: loading ? "var(--text-muted)" : "var(--accent)",
              border: "1px solid transparent",
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? "Importing..." : "Import"}
          </button>
        </div>
      </div>
    </>
  );
}
