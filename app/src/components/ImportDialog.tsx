import { useState } from "react";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";

interface ImportDialogProps {
  open: boolean;
  onClose: () => void;
  /** Resolves to false when the import was called off (e.g. the user cancelled at the unsaved-changes prompt) */
  onImport: (code: string) => Promise<boolean>;
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

  const handleImport = async () => {
    const trimmed = code.trim();
    if (!trimmed) {
      setError("Paste some PyTorch code first");
      return;
    }
    setError("");
    setLoading(true);
    try {
      if (!(await onImport(trimmed))) return;
      setCode("");
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      width={600}
      title="Import PyTorch code"
      description="Paste a model definition. It's traced with torch.fx and turned into a graph."
      ignoreEscapeInTextarea={code.trim() !== ""}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={handleImport} disabled={loading}>{loading ? "Importing…" : "Import"}</Button>
        </>
      }
    >
      <div className="p-4">
        <textarea
          value={code}
          onChange={(e) => { setCode(e.target.value); setError(""); }}
          placeholder={PLACEHOLDER}
          spellCheck={false}
          aria-label="PyTorch code"
          className="w-full h-[300px] p-3 text-xs font-mono leading-relaxed rounded-lg resize-none outline-none bg-surface-raised border border-line text-fg focus:border-accent"
        />
        {error && (
          <div role="alert" className="mt-2 px-3 py-2 rounded-lg text-xs bg-danger-subtle text-danger">
            {error}
          </div>
        )}
      </div>
    </Dialog>
  );
}
