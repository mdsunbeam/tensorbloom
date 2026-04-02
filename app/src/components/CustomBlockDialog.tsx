import { useState } from "react";
import { useGraphStore } from "../store/graph-store";
import { MODULE_REGISTRY } from "../lib/registry";
import type { ModuleDef, NodeCategory } from "../lib/types";
import { toast } from "./Toast";

interface CustomBlockDialogProps {
  open: boolean;
  onClose: () => void;
}

const TEMPLATE = `# Define a custom PyTorch module.
# It will appear as a draggable block in the palette.

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU block"""
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))`;

export default function CustomBlockDialog({ open, onClose }: CustomBlockDialogProps) {
  const [code, setCode] = useState(TEMPLATE);
  const [name, setName] = useState("ConvBlock");
  const [category, setCategory] = useState<NodeCategory>("conv");
  const [params, setParams] = useState("in_channels:int=64, out_channels:int=64, kernel_size:int=3");

  if (!open) return null;

  const handleCreate = () => {
    if (!name.trim()) { toast.error("Enter a block name"); return; }
    if (MODULE_REGISTRY[name]) { toast.error(`"${name}" already exists in the registry`); return; }

    // Parse params string into ParamDef array
    const paramDefs = params.split(",").map((p) => p.trim()).filter(Boolean).map((p) => {
      const match = p.match(/^(\w+):(\w+)=(.+)$/);
      if (!match) return null;
      const [, pName, pType, pDefault] = match;
      return {
        name: pName,
        type: pType === "float" ? "float" as const : "int" as const,
        default: pType === "float" ? parseFloat(pDefault) : parseInt(pDefault, 10),
      };
    }).filter(Boolean) as ModuleDef["params"];

    // Register the custom module
    const def: ModuleDef = {
      type: name,
      label: name,
      category,
      description: `Custom block: ${name}`,
      params: paramDefs,
      inputHandles: [{ id: "in", label: "in", dataType: "tensor" }],
      outputHandles: [{ id: "out", label: "out", dataType: "tensor" }],
    };

    MODULE_REGISTRY[name] = def;

    // Store the code so the sidecar can use it
    const existing = localStorage.getItem("tensorbloom-custom-blocks") ?? "{}";
    const blocks = JSON.parse(existing);
    blocks[name] = { code, params: paramDefs, category };
    localStorage.setItem("tensorbloom-custom-blocks", JSON.stringify(blocks));

    toast.success(`Custom block "${name}" created`);
    onClose();

    // Add an instance to the graph
    useGraphStore.getState().addNode(name, { x: 300, y: 200 });
  };

  return (
    <>
      <div className="fixed inset-0 z-50" style={{ background: "rgba(0,0,0,0.5)" }} onClick={onClose} />
      <div
        className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[620px] max-h-[80vh] rounded-xl shadow-2xl flex flex-col overflow-hidden"
        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}
      >
        <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <div>
            <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Create Custom Block</div>
            <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
              Write a PyTorch nn.Module that becomes a reusable node
            </div>
          </div>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-md"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-tertiary)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {/* Name + Category */}
          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-[11px] font-medium mb-1" style={{ color: "var(--text-secondary)" }}>Block Name</label>
              <input type="text" value={name} onChange={(e) => setName(e.target.value)}
                className="w-full px-2.5 py-[6px] text-xs rounded-lg"
                style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
            </div>
            <div className="w-36">
              <label className="block text-[11px] font-medium mb-1" style={{ color: "var(--text-secondary)" }}>Category</label>
              <select value={category} onChange={(e) => setCategory(e.target.value as NodeCategory)}
                className="w-full px-2.5 py-[6px] text-xs rounded-lg"
                style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                {["conv", "linear", "norm", "activation", "pool", "transformer", "recurrent", "dropout", "reshape", "math", "vision"].map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Parameters */}
          <div>
            <label className="block text-[11px] font-medium mb-1" style={{ color: "var(--text-secondary)" }}>
              Parameters (name:type=default, comma-separated)
            </label>
            <input type="text" value={params} onChange={(e) => setParams(e.target.value)}
              className="w-full px-2.5 py-[6px] text-xs font-mono rounded-lg"
              style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
              placeholder="in_channels:int=64, out_channels:int=64" />
          </div>

          {/* Code */}
          <div>
            <label className="block text-[11px] font-medium mb-1" style={{ color: "var(--text-secondary)" }}>
              PyTorch Code
            </label>
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              spellCheck={false}
              className="w-full h-[240px] p-3 text-[11px] font-mono leading-relaxed rounded-lg resize-none"
              style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-primary)", outline: "none" }}
            />
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-3" style={{ borderTop: "1px solid var(--border)" }}>
          <button onClick={onClose} className="px-4 py-1.5 text-xs font-medium rounded-lg"
            style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}>
            Cancel
          </button>
          <button onClick={handleCreate}
            className="px-4 py-1.5 text-xs font-medium rounded-lg text-white"
            style={{ background: "var(--accent)" }}>
            Create Block
          </button>
        </div>
      </div>
    </>
  );
}
