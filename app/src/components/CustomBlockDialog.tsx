import { useState } from "react";
import { useGraphStore } from "../store/graph-store";
import { MODULE_REGISTRY } from "../lib/registry";
import type { ModuleDef, NodeCategory } from "../lib/types";
import { customBlockDef, saveCustomBlock } from "../lib/custom-blocks";
import { toast } from "./Toast";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";
import Select from "./ui/Select";

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

    // Register the custom module and persist its code so training/export can build it
    MODULE_REGISTRY[name] = customBlockDef(name, paramDefs, category);
    saveCustomBlock(name, { code, params: paramDefs, category });

    toast.success(`Custom block "${name}" created`);
    onClose();

    // Add an instance to the graph
    useGraphStore.getState().addNode(name, { x: 300, y: 200 });
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      width={620}
      title="Create custom block"
      description="Write a PyTorch nn.Module that becomes a reusable layer in the palette"
      ignoreEscapeInTextarea={code !== TEMPLATE}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={handleCreate}>Create block</Button>
        </>
      }
    >
      <div className="p-5 space-y-4">
        <div className="flex gap-3">
          <div className="flex-1">
            <label htmlFor="custom-block-name" className="block text-[11px] font-medium mb-1 text-fg-secondary">Block name</label>
            <input id="custom-block-name" type="text" value={name} onChange={(e) => setName(e.target.value)} className="w-full px-2.5 h-8 text-xs rounded-lg bg-surface-raised border border-line text-fg outline-none focus:border-accent" />
          </div>
          <div className="w-36">
            <div className="block text-[11px] font-medium mb-1 text-fg-secondary">Category</div>
            <Select value={category} onChange={(v) => setCategory(v as NodeCategory)} aria-label="Category"
              options={["conv", "linear", "norm", "activation", "pool", "transformer", "recurrent", "dropout", "reshape", "math", "vision"]}
              className="w-full px-2.5 h-8 text-xs rounded-lg bg-surface-raised border border-line text-fg outline-none focus:border-accent" />
          </div>
        </div>

        <div>
          <label htmlFor="custom-block-params" className="block text-[11px] font-medium mb-1 text-fg-secondary">Parameters (name:type=default, comma-separated)</label>
          <input id="custom-block-params" type="text" value={params} onChange={(e) => setParams(e.target.value)}
            className="w-full px-2.5 h-8 text-xs rounded-lg bg-surface-raised border border-line text-fg outline-none focus:border-accent font-mono"
            placeholder="in_channels:int=64, out_channels:int=64" />
        </div>

        <div>
          <label htmlFor="custom-block-code" className="block text-[11px] font-medium mb-1 text-fg-secondary">PyTorch code</label>
          <textarea
            id="custom-block-code"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            spellCheck={false}
            className="w-full h-[240px] p-3 text-[11px] font-mono leading-relaxed rounded-lg resize-none outline-none bg-surface-raised border border-line text-fg focus:border-accent"
          />
        </div>
      </div>
    </Dialog>
  );
}
