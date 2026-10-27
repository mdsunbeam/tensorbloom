/**
 * Persistence for user-defined custom blocks (Create Custom Block dialog).
 * Blocks live in localStorage and are re-registered into MODULE_REGISTRY on startup.
 */

import { MODULE_REGISTRY } from "./registry";
import type { ModuleDef, NodeCategory } from "./types";

const STORAGE_KEY = "tensorbloom-custom-blocks";
// Key used before the TensorLab → TensorBloom rename
const LEGACY_STORAGE_KEY = "tensorlab-custom-blocks";

export interface CustomBlock {
  code: string;
  params: ModuleDef["params"];
  category: string;
}

export function loadCustomBlocks(): Record<string, CustomBlock> {
  try {
    const current = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
    const legacy = JSON.parse(localStorage.getItem(LEGACY_STORAGE_KEY) ?? "{}");
    return { ...legacy, ...current };
  } catch {
    return {};
  }
}

export function saveCustomBlock(name: string, block: CustomBlock) {
  const blocks = loadCustomBlocks();
  blocks[name] = block;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(blocks));
}

export function getCustomBlock(name: string): CustomBlock | undefined {
  return loadCustomBlocks()[name];
}

export function customBlockDef(name: string, params: ModuleDef["params"], category: string): ModuleDef {
  return {
    type: name,
    label: name,
    category: category as NodeCategory,
    description: `Custom block: ${name}`,
    params,
    inputHandles: [{ id: "in", label: "in", dataType: "tensor" }],
    outputHandles: [{ id: "out", label: "out", dataType: "tensor" }],
  };
}

/** Register all saved custom blocks into the module registry. */
export function restoreCustomBlocks() {
  const blocks = loadCustomBlocks();
  for (const [name, block] of Object.entries(blocks)) {
    if (!MODULE_REGISTRY[name]) {
      MODULE_REGISTRY[name] = customBlockDef(name, block.params, block.category);
    }
  }
  // Migrate legacy entries to the current key
  try {
    if (localStorage.getItem(LEGACY_STORAGE_KEY) !== null) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(blocks));
      localStorage.removeItem(LEGACY_STORAGE_KEY);
    }
  } catch { /* storage unavailable */ }
}
