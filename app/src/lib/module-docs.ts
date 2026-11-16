/**
 * Builds docs/guides/modules.md, the module reference, from the registry, so the docs list exactly
 * the layers in the palette. `npm run docs:modules` writes it; module-docs.test.ts fails when it is stale.
 */
import { MODULE_REGISTRY, CATEGORY_LABELS } from "./registry";
import type { ModuleDef, NodeCategory, ParamDef } from "./types";

/** Layers in the palette's order of categories */
const CATEGORY_ORDER = Object.keys(CATEGORY_LABELS) as NodeCategory[];

function formatDefault(p: ParamDef): string {
  const v = p.default;
  if (v === "" || v === null || v === undefined) return "";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

function formatParams(def: ModuleDef): string {
  if (def.type === "Data") return "Built-in datasets or your own files; see [Custom Data](custom-data.md)";
  if (!def.params.length) return "—";
  return def.params
    .map((p) => {
      const d = formatDefault(p);
      return d ? `\`${p.name}\`=${"`"}${d.replace(/\|/g, "\\|")}${"`"}` : `\`${p.name}\``;
    })
    .join(", ");
}

/** A table cell can't hold a pipe or a line break */
function cell(text: string): string {
  return text.replace(/\|/g, "\\|").replace(/\s*\n\s*/g, " ");
}

export function moduleDocs(): string {
  const byCategory = new Map<NodeCategory, ModuleDef[]>();
  for (const def of Object.values(MODULE_REGISTRY)) {
    const list = byCategory.get(def.category) ?? [];
    list.push(def);
    byCategory.set(def.category, list);
  }
  const total = Object.keys(MODULE_REGISTRY).length;

  const out: string[] = [
    "# Module Reference",
    "",
    "<!-- Generated from app/src/lib/registry.ts by `npm run docs:modules` in app/. Don't edit by hand. -->",
    "",
    `The layer palette has ${total} modules in ${byCategory.size} categories. Each maps to its PyTorch counterpart with the same parameters, shown as a node you configure in the Properties panel, where the **?** beside each parameter explains it. The defaults below are what a new node starts with.`,
    "",
    "To add one, drag it from the **Layers** panel, click it there, or press **Ctrl+K** and type its name.",
  ];

  for (const category of CATEGORY_ORDER) {
    const defs = byCategory.get(category);
    if (!defs?.length) continue;
    out.push("", `## ${CATEGORY_LABELS[category]}`, "", "| Module | What it does | Parameters (defaults) |", "|---|---|---|");
    for (const def of defs) {
      out.push(`| **${def.label}** | ${cell(def.description)} | ${formatParams(def)} |`);
    }
  }
  out.push(
    "",
    "## Your own layers",
    "",
    "When none of these does the job, write one: **Insert > Custom Block** adds a reusable layer with its own `__init__` and `forward`, and **Convert to Code Block** turns any node into code. See [Code Blocks](code-blocks.md).",
    "",
  );
  return out.join("\n");
}
