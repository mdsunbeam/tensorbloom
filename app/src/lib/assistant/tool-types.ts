import type Anthropic from "@anthropic-ai/sdk";

export type ToolDefinition = Anthropic.Beta.BetaTool;
export type ToolResultContent = Anthropic.Beta.BetaToolResultBlockParam["content"];

/** What the user sees before a tool that runs code (or starts training) is allowed to go ahead */
export interface ApprovalRequest {
  title: string;
  /** What will happen, in a sentence or two */
  summary: string;
  /** The exact code that would run, shown in full */
  code?: string;
  /** What running it means for their computer */
  warning?: string;
  approveLabel: string;
}

/**
 * A tool the assistant can call. "read" tools only look and run right away. "edit" tools change the
 * model as a preview the user applies or discards. "confirm" tools run code or start training, so the
 * user approves each call first.
 */
export interface AssistantTool<I = Record<string, unknown>> {
  definition: ToolDefinition;
  kind: "read" | "edit" | "confirm";
  /** For "confirm" tools: what to show the user. May throw a ToolError when the input can't work. */
  confirm?: (input: I) => ApprovalRequest;
  /** Shown in the chat while the tool runs, e.g. "Reading the graph" */
  describe: (input: I) => string;
  run: (input: I) => Promise<unknown>;
}

/** Thrown by a tool when its input is wrong or it can't do what was asked. The message goes back to the model. */
export class ToolError extends Error {}

type Schema = {
  type?: string | string[];
  properties?: Record<string, Schema>;
  required?: string[];
  enum?: unknown[];
  items?: Schema;
};

/**
 * Checks a tool's input against its JSON schema (the subset our tools use), so a malformed call gets a
 * readable error instead of misbehaving.
 */
export function validateInput(schema: Schema, value: unknown, path = "input"): string | null {
  if (Array.isArray(schema.type)) {
    // Any one of the types will do, e.g. ["number", "null"]
    const errors = schema.type.map((t) => (t === "null" ? (value === null ? null : `${path} must be null`) : validateInput({ ...schema, type: t }, value, path)));
    return errors.includes(null) ? null : errors.join(", or ");
  }
  if (schema.enum && !schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(value))) {
    return `${path} must be one of ${schema.enum.map((e) => JSON.stringify(e)).join(", ")}`;
  }
  switch (schema.type) {
    case "object": {
      if (!value || typeof value !== "object" || Array.isArray(value)) return `${path} must be an object`;
      const obj = value as Record<string, unknown>;
      for (const r of schema.required ?? []) if (obj[r] === undefined) return `${path}.${r} is required`;
      for (const [k, v] of Object.entries(obj)) {
        const sub = schema.properties?.[k];
        if (!sub) {
          if (schema.properties) return `${path}.${k} isn't a known field (known: ${Object.keys(schema.properties).join(", ") || "none"})`;
          continue;
        }
        const err = validateInput(sub, v, `${path}.${k}`);
        if (err) return err;
      }
      return null;
    }
    case "array": {
      if (!Array.isArray(value)) return `${path} must be an array`;
      for (let i = 0; i < value.length; i++) {
        const err = schema.items ? validateInput(schema.items, value[i], `${path}[${i}]`) : null;
        if (err) return err;
      }
      return null;
    }
    case "string": return typeof value === "string" ? null : `${path} must be a string`;
    case "integer": return Number.isInteger(value) ? null : `${path} must be an integer`;
    case "number": return typeof value === "number" && Number.isFinite(value) ? null : `${path} must be a number`;
    case "boolean": return typeof value === "boolean" ? null : `${path} must be true or false`;
    default: return null;
  }
}

/** Layer types that are closest to a name the model gave, for "did you mean" hints */
export function closestNames(name: string, candidates: string[], limit = 3): string[] {
  const n = name.toLowerCase();
  const score = (c: string) => {
    const l = c.toLowerCase();
    if (l === n) return 0;
    if (l.includes(n) || n.includes(l)) return 1;
    return 2 + editDistance(n, l);
  };
  return [...candidates].sort((a, b) => score(a) - score(b)).slice(0, limit);
}

function editDistance(a: string, b: string): number {
  const dp = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) dp[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
  }
  return dp[a.length][b.length];
}
