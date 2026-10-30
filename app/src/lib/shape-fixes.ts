import type { ShapeError } from "./shape-inference";

/** For each parameter that sizes a layer's input, where to read the right value off the shape it receives */
const FIXABLE_PARAMS: Record<string, (received: number[]) => number | undefined> = {
  in_features: (s) => s[s.length - 1],
  input_size: (s) => s[s.length - 1],
  embed_dim: (s) => s[s.length - 1],
  d_model: (s) => s[s.length - 1],
  in_channels: (s) => (s.length >= 2 ? s[1] : undefined),
  num_features: (s) => (s.length >= 2 ? s[1] : undefined),
  num_channels: (s) => (s.length >= 2 ? s[1] : undefined),
};

/**
 * The one-parameter change that makes a layer accept the shape it receives, e.g. in_features=784 → 3072.
 * Null when the error isn't about a size parameter, or the value is already right.
 */
export function shapeErrorFix(
  error: ShapeError,
  params: Record<string, unknown>,
): { param: string; from: number; to: number } | null {
  if (error.type !== "root" || !error.paramHint || !error.receivedShape) return null;
  const read = FIXABLE_PARAMS[error.paramHint];
  const to = read?.(error.receivedShape);
  const from = Number(params[error.paramHint] ?? 0);
  if (to == null || !(to > 0) || to === from) return null;
  return { param: error.paramHint, from, to };
}

/** Short wording of a shape error for tight spaces, e.g. "Gets [1, 3072], expects [1, 784]" */
export function shapeErrorSummary(error: ShapeError, format: (shape: number[]) => string): string {
  if (error.receivedShape && error.expectedShape) {
    return `Gets ${format(error.receivedShape)}, expects ${format(error.expectedShape)}`;
  }
  return error.message;
}
