/**
 * Lets anything in the app (the toolbar, a panel) bring a node into view and select it.
 * The canvas registers the handler, since only it knows the viewport.
 */
type FocusHandler = (nodeId: string) => void;

let handler: FocusHandler | null = null;

export function setCanvasFocusHandler(h: FocusHandler): () => void {
  handler = h;
  return () => { if (handler === h) handler = null; };
}

/** Select a node and center the canvas on it */
export function focusNode(nodeId: string) {
  handler?.(nodeId);
}
