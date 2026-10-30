import { createContext } from "react";
import type { Node, Edge } from "@xyflow/react";

/**
 * The nodes and edges the canvas is drawing, which differ from the graph's when a group is
 * collapsed. Edges route around what's actually on screen.
 */
export const CanvasViewContext = createContext<{ nodes: Node[]; edges: Edge[] } | null>(null);
