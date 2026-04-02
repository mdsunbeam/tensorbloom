/**
 * Node alignment and distribution tools.
 */

import type { Node } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";

type TLNode = Node<NodeData>;

export function alignLeft(nodes: TLNode[], selectedIds: Set<string>): TLNode[] {
  const selected = nodes.filter((n) => selectedIds.has(n.id));
  if (selected.length < 2) return nodes;
  const minX = Math.min(...selected.map((n) => n.position.x));
  return nodes.map((n) => selectedIds.has(n.id) ? { ...n, position: { ...n.position, x: minX } } : n);
}

export function alignCenter(nodes: TLNode[], selectedIds: Set<string>): TLNode[] {
  const selected = nodes.filter((n) => selectedIds.has(n.id));
  if (selected.length < 2) return nodes;
  const avgX = selected.reduce((s, n) => s + n.position.x, 0) / selected.length;
  return nodes.map((n) => selectedIds.has(n.id) ? { ...n, position: { ...n.position, x: avgX } } : n);
}

export function alignTop(nodes: TLNode[], selectedIds: Set<string>): TLNode[] {
  const selected = nodes.filter((n) => selectedIds.has(n.id));
  if (selected.length < 2) return nodes;
  const minY = Math.min(...selected.map((n) => n.position.y));
  return nodes.map((n) => selectedIds.has(n.id) ? { ...n, position: { ...n.position, y: minY } } : n);
}

export function distributeVertically(nodes: TLNode[], selectedIds: Set<string>): TLNode[] {
  const selected = nodes.filter((n) => selectedIds.has(n.id));
  if (selected.length < 3) return nodes;
  const sorted = [...selected].sort((a, b) => a.position.y - b.position.y);
  const minY = sorted[0].position.y;
  const maxY = sorted[sorted.length - 1].position.y;
  const gap = (maxY - minY) / (sorted.length - 1);
  const posMap = new Map<string, number>();
  sorted.forEach((n, i) => posMap.set(n.id, minY + i * gap));
  return nodes.map((n) => posMap.has(n.id) ? { ...n, position: { ...n.position, y: posMap.get(n.id)! } } : n);
}

export function distributeHorizontally(nodes: TLNode[], selectedIds: Set<string>): TLNode[] {
  const selected = nodes.filter((n) => selectedIds.has(n.id));
  if (selected.length < 3) return nodes;
  const sorted = [...selected].sort((a, b) => a.position.x - b.position.x);
  const minX = sorted[0].position.x;
  const maxX = sorted[sorted.length - 1].position.x;
  const gap = (maxX - minX) / (sorted.length - 1);
  const posMap = new Map<string, number>();
  sorted.forEach((n, i) => posMap.set(n.id, minX + i * gap));
  return nodes.map((n) => posMap.has(n.id) ? { ...n, position: { ...n.position, x: posMap.get(n.id)! } } : n);
}
