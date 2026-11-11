/**
 * Share links: the whole model packed into the link's #fragment, so it opens in the web version
 * with nothing stored on a server. The fragment never reaches a server either.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore, type TrainingConfig } from "../store/training-store";
import { normalizeGroups, type LayerGroup } from "./groups";
import { IS_WEB } from "./platform";
import { LINKS } from "./links";
import { toast } from "../components/Toast";

const PREFIX = "#m=";

export interface SharedModel {
  name: string;
  nodes: Node<NodeData>[];
  edges: Edge[];
  groups: LayerGroup[];
  training?: TrainingConfig;
}

function toBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(s: string): Uint8Array {
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/"));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function pipe(bytes: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(stream));
  return new Uint8Array(await out.arrayBuffer());
}

/** Packs a model into the text after "#m=" */
export async function encodeModel(model: SharedModel): Promise<string> {
  const json = JSON.stringify({
    v: 1,
    name: model.name,
    nodes: model.nodes.map((n) => ({ id: n.id, type: n.type, position: n.position, data: n.data })),
    edges: model.edges.map((e) => ({ id: e.id, source: e.source, target: e.target, sourceHandle: e.sourceHandle, targetHandle: e.targetHandle, type: e.type })),
    groups: model.groups,
    training: model.training,
  });
  return toBase64Url(await pipe(new TextEncoder().encode(json), new CompressionStream("deflate-raw")));
}

/** Unpacks what encodeModel made, or returns null when the text isn't a model */
export async function decodeModel(packed: string): Promise<SharedModel | null> {
  try {
    const json = JSON.parse(new TextDecoder().decode(await pipe(fromBase64Url(packed), new DecompressionStream("deflate-raw"))));
    if (!json || json.v !== 1 || !Array.isArray(json.nodes) || !Array.isArray(json.edges)) return null;
    return {
      name: typeof json.name === "string" && json.name ? json.name : "Shared model",
      nodes: json.nodes.map((n: Node<NodeData>) => ({ ...n, type: n.type || "tensorLabNode" })),
      edges: json.edges.map((e: Edge) => ({ ...e, type: e.type ?? "smoothstep" })),
      groups: normalizeGroups(json.groups),
      training: json.training && typeof json.training === "object" ? json.training : undefined,
    };
  } catch {
    return null;
  }
}

/** A link to the web version that opens the current model */
export async function shareLinkForCurrentModel(): Promise<string> {
  const { nodes, edges, groups, projectName } = useGraphStore.getState();
  const packed = await encodeModel({ name: projectName, nodes, edges, groups, training: useTrainingStore.getState().config });
  const base = IS_WEB ? `${location.origin}${location.pathname}` : LINKS.webEditor;
  return `${base}${PREFIX}${packed}`;
}

/** The packed model in a #m= fragment, if there is one */
export function packedModelFromHash(hash: string): string | null {
  return hash.startsWith(PREFIX) ? hash.slice(PREFIX.length) : null;
}

/** "ResNet-18" → "resnet-18", for ?template= links */
export function templateSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

/** Copies a share link for the current model, and says how long it is when that may matter */
export async function copyShareLink() {
  if (useGraphStore.getState().nodes.length === 0) {
    toast.info("Add some layers first. A share link carries the model on the canvas.");
    return;
  }
  try {
    const url = await shareLinkForCurrentModel();
    await navigator.clipboard.writeText(url);
    toast.success(url.length > 8000
      ? "Link copied. It's long because the whole model is inside it, so some chat apps may cut it short; a saved file is safer."
      : "Link copied. Anyone who opens it gets this model in the web version, no install needed.");
  } catch {
    toast.error("Couldn't copy the link. Your browser may have blocked the clipboard.");
  }
}
