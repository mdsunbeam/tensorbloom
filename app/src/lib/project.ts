/**
 * Project save/load utilities.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import { useTrainingStore, type TrainingConfig, type EnvironmentInfo, type TrainingResults } from "../store/training-store";
import { getCachedEnvironment } from "./sidecar";

export interface ProjectFile {
  version: "1.0" | "1.1" | "2.0";
  name: string;
  nodes: Array<{
    id: string;
    type: string;
    position: { x: number; y: number };
    data: NodeData;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    sourceHandle?: string | null;
    targetHandle?: string | null;
    type?: string;
  }>;
  training?: TrainingConfig;
  metadata?: {
    description?: string;
    tags?: string[];
    createdAt?: string;
    savedAt?: string;
  };
  environment?: EnvironmentInfo;
  trainingResults?: TrainingResults;
}

export function serializeProject(
  name: string,
  nodes: Node<NodeData>[],
  edges: Edge[],
  existingMetadata?: ProjectFile["metadata"],
): ProjectFile {
  const now = new Date().toISOString();
  const store = useTrainingStore.getState();
  return {
    version: "2.0",
    name,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type ?? "tensorLabNode",
      position: n.position,
      data: n.data,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
      type: e.type,
    })),
    training: store.config,
    metadata: {
      ...existingMetadata,
      createdAt: existingMetadata?.createdAt ?? now,
      savedAt: now,
    },
    environment: getCachedEnvironment() ?? undefined,
    trainingResults: store.trainingResults ?? undefined,
  };
}

export function deserializeProject(
  json: ProjectFile,
): { nodes: Node<NodeData>[]; edges: Edge[]; warnings: string[] } {
  const warnings: string[] = [];

  // Restore training config if present
  if (json.training) {
    useTrainingStore.getState().updateConfig(json.training);
  }

  // Restore training results if present (v2.0)
  if (json.trainingResults) {
    useTrainingStore.getState().setTrainingResults(json.trainingResults);
  } else {
    useTrainingStore.getState().setTrainingResults(null);
  }

  // Compare environment if present (v2.0)
  if (json.environment) {
    const current = getCachedEnvironment();
    if (current && json.environment.torchVersion && current.torchVersion &&
        json.environment.torchVersion !== current.torchVersion) {
      warnings.push(
        `Project was saved with PyTorch ${json.environment.torchVersion}, but current environment has ${current.torchVersion}. Results may differ.`
      );
    }
  }

  return {
    nodes: json.nodes.map((n) => ({
      id: n.id,
      type: n.type || "tensorLabNode",
      position: n.position,
      data: n.data,
    })),
    edges: json.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
      type: e.type ?? "smoothstep",
    })),
    warnings,
  };
}

/**
 * Writes the project to `path`, or asks where to save it when there's no path yet (Save As).
 * Resolves to the path written (null when the browser downloaded it instead), or null if the user cancelled.
 */
export async function writeProjectFile(
  name: string,
  nodes: Node<NodeData>[],
  edges: Edge[],
  path: string | null,
): Promise<{ path: string | null } | null> {
  const project = serializeProject(name, nodes, edges);
  const json = JSON.stringify(project, null, 2);

  try {
    const { writeTextFile } = await import("@tauri-apps/plugin-fs");
    if (!path) {
      // Tauri native save dialog
      const { save } = await import("@tauri-apps/plugin-dialog");
      path = await save({
        defaultPath: `${name.replace(/\s+/g, "_").toLowerCase()}.tbloom`,
        filters: [{ name: "TensorBloom Project", extensions: ["tbloom"] }, { name: "JSON", extensions: ["json"] }],
        title: "Save Project",
      });
      if (!path) return null; // user cancelled
    }
    await writeTextFile(path, json);
    return { path };
  } catch (e) {
    console.error("[save] Tauri save failed, falling back to download:", e);
    // Fallback: browser download
  }

  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${name.replace(/\s+/g, "_").toLowerCase()}.tbloom`;
  a.click();
  URL.revokeObjectURL(url);
  return { path: null };
}

export function downloadCode(code: string, filename: string = "model.py") {
  const blob = new Blob([code], { type: "text/x-python" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Asks for a project file and reads it. `path` is null when the browser's file picker was used. */
export async function loadProjectFile(): Promise<{ project: ProjectFile; path: string | null } | null> {
  try {
    // Tauri native open dialog
    const { open } = await import("@tauri-apps/plugin-dialog");
    const { readTextFile } = await import("@tauri-apps/plugin-fs");
    const path = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "TensorBloom Project", extensions: ["tbloom", "json"] }],
      title: "Open Project",
    });
    if (path && typeof path === "string") {
      const text = await readTextFile(path);
      const parsed = JSON.parse(text) as ProjectFile;
      if (!parsed.version || !parsed.nodes) throw new Error("Invalid project file");
      return { project: parsed, path };
    }
    return null;
  } catch {
    // Fallback: browser file input
    return new Promise((resolve) => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".tbloom,.json";
      input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) return resolve(null);
        try {
          const text = await file.text();
          const parsed = JSON.parse(text) as ProjectFile;
          if (!parsed.version || !parsed.nodes) throw new Error("Invalid project file");
          resolve({ project: parsed, path: null });
        } catch {
          resolve(null);
        }
      };
      input.click();
    });
  }
}
