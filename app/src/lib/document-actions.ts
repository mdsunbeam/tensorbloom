/**
 * Project-level actions (new, open, save, replace the graph) shared by the menus, shortcuts and
 * dialogs, so each one checks for unsaved changes the same way.
 */

import type { Node, Edge } from "@xyflow/react";
import { useGraphStore, type NodeData } from "../store/graph-store";
import { useHistoryStore } from "../store/history-store";
import { useDocumentStore } from "../store/document-store";
import { useRunStore } from "../store/run-store";
import { normalizeGroups, type LayerGroup } from "./groups";
import { writeProjectFile, loadProjectFile, deserializeProject } from "./project";
import { clearAutoSave, type AutoSaveData } from "./auto-save";
import { askToSaveChanges } from "../components/ConfirmDialog";
import { toast } from "../components/Toast";

/**
 * Before an action that replaces or closes the project: if there are unsaved changes, asks whether
 * to save them. Resolves to true when it's fine to go ahead.
 * `action` finishes the sentence "If you don't save before ..., they'll be lost".
 */
export async function confirmDiscardChanges(action: string): Promise<boolean> {
  if (!useDocumentStore.getState().dirty) return true;
  const choice = await askToSaveChanges(useGraphStore.getState().projectName, action);
  if (choice === "cancel") return false;
  if (choice === "save") return saveProject();
  return true;
}

/** Saves to the project's file, or asks where when it has none yet or `saveAs` is set. Resolves to true once saved. */
export async function saveProject({ saveAs = false } = {}): Promise<boolean> {
  const { nodes, edges, projectName } = useGraphStore.getState();
  const knownPath = saveAs ? null : useDocumentStore.getState().filePath;
  const result = await writeProjectFile(projectName, nodes, edges, knownPath);
  if (!result) return false;
  if (result.path && result.path !== knownPath) {
    // Name the project after the file it was saved as: "/path/to/my_model.tbloom" → "my model"
    const filename = result.path.split(/[/\\]/).pop() ?? "";
    const name = filename.replace(/\.(tbloom|json)$/i, "").replace(/_/g, " ");
    if (name && name !== projectName) useGraphStore.getState().setProjectName(name);
  }
  useDocumentStore.getState().markSaved(result.path);
  clearAutoSave();
  toast.success("Project saved");
  return true;
}

/** Replaces the graph with a different project. Undo can still bring the old graph back. */
export function replaceGraph(nodes: Node<NodeData>[], edges: Edge[], projectName: string, groups: LayerGroup[] = []) {
  useHistoryStore.getState().pushSnapshot();
  useGraphStore.getState().loadGraph(nodes, edges, groups);
  useGraphStore.getState().setProjectName(projectName);
}

export async function newProject() {
  if (!(await confirmDiscardChanges("starting a new project"))) return;
  useHistoryStore.getState().pushSnapshot();
  useGraphStore.getState().clearGraph();
  useGraphStore.getState().setProjectName("Untitled");
  useRunStore.getState().setRuns([]);
  useDocumentStore.getState().markSaved(null);
  clearAutoSave();
}

export async function openProject() {
  if (!(await confirmDiscardChanges("opening another project"))) return;
  const opened = await loadProjectFile();
  if (!opened) return;
  const { project, path } = opened;
  const { nodes, edges, groups, warnings } = deserializeProject(project);
  replaceGraph(nodes, edges, project.name, groups);
  useDocumentStore.getState().markSaved(path);
  clearAutoSave();
  for (const w of warnings) toast.warning(w);
  toast.success(`Opened "${project.name}"`);
}

/** Brings back unsaved work from the last session. It stays unsaved until the user saves it. */
export function restoreAutoSave(data: AutoSaveData) {
  replaceGraph(data.nodes, data.edges, data.projectName, normalizeGroups(data.groups));
  useRunStore.getState().setRuns(data.runs ?? []);
  useDocumentStore.getState().markSaved(data.filePath ?? null);
  useDocumentStore.getState().markUnsaved();
}

/** Groups the selected layers, or says how to select several */
export function groupSelection() {
  const s = useGraphStore.getState();
  const count = s.nodes.filter((n) => n.selected || n.id === s.selectedNodeId).length;
  if (count < 2) {
    toast.info("Select two or more layers to group them: Shift-drag a box around them, or Ctrl-click each one.");
    return;
  }
  useHistoryStore.getState().pushSnapshot();
  s.groupSelected();
}

/** Removes the selected group, or the group of the selected layer. The layers stay. */
export function ungroupSelection() {
  const s = useGraphStore.getState();
  const id = s.selectedGroupId ?? s.groups.find((g) => s.selectedNodeId && g.nodeIds.includes(s.selectedNodeId))?.id;
  if (!id) return;
  useHistoryStore.getState().pushSnapshot();
  s.ungroup(id);
}
