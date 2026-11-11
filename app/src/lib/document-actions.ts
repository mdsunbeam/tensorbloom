/**
 * Project-level actions (new, open, save, replace the graph) shared by the menus, shortcuts and
 * dialogs, so each one checks for unsaved changes the same way.
 */

import { settlePendingChanges } from "./assistant/change-set";
import type { Node, Edge } from "@xyflow/react";
import { useGraphStore, type NodeData } from "../store/graph-store";
import { useHistoryStore } from "../store/history-store";
import { useDocumentStore } from "../store/document-store";
import { useRunStore } from "../store/run-store";
import { useLLMStore } from "../store/llm-store";
import { useLessonStore } from "../store/lesson-store";
import { normalizeGroups, type LayerGroup } from "./groups";
import type { LLMSpec } from "./llm";
import { writeProjectFile, loadProjectFile, deserializeProject } from "./project";
import { clearAutoSave, type AutoSaveData } from "./auto-save";
import { templateGraph, templateGroups, type Template } from "./templates";
import { useTrainingStore } from "../store/training-store";
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
  toast.success(result.path ? "Project saved" : "Project downloaded as a .tbloom file");
  return true;
}

/** Replaces the graph with a different project. Undo can still bring the old graph back. */
export function replaceGraph(nodes: Node<NodeData>[], edges: Edge[], projectName: string, groups: LayerGroup[] = [], llm: LLMSpec | null = null) {
  settlePendingChanges();
  useHistoryStore.getState().pushSnapshot();
  // The pretrained LLM the graph came from, if any
  useLLMStore.getState().setSpec(llm);
  useGraphStore.getState().loadGraph(nodes, edges, groups);
  useGraphStore.getState().setProjectName(projectName);
}

/** Opens a template in place of the graph, with its stages as collapsed groups and its training defaults. Doesn't ask first. */
export function applyTemplate(tpl: Template) {
  const { nodes, edges } = templateGraph(tpl);
  const groups = templateGroups(tpl, nodes);
  replaceGraph(nodes, edges, tpl.name, groups);
  // Collapse after loading, so each fold moves the layers below it up
  for (const g of groups) useGraphStore.getState().setGroupCollapsed(g.id, true);
  if (tpl.trainingDefaults) {
    const d = tpl.trainingDefaults;
    useTrainingStore.getState().applyTemplateDefaults(tpl.name, {
      ...(d.optimizer && { optimizer: { ...useTrainingStore.getState().config.optimizer, ...d.optimizer } }),
      ...(d.loss && { loss: d.loss }),
      ...(d.epochs != null && { epochs: d.epochs }),
      ...(d.batchSize != null && { batchSize: d.batchSize }),
      ...(d.dataset && { dataset: d.dataset }),
    });
  }
  // A fresh template has nothing to lose until it's edited
  useDocumentStore.getState().markSaved(null);
}

export async function newProject() {
  if (!(await confirmDiscardChanges("starting a new project"))) return;
  settlePendingChanges();
  useLessonStore.getState().exit();
  useHistoryStore.getState().pushSnapshot();
  useGraphStore.getState().clearGraph();
  useLLMStore.getState().setSpec(null);
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
  useLessonStore.getState().exit();
  const { nodes, edges, groups, warnings } = deserializeProject(project);
  replaceGraph(nodes, edges, project.name, groups, project.llm ?? null);
  if (project.llm && project.llmFinetune) useLLMStore.setState({ finetune: { ...useLLMStore.getState().finetune, ...project.llmFinetune } });
  useDocumentStore.getState().markSaved(path);
  clearAutoSave();
  for (const w of warnings) toast.warning(w);
  toast.success(`Opened "${project.name}"`);
}

/** Brings back unsaved work from the last session. It stays unsaved until the user saves it. */
export function restoreAutoSave(data: AutoSaveData) {
  useLessonStore.getState().exit();
  replaceGraph(data.nodes, data.edges, data.projectName, normalizeGroups(data.groups), data.llm ?? null);
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
