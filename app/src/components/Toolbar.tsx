import { useCallback, useMemo, useState, useRef, useEffect } from "react";
import { useThemeStore } from "../store/theme-store";
import { useGraphStore } from "../store/graph-store";
import { useHistoryStore } from "../store/history-store";
import { downloadProject, loadProjectFile, deserializeProject } from "../lib/project";
import { autoLayout } from "../lib/auto-layout";
import { computeModelSummary, formatParamCount } from "../lib/model-summary";
import { toast } from "./Toast";
import { useEscapeKey } from "../lib/use-escape-key";

interface ToolbarProps {
  onShowImport: () => void;
  onShowTemplates: () => void;
  onShowExport: () => void;
  onShowSpotlight: () => void;
  onShowCustomBlock: () => void;
  gpuInfo: string | null;
}

export default function Toolbar({ onShowImport, onShowTemplates, onShowExport, onShowSpotlight, onShowCustomBlock, gpuInfo }: ToolbarProps) {
  const theme = useThemeStore((s) => s.theme);
  const toggleTheme = useThemeStore((s) => s.toggleTheme);
  const nodes = useGraphStore((s) => s.nodes);
  const edgeCount = useGraphStore((s) => s.edges.length);
  const summary = useMemo(() => computeModelSummary(nodes), [nodes]);
  const nodeCount = nodes.length;
  const canUndo = useHistoryStore((s) => s.canUndo);
  const canRedo = useHistoryStore((s) => s.canRedo);
  const undo = useHistoryStore((s) => s.undo);
  const redo = useHistoryStore((s) => s.redo);

  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const menuBarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (openMenu && menuBarRef.current && !menuBarRef.current.contains(e.target as HTMLElement)) setOpenMenu(null);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [openMenu]);

  const closeMenu = useCallback(() => setOpenMenu(null), []);
  useEscapeKey(closeMenu, { enabled: openMenu !== null });

  const handleSave = useCallback(async () => {
    const { nodes, edges, projectName } = useGraphStore.getState();
    const result = await downloadProject(projectName, nodes, edges);
    if (result && typeof result === "string") {
      // Extract name from saved path: "/path/to/my_model.tbloom" → "my_model"
      const filename = result.split(/[/\\]/).pop() ?? "";
      const name = filename.replace(/\.(tbloom|json)$/i, "").replace(/_/g, " ");
      if (name && name !== projectName) {
        useGraphStore.getState().setProjectName(name);
      }
      toast.success("Project saved");
    }
  }, []);

  const handleOpen = useCallback(async () => {
    const project = await loadProjectFile();
    if (!project) return;
    const { nodes, edges, warnings } = deserializeProject(project);
    useHistoryStore.getState().pushSnapshot();
    useGraphStore.getState().loadGraph(nodes, edges);
    useGraphStore.getState().setProjectName(project.name);
    for (const w of warnings) toast.warning(w);
    toast.success(`Opened "${project.name}"`);
  }, []);

  const handleNew = useCallback(() => {
    useHistoryStore.getState().pushSnapshot();
    useGraphStore.getState().clearGraph();
    useGraphStore.getState().setProjectName("Untitled");
  }, []);

  const handleAutoLayout = useCallback(() => {
    const { nodes, edges } = useGraphStore.getState();
    if (nodes.length === 0) return;
    useHistoryStore.getState().pushSnapshot();
    const laid = autoLayout(nodes, edges);
    useGraphStore.getState().loadGraph(laid, edges);
  }, []);

  const menus: { label: string; id: string; items: MenuItem[] }[] = [
    {
      label: "File", id: "file", items: [
        { label: "New Project", shortcut: "Ctrl+N", action: handleNew },
        { label: "Open Project...", shortcut: "Ctrl+O", action: handleOpen },
        { label: "Save Project", shortcut: "Ctrl+S", action: handleSave },
        { type: "separator" },
        { label: "Import PyTorch Code...", shortcut: "Ctrl+I", action: onShowImport },
        { label: "Export Model...", shortcut: "Ctrl+E", action: onShowExport },
      ],
    },
    {
      label: "Edit", id: "edit", items: [
        { label: "Undo", shortcut: "Ctrl+Z", action: undo, disabled: !canUndo },
        { label: "Redo", shortcut: "Ctrl+Y", action: redo, disabled: !canRedo },
        { type: "separator" },
        { label: "Select All", shortcut: "Ctrl+A", action: () => useGraphStore.getState().selectAll() },
        { label: "Duplicate", shortcut: "Ctrl+D", action: () => { useHistoryStore.getState().pushSnapshot(); useGraphStore.getState().duplicateSelected(); } },
        { label: "Delete Selected", shortcut: "Del", action: () => { useHistoryStore.getState().pushSnapshot(); useGraphStore.getState().deleteSelected(); } },
      ],
    },
    {
      label: "Insert", id: "insert", items: [
        { label: "Add Layer...", shortcut: "Ctrl+K", action: onShowSpotlight },
        { label: "From Template...", action: onShowTemplates },
        { label: "Custom Block...", action: onShowCustomBlock },
      ],
    },
    {
      label: "View", id: "view", items: [
        { label: "Auto-Layout", shortcut: "Ctrl+L", action: handleAutoLayout },
        { type: "separator" },
        { label: `${theme === "dark" ? "Light" : "Dark"} Theme`, action: toggleTheme },
      ],
    },
  ];

  return (
    <div
      className="flex items-center h-9 px-1 flex-shrink-0 select-none"
      style={{ background: "var(--bg-secondary)", borderBottom: "1px solid var(--border)" }}
    >
      {/* Menu bar */}
      <div className="flex items-center" ref={menuBarRef}>
        {menus.map((menu) => (
          <div key={menu.id} className="relative">
            <button
              onClick={() => setOpenMenu(openMenu === menu.id ? null : menu.id)}
              onMouseEnter={() => { if (openMenu) setOpenMenu(menu.id); }}
              className="px-2.5 py-1 text-[11px] font-medium rounded-sm"
              style={{
                color: openMenu === menu.id ? "var(--text-primary)" : "var(--text-secondary)",
                background: openMenu === menu.id ? "var(--bg-tertiary)" : "transparent",
              }}
              onMouseLeave={(e) => {
                if (openMenu !== menu.id) e.currentTarget.style.background = "transparent";
              }}
            >
              {menu.label}
            </button>
            {openMenu === menu.id && (
              <div
                className="absolute top-full left-0 mt-0.5 py-1 rounded-md z-50 min-w-[220px]"
                style={{
                  background: "var(--bg-secondary)",
                  border: "1px solid var(--border)",
                  boxShadow: "0 4px 16px rgba(0,0,0,0.2), 0 1px 4px rgba(0,0,0,0.1)",
                }}
              >
                {menu.items.map((item, i) =>
                  item.type === "separator" ? (
                    <div key={i} className="my-1 mx-2" style={{ borderTop: "1px solid var(--border)" }} />
                  ) : (
                    <button
                      key={i}
                      onClick={() => { setOpenMenu(null); item.action?.(); }}
                      disabled={item.disabled}
                      className="w-full flex items-center justify-between px-3 py-1.5 text-[11px] text-left transition-colors"
                      style={{
                        color: item.disabled ? "var(--text-muted)" : "var(--text-secondary)",
                        opacity: item.disabled ? 0.5 : 1,
                      }}
                      onMouseEnter={(e) => { if (!item.disabled) { e.currentTarget.style.background = "var(--bg-tertiary)"; e.currentTarget.style.color = "var(--text-primary)"; } }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = item.disabled ? "var(--text-muted)" : "var(--text-secondary)"; }}
                    >
                      <span>{item.label}</span>
                      {item.shortcut && <span className="text-[9px] ml-4" style={{ color: "var(--text-muted)" }}>{item.shortcut}</span>}
                    </button>
                  )
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Editable project name */}
      <ProjectNameInput />

      <div className="flex-1" />

      {/* Stats */}
      <span className="text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>
        {nodeCount} nodes · {edgeCount} edges
        {summary.totalParams > 0 && <> · {formatParamCount(summary.totalParams)} params</>}
      </span>

      <div className="flex-1" />

      {/* GPU info */}
      {gpuInfo && (
        <div className="flex items-center gap-1 mr-2" title={gpuInfo}>
          <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--text-muted)" }}>
            <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="8" y="8" width="8" height="8" rx="1"/>
          </svg>
          <span className="text-[10px] tabular-nums whitespace-nowrap" style={{ color: "var(--text-muted)" }}>
            {gpuInfo}
          </span>
        </div>
      )}

      {/* Theme toggle */}
      <button
        onClick={toggleTheme}
        title={`${theme === "dark" ? "Light" : "Dark"} mode`}
        className="w-7 h-7 flex items-center justify-center rounded-md"
        style={{ color: "var(--text-muted)" }}
        onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text-secondary)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-muted)"; }}
      >
        {theme === "dark" ? ICONS.sun : ICONS.moon}
      </button>
    </div>
  );
}

function ProjectNameInput() {
  const projectName = useGraphStore((s) => s.projectName);
  const setProjectName = useGraphStore((s) => s.setProjectName);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(projectName);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setDraft(projectName); }, [projectName]);
  useEffect(() => { if (editing) inputRef.current?.select(); }, [editing]);

  const commit = () => {
    const name = draft.trim() || "Untitled";
    setProjectName(name);
    setEditing(false);
  };

  if (editing) {
    return (
      <input ref={inputRef} value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") commit(); if (e.key === "Escape") { setDraft(projectName); setEditing(false); } }}
        className="ml-2 px-1.5 py-0.5 text-[11px] rounded-sm max-w-[160px]"
        style={{ background: "transparent", border: "1px solid var(--accent)", color: "var(--text-primary)", outline: "none" }}
      />
    );
  }

  return (
    <button
      onClick={() => setEditing(true)}
      className="ml-2 px-1.5 py-0.5 text-[11px] truncate max-w-[160px] rounded-sm"
      style={{ color: "var(--text-muted)" }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "var(--bg-tertiary)"; e.currentTarget.style.color = "var(--text-secondary)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--text-muted)"; }}
      title="Click to rename project"
    >
      {projectName}
    </button>
  );
}

type MenuItem = { label?: string; shortcut?: string; action?: () => void; disabled?: boolean; type?: "separator" };

const s = { width: 13, height: 13, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
const ICONS = {
  sun: <svg {...s}><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>,
  moon: <svg {...s}><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>,
};
