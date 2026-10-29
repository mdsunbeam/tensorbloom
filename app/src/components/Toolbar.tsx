import { useCallback, useMemo, useState, useRef, useEffect } from "react";
import { useThemeStore } from "../store/theme-store";
import { useGraphStore } from "../store/graph-store";
import { useHistoryStore } from "../store/history-store";
import { useDocumentStore } from "../store/document-store";
import { autoLayout } from "../lib/auto-layout";
import { computeModelSummary, formatParamCount } from "../lib/model-summary";
import { newProject, openProject, saveProject } from "../lib/document-actions";
import { useEscapeKey } from "../lib/use-escape-key";
import RunControls from "./RunControls";
import ProblemsButton from "./ProblemsButton";
import { IconButton } from "./ui/Button";
import { LogoMark, Wordmark } from "./ui/Logo";

interface ToolbarProps {
  onShowImport: () => void;
  onShowTemplates: () => void;
  onShowExport: () => void;
  onShowSpotlight: () => void;
  onShowCustomBlock: () => void;
  /** Opens the Charts tab, where the run's details are */
  onShowRun: () => void;
  gpuInfo: string | null;
}

export default function Toolbar({ onShowImport, onShowTemplates, onShowExport, onShowSpotlight, onShowCustomBlock, onShowRun, gpuInfo }: ToolbarProps) {
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
        { label: "New Project", shortcut: "Ctrl+N", action: newProject },
        { label: "Open Project...", shortcut: "Ctrl+O", action: openProject },
        { label: "Save", shortcut: "Ctrl+S", action: () => saveProject() },
        { label: "Save As...", shortcut: "Ctrl+Shift+S", action: () => saveProject({ saveAs: true }) },
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
    <div className="flex items-center gap-4 h-10 px-2 flex-shrink-0 select-none bg-surface border-b border-line">
      {/* Left: brand and menus */}
      <div className="flex-1 basis-0 flex items-center min-w-0">
        <div className="flex items-center gap-1.5 pl-1 pr-3 mr-1 h-5 border-r border-line">
          <LogoMark size={16} className="text-accent-text" />
          <Wordmark className="text-[13px]" />
        </div>
        <div className="flex items-center" ref={menuBarRef}>
          {menus.map((menu) => (
            <div key={menu.id} className="relative">
              <button
                aria-expanded={openMenu === menu.id}
                onClick={() => setOpenMenu(openMenu === menu.id ? null : menu.id)}
                onMouseEnter={() => { if (openMenu) setOpenMenu(menu.id); }}
                className={`px-2.5 h-7 text-xs rounded-md transition-colors ${
                  openMenu === menu.id ? "bg-surface-raised text-fg" : "text-fg-secondary hover:bg-surface-raised hover:text-fg"
                }`}
              >
                {menu.label}
              </button>
              {openMenu === menu.id && (
                <div
                  className="absolute top-full left-0 mt-1 py-1 rounded-lg z-50 min-w-[230px] bg-surface border border-line"
                  style={{ boxShadow: "0 8px 24px rgba(0,0,0,0.25), 0 2px 6px rgba(0,0,0,0.12)" }}
                >
                  {menu.items.map((item, i) =>
                    item.type === "separator" ? (
                      <div key={i} role="separator" className="my-1 mx-2 border-t border-line" />
                    ) : (
                      <button
                        key={i}
                        onClick={() => { setOpenMenu(null); item.action?.(); }}
                        disabled={item.disabled}
                        className="w-full flex items-center justify-between px-3 py-1.5 text-xs text-left transition-colors text-fg-secondary enabled:hover:bg-surface-raised enabled:hover:text-fg disabled:text-fg-muted disabled:opacity-60"
                      >
                        <span>{item.label}</span>
                        {item.shortcut && <span className="text-[11px] ml-4 text-fg-muted">{item.shortcut}</span>}
                      </button>
                    )
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Center: the project's name, which is also where it's renamed */}
      <div className="flex items-center gap-2 min-w-0 max-w-[40%]">
        <ProjectNameInput />
        <span className="text-[11px] tabular-nums whitespace-nowrap text-fg-muted">
          {nodeCount} {nodeCount === 1 ? "node" : "nodes"} · {edgeCount} {edgeCount === 1 ? "edge" : "edges"}
          {summary.totalParams > 0 && <> · {formatParamCount(summary.totalParams)} params</>}
        </span>
        <ProblemsButton />
      </div>

      {/* Right: hardware, the run, theme */}
      <div className="flex-1 basis-0 flex items-center justify-end gap-2 min-w-0">
        {gpuInfo && (
          <div className="flex items-center gap-1 min-w-0" title={gpuInfo}>
            <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 text-fg-muted" aria-hidden="true">
              <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="8" y="8" width="8" height="8" rx="1"/>
            </svg>
            {/* The full name is in the tooltip; the vendor prefix only takes room */}
            <span className="text-[11px] tabular-nums whitespace-nowrap truncate text-fg-muted">{gpuInfo.replace(/^NVIDIA (GeForce )?|^AMD (Radeon )?/, "")}</span>
          </div>
        )}
        <RunControls onShowRun={onShowRun} />
        <IconButton label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} onClick={toggleTheme} className="text-fg-muted">
          {theme === "dark" ? ICONS.sun : ICONS.moon}
        </IconButton>
      </div>
    </div>
  );
}

function ProjectNameInput() {
  const projectName = useGraphStore((s) => s.projectName);
  const dirty = useDocumentStore((s) => s.dirty);
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
        aria-label="Project name"
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") commit(); if (e.key === "Escape") { e.stopPropagation(); setDraft(projectName); setEditing(false); } }}
        className="px-1.5 h-7 text-xs font-semibold rounded-md w-[200px] min-w-0 bg-transparent border border-accent text-fg outline-none"
      />
    );
  }

  return (
    <button
      onClick={() => setEditing(true)}
      className="flex items-center px-1.5 h-7 text-xs font-semibold rounded-md min-w-0 text-fg hover:bg-surface-raised transition-colors"
      title={dirty ? "Unsaved changes. Click to rename the project." : "Click to rename the project"}
    >
      <span className="truncate">{projectName}</span>
      {dirty && <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full ml-1.5 bg-fg-secondary" aria-label="Unsaved changes" />}
    </button>
  );
}

type MenuItem = { label?: string; shortcut?: string; action?: () => void; disabled?: boolean; type?: "separator" };

const s = { width: 13, height: 13, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
const ICONS = {
  sun: <svg {...s}><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>,
  moon: <svg {...s}><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>,
};
