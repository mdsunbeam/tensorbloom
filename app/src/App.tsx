import { useState, useCallback, useRef, useEffect, lazy, Suspense } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import GraphEditor from "./components/GraphEditor";
import NodePalette from "./components/NodePalette";
import PropertyPanel from "./components/PropertyPanel";
import CodePanel from "./components/CodePanel";
import TrainingCharts from "./components/TrainingCharts";
import TrainingLogs from "./components/TrainingLogs";
import Toolbar from "./components/Toolbar";
import ToastContainer, { toast } from "./components/Toast";
import PythonSetup from "./components/PythonSetup";
import DragOverlay from "./components/DragOverlay";
import type { ConnectFromContext } from "./components/SpotlightDialog";

// Lazy-loaded dialogs — only loaded when opened
const ImportDialog = lazy(() => import("./components/ImportDialog"));
const TemplatesDialog = lazy(() => import("./components/TemplatesDialog"));
const SpotlightDialog = lazy(() => import("./components/SpotlightDialog"));
const ExportDialog = lazy(() => import("./components/ExportDialog"));
const ShortcutsDialog = lazy(() => import("./components/ShortcutsDialog"));
const SweepDialog = lazy(() => import("./components/SweepDialog"));
const CompareDialog = lazy(() => import("./components/CompareDialog"));
const CustomBlockDialog = lazy(() => import("./components/CustomBlockDialog"));
import { useThemeStore } from "./store/theme-store";
import { useGraphStore } from "./store/graph-store";
import { useHistoryStore } from "./store/history-store";
import { useTrainingStore } from "./store/training-store";
import { downloadProject, loadProjectFile, deserializeProject } from "./lib/project";
import { autoLayout } from "./lib/auto-layout";
import { TEMPLATES, prependDatasetNode } from "./lib/templates";
import { startAutoSave, restoreAutoSave } from "./lib/auto-save";
import { sidecarCall } from "./lib/sidecar";
import { MODULE_REGISTRY } from "./lib/registry";
import type { NodeCategory, ModuleDef } from "./lib/types";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "./store/graph-store";

type BottomTab = "code" | "charts" | "logs";

export default function App() {
  const theme = useThemeStore((s) => s.theme);
  const trainingStatus = useTrainingStore((s) => s.status);
  const logCount = useTrainingStore((s) => s.logs.length);
  const [tabHeights, setTabHeights] = useState<Record<BottomTab, number>>({ code: 220, charts: 300, logs: 280 });
  const [bottomPanelOpen, setBottomPanelOpen] = useState(true);
  const [bottomTab, setBottomTab] = useState<BottomTab>("code");
  const [importOpen, setImportOpen] = useState(false);
  const [templatesOpen, setTemplatesOpen] = useState(false);
  const [spotlightOpen, setSpotlightOpen] = useState(false);
  const [connectFrom, setConnectFrom] = useState<ConnectFromContext | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [sweepOpen, setSweepOpen] = useState(false);
  const [customBlockOpen, setCustomBlockOpen] = useState(false);
  const [compareOpen, setCompareOpen] = useState(false);
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [minimapVisible, setMinimapVisible] = useState(true);
  const [pythonReady, setPythonReady] = useState(false);
  const [checkingPython, setCheckingPython] = useState(true);
  const [gpuInfo, setGpuInfo] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [compareModel, _setCompareModel] = useState<{ name: string; nodes: Node<NodeData>[] } | null>(null);
  const isDragging = useRef(false);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // Restore custom blocks from localStorage on startup
  useEffect(() => {
    try {
      const saved = localStorage.getItem("tensorlab-custom-blocks");
      if (saved) {
        const blocks = JSON.parse(saved);
        for (const [name, block] of Object.entries(blocks)) {
          const b = block as { params: ModuleDef["params"]; category: string };
          if (!MODULE_REGISTRY[name]) {
            MODULE_REGISTRY[name] = {
              type: name,
              label: name,
              category: b.category as NodeCategory,
              description: `Custom block: ${name}`,
              params: b.params,
              inputHandles: [{ id: "in", label: "in", dataType: "tensor" }],
              outputHandles: [{ id: "out", label: "out", dataType: "tensor" }],
            };
          }
        }
      }
    } catch { /* ignore */ }
  }, []);

  // Check Python environment on startup
  useEffect(() => {
    (async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const status = await invoke<{ ready: boolean }>("check_python_env");
        if (status.ready) {
          setPythonReady(true);
        }
      } catch {
        // Browser mode or Tauri not available — skip setup screen
        setPythonReady(true);
      }
      setCheckingPython(false);
    })();
  }, []);

  // Detect GPU via nvidia-smi (no sidecar dependency)
  useEffect(() => {
    (async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const result = await invoke<{ name: string; memory_total_mb: number } | null>("detect_gpu");
        if (result) {
          const mem = result.memory_total_mb ? ` ${Math.round(result.memory_total_mb / 1024)}GB` : "";
          setGpuInfo(`${result.name}${mem}`);
        } else {
          setGpuInfo("CPU only");
        }
      } catch {
        // Browser mode or no Tauri — skip GPU detection
      }
    })();
  }, []);

  // Auto-save + restore
  useEffect(() => {
    const restored = restoreAutoSave();
    if (restored) toast.info("Restored auto-saved project");
    const stopAutoSave = startAutoSave();
    return stopAutoSave;
  }, []);

  // Window title
  useEffect(() => {
    const name = useGraphStore.getState().projectName;
    document.title = `${name} - TensorBloom`;
    const unsub = useGraphStore.subscribe((s) => {
      document.title = `${s.projectName} - TensorBloom`;
    });
    return unsub;
  }, []);

  // Auto-switch to training tab when training starts
  useEffect(() => {
    if (trainingStatus === "running" || trainingStatus === "starting") {
      setBottomTab("charts");
      setBottomPanelOpen(true);
    }
  }, [trainingStatus]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey;
      // Let browser handle clipboard/undo shortcuts when focus is in a text field
      const tag = (e.target as HTMLElement)?.tagName;
      const editable = tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable;
      if (editable && ctrl && ["a", "c", "v", "x", "z", "y"].includes(e.key)) return;
      if (ctrl && e.key === "z" && !e.shiftKey) { e.preventDefault(); useHistoryStore.getState().undo(); }
      else if (ctrl && e.key === "z" && e.shiftKey) { e.preventDefault(); useHistoryStore.getState().redo(); }
      else if (ctrl && e.key === "y") { e.preventDefault(); useHistoryStore.getState().redo(); }
      else if (ctrl && e.key === "s") { e.preventDefault(); const s = useGraphStore.getState(); downloadProject(s.projectName, s.nodes, s.edges).then((r) => { if (r !== null) toast.success("Project saved"); }); }
      else if (ctrl && e.key === "o") { e.preventDefault(); handleOpenFile(); }
      else if (ctrl && e.key === "n") { e.preventDefault(); useHistoryStore.getState().pushSnapshot(); useGraphStore.getState().clearGraph(); useGraphStore.getState().setProjectName("Untitled"); }
      else if (ctrl && e.key === "i") { e.preventDefault(); setImportOpen(true); }
      else if (ctrl && e.key === "l") { e.preventDefault(); handleAutoLayout(); }
      else if (ctrl && e.key === "k") { e.preventDefault(); setSpotlightOpen(true); }
      else if (ctrl && e.key === "e") { e.preventDefault(); setExportOpen(true); }
      else if (ctrl && e.key === "d") { e.preventDefault(); useHistoryStore.getState().pushSnapshot(); useGraphStore.getState().duplicateSelected(); }
      else if (ctrl && e.key === "c") { useGraphStore.getState().copySelected(); }
      else if (ctrl && e.key === "v") { e.preventDefault(); useHistoryStore.getState().pushSnapshot(); useGraphStore.getState().paste(); }
      else if (ctrl && e.key === "a") { e.preventDefault(); useGraphStore.getState().selectAll(); }
      else if (ctrl && e.key === "g") {
        e.preventDefault();
        const name = prompt("Group name:", "Block") ?? "Block";
        if (name) { useHistoryStore.getState().pushSnapshot(); useGraphStore.getState().groupSelected(name); }
      }
      else if (ctrl && e.key === "b") { e.preventDefault(); setLeftPanelOpen((p) => !p); }
      else if (ctrl && e.key === ".") { e.preventDefault(); setRightPanelOpen((p) => !p); }
      else if (ctrl && e.key === "m") { e.preventDefault(); setMinimapVisible((v) => !v); }
      else if (ctrl && (e.key === "/" || e.key === "?")) { e.preventDefault(); setShortcutsOpen(true); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Undo snapshots on structural changes
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    const unsub = useGraphStore.subscribe((state, prev) => {
      if (state.nodes.length !== prev.nodes.length || state.edges.length !== prev.edges.length) {
        clearTimeout(timer);
        timer = setTimeout(() => useHistoryStore.getState().pushSnapshot(), 300);
      }
    });
    return () => { unsub(); clearTimeout(timer); };
  }, []);

  const handleOpenFile = useCallback(async () => {
    const project = await loadProjectFile();
    if (!project) return;
    const { nodes, edges, warnings } = deserializeProject(project);
    useHistoryStore.getState().pushSnapshot();
    useGraphStore.getState().loadGraph(nodes, edges);
    useGraphStore.getState().setProjectName(project.name);
    for (const w of warnings) toast.warning(w);
    toast.success(`Opened "${project.name}"`);
  }, []);

  const handleAutoLayout = useCallback(() => {
    const { nodes, edges } = useGraphStore.getState();
    if (nodes.length === 0) return;
    useHistoryStore.getState().pushSnapshot();
    useGraphStore.getState().loadGraph(autoLayout(nodes, edges), edges);
  }, []);

  const handleImport = useCallback(async (code: string) => {
    try {
      const result = await sidecarCall<{
        nodes: Array<{ id: string; type: string; moduleType?: string; label: string; position: { x: number; y: number }; params: Record<string, unknown> }>;
        edges: Array<{ id: string; source: string; target: string; sourceHandle: string; targetHandle: string }>;
      }>("compile", { code });

      const nodes: Node<NodeData>[] = result.nodes.map((n) => {
        const mt = n.moduleType ?? "Identity";
        const def = MODULE_REGISTRY[mt];
        return { id: n.id, type: "tensorLabNode", position: n.position,
          data: { moduleType: mt, label: n.label, category: (def?.category ?? "linear") as NodeCategory, params: n.params } };
      });
      const edges: Edge[] = result.edges.map((e) => ({ ...e, type: "smoothstep" }));
      useHistoryStore.getState().pushSnapshot();
      useGraphStore.getState().loadGraph(autoLayout(nodes, edges), edges);
      useGraphStore.getState().setProjectName("Imported Model");
      toast.success("Model imported successfully");
    } catch {
      throw new Error("Import requires the Python sidecar. Start with `npx tauri dev` or run: python python/tensorlab_sidecar.py");
    }
  }, []);

  const handleTemplate = useCallback((index: number) => {
    const tpl = TEMPLATES[index];
    if (!tpl) return;
    const raw = tpl.build();
    const dsType = tpl.trainingDefaults?.dataset?.type;
    const { nodes, edges } = dsType ? prependDatasetNode(raw, dsType, tpl.supervised !== false) : raw;
    useHistoryStore.getState().pushSnapshot();
    useGraphStore.getState().loadGraph(autoLayout(nodes, edges), edges);
    useGraphStore.getState().setProjectName(tpl.name);
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
  }, []);

  const handleSpotlightSelect = useCallback((moduleType: string) => {
    if (connectFrom) {
      // Connected add — create node and edge
      useGraphStore.getState().addNodeConnected(moduleType, connectFrom.sourceNodeId, connectFrom.sourceHandleId);
      setConnectFrom(null);
    } else {
      // Regular add — place near center of viewport
      useGraphStore.getState().addNode(moduleType, {
        x: 200 + Math.random() * 100,
        y: 100 + useGraphStore.getState().nodes.length * 80,
      });
    }
  }, [connectFrom]);

  const onResizeStart = useCallback(() => {
    isDragging.current = true;
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const h = Math.max(100, Math.min(window.innerHeight - e.clientY, window.innerHeight - 200));
      setTabHeights((prev) => ({ ...prev, [bottomTab]: h }));
    };
    const onMouseUp = () => {
      isDragging.current = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, []);

  // Show setup screen if Python not ready
  if (!checkingPython && !pythonReady) {
    return <PythonSetup onReady={() => setPythonReady(true)} />;
  }

  // Show loading while checking
  if (checkingPython) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 h-screen w-screen" style={{ background: "var(--bg-primary)" }}>
        <svg width="48" height="48" viewBox="0 0 36 36" fill="none" style={{ opacity: 0.5 }}>
          <g transform="translate(18,18)">
            <circle cx="0" cy="-10" r="5" fill="var(--text-primary)"/>
            <circle cx="7" cy="-7" r="5" fill="var(--text-primary)"/>
            <circle cx="10" cy="0" r="5" fill="var(--text-primary)"/>
            <circle cx="7" cy="7" r="5" fill="var(--text-primary)"/>
            <circle cx="0" cy="10" r="5" fill="var(--text-primary)"/>
            <circle cx="-7" cy="7" r="5" fill="var(--text-primary)"/>
            <circle cx="-10" cy="0" r="5" fill="var(--text-primary)"/>
            <circle cx="-7" cy="-7" r="5" fill="var(--text-primary)"/>
          </g>
        </svg>
        <div className="text-sm" style={{ color: "var(--text-muted)" }}>Starting TensorBloom...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden" style={{ background: "var(--bg-primary)" }}>
      <Toolbar
        onShowImport={() => setImportOpen(true)}
        onShowTemplates={() => setTemplatesOpen(true)}
        onShowExport={() => setExportOpen(true)}
        onShowSpotlight={() => setSpotlightOpen(true)}
        onShowSweep={() => setSweepOpen(true)}
        onShowCustomBlock={() => setCustomBlockOpen(true)}
        gpuInfo={gpuInfo}
      />

      <ReactFlowProvider>
      <div className="flex flex-1 min-h-0">
        {/* Left panel */}
        <div
          className="flex-shrink-0 relative"
          style={{
            width: leftPanelOpen ? 240 : 36,
            transition: "width 0.2s ease-out",
          }}
        >
          <div className="h-full overflow-hidden" style={{ borderRight: "1px solid var(--border)" }}>
            {leftPanelOpen ? <NodePalette /> : (
              <CollapsedSideBar label="Layers" side="left" onClick={() => setLeftPanelOpen(true)} />
            )}
          </div>
          {leftPanelOpen && (
            <EdgeCollapseButton side="left" onClick={() => setLeftPanelOpen(false)} shortcut="Ctrl+B" />
          )}
        </div>

        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 relative min-h-0">
            <GraphEditor onShowSpotlight={(ctx) => { setConnectFrom(ctx ?? null); setSpotlightOpen(true); }} minimapVisible={minimapVisible} />
          </div>

          {/* Bottom panel bar with tabs */}
          <div
            className="flex items-center h-8 px-2 flex-shrink-0 select-none gap-0.5 relative"
            style={{
              background: "var(--bg-secondary)",
              borderTop: "1px solid var(--border)",
              borderBottom: bottomPanelOpen ? "1px solid var(--border)" : undefined,
              cursor: bottomPanelOpen ? "row-resize" : "default",
            }}
            onMouseDown={(e) => { if (bottomPanelOpen) { e.preventDefault(); onResizeStart(); } }}
          >
            {/* Edge chevron on top border, centered */}
            <button
              onClick={() => setBottomPanelOpen(!bottomPanelOpen)}
              className="absolute left-1/2 -translate-x-1/2 z-10 flex items-center justify-center"
              style={{
                top: -6,
                width: 28,
                height: 12,
                background: "var(--bg-secondary)",
                borderTop: "1px solid var(--border)",
                borderLeft: "1px solid var(--border)",
                borderRight: "1px solid var(--border)",
                borderBottom: "none",
                borderRadius: "3px 3px 0 0",
                color: "var(--text-muted)",
                cursor: "pointer",
              }}
              onMouseDown={(e) => e.stopPropagation()}
              onMouseEnter={(e) => { e.currentTarget.style.background = "var(--bg-tertiary)"; e.currentTarget.style.color = "var(--text-secondary)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "var(--bg-secondary)"; e.currentTarget.style.color = "var(--text-muted)"; }}
            >
              <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
                style={{ transform: bottomPanelOpen ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.15s" }}>
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>

            <BottomTabButton label="Code" active={bottomTab === "code"} onClick={() => { setBottomTab("code"); setBottomPanelOpen(true); }} />
            <BottomTabButton
              label={trainingStatus === "running" ? "Charts..." : trainingStatus === "starting" ? "Starting..." : "Charts"}
              active={bottomTab === "charts"}
              onClick={() => { setBottomTab("charts"); setBottomPanelOpen(true); }}
              accent={trainingStatus === "running" || trainingStatus === "starting"}
            />
            <BottomTabButton
              label={logCount > 0 ? `Logs (${logCount})` : "Logs"}
              active={bottomTab === "logs"}
              onClick={() => { setBottomTab("logs"); setBottomPanelOpen(true); }}
            />

            {bottomPanelOpen && (
              <div className="flex-1 flex justify-center">
                <div className="w-8 h-0.5 rounded-full opacity-30" style={{ background: "var(--text-muted)" }} />
              </div>
            )}
          </div>

          {bottomPanelOpen && (
            <div className="flex-shrink-0 overflow-hidden" style={{ height: tabHeights[bottomTab], background: "var(--bg-secondary)" }}>
              {bottomTab === "code" && <CodePanel />}
              {bottomTab === "charts" && <TrainingCharts onSwitchToLogs={() => setBottomTab("logs")} />}
              {bottomTab === "logs" && <TrainingLogs />}
            </div>
          )}
        </div>

        {/* Right panel */}
        <div
          className="flex-shrink-0 relative"
          style={{
            width: rightPanelOpen ? 288 : 36,
            transition: "width 0.2s ease-out",
          }}
        >
          <div className="h-full overflow-hidden" style={{ borderLeft: "1px solid var(--border)" }}>
            {rightPanelOpen ? <PropertyPanel /> : (
              <CollapsedSideBar label="Properties" side="right" onClick={() => setRightPanelOpen(true)} />
            )}
          </div>
          {rightPanelOpen && (
            <EdgeCollapseButton side="right" onClick={() => setRightPanelOpen(false)} shortcut="Ctrl+." />
          )}
        </div>
      </div>
      </ReactFlowProvider>

      <Suspense fallback={null}>
        {importOpen && <ImportDialog open={importOpen} onClose={() => setImportOpen(false)} onImport={handleImport} />}
        {templatesOpen && <TemplatesDialog open={templatesOpen} onClose={() => setTemplatesOpen(false)} onSelect={handleTemplate} />}
        {spotlightOpen && <SpotlightDialog open={spotlightOpen} onClose={() => { setSpotlightOpen(false); setConnectFrom(null); }} onSelect={handleSpotlightSelect} connectFrom={connectFrom} />}
        {exportOpen && <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} />}
        {shortcutsOpen && <ShortcutsDialog open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />}
        {sweepOpen && <SweepDialog open={sweepOpen} onClose={() => setSweepOpen(false)} onLaunch={(runs) => {
          toast.info(`Sweep launched: ${runs.length} run${runs.length !== 1 ? "s" : ""}`);
        }} />}
        {customBlockOpen && <CustomBlockDialog open={customBlockOpen} onClose={() => setCustomBlockOpen(false)} />}
        {compareOpen && compareModel && <CompareDialog open={compareOpen} onClose={() => setCompareOpen(false)}
          modelA={{ name: useGraphStore.getState().projectName, nodes: useGraphStore.getState().nodes }}
          modelB={compareModel}
        />}
      </Suspense>
      <ToastContainer />
      <DragOverlay />
    </div>
  );
}

function EdgeCollapseButton({ side, onClick, shortcut }: { side: "left" | "right"; onClick: () => void; shortcut: string }) {
  const isLeft = side === "left";
  return (
    <button
      onClick={onClick}
      className="absolute top-1/2 -translate-y-1/2 z-10 flex items-center justify-center"
      style={{
        [isLeft ? "right" : "left"]: -6,
        width: 12,
        height: 28,
        background: "var(--bg-secondary)",
        borderTop: "1px solid var(--border)",
        borderBottom: "1px solid var(--border)",
        [isLeft ? "borderRight" : "borderLeft"]: "1px solid var(--border)",
        [isLeft ? "borderLeft" : "borderRight"]: "none",
        borderRadius: isLeft ? "0 3px 3px 0" : "3px 0 0 3px",
        color: "var(--text-muted)",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "var(--bg-tertiary)"; e.currentTarget.style.color = "var(--text-secondary)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "var(--bg-secondary)"; e.currentTarget.style.color = "var(--text-muted)"; }}
      title={`Collapse (${shortcut})`}
    >
      <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        {isLeft ? <polyline points="15 18 9 12 15 6" /> : <polyline points="9 18 15 12 9 6" />}
      </svg>
    </button>
  );
}

function CollapsedSideBar({ label, side, onClick }: { label: string; side: "left" | "right"; onClick: () => void }) {
  const shortcut = side === "left" ? "Ctrl+B" : "Ctrl+.";
  return (
    <button
      onClick={onClick}
      className="w-full h-full flex flex-col items-center pt-4 gap-3 cursor-pointer group"
      style={{ background: "var(--bg-secondary)" }}
      title={`Expand ${label} (${shortcut})`}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
        className="opacity-60 group-hover:opacity-100" style={{ transition: "opacity 0.15s ease" }}>
        {side === "left" ? (
          <><rect x="3" y="3" width="18" height="18" rx="2" /><line x1="9" y1="3" x2="9" y2="21" /><polyline points="14 10 17 12 14 14" /></>
        ) : (
          <><rect x="3" y="3" width="18" height="18" rx="2" /><line x1="15" y1="3" x2="15" y2="21" /><polyline points="10 10 7 12 10 14" /></>
        )}
      </svg>
      <span
        className="text-[10px] font-semibold uppercase tracking-widest opacity-50 group-hover:opacity-80"
        style={{ color: "var(--text-muted)", writingMode: "vertical-lr", transition: "opacity 0.15s ease" }}
      >
        {label}
      </span>
    </button>
  );
}

function BottomTabButton({ label, active, onClick, accent }: { label: string; active: boolean; onClick: () => void; accent?: boolean }) {
  return (
    <button
      onClick={onClick}
      onMouseDown={(e) => e.stopPropagation()}
      className="relative px-3 py-1 text-[10px] font-semibold uppercase tracking-wider rounded-md"
      style={{
        color: active ? (accent ? "#10b981" : "var(--text-primary)") : "var(--text-muted)",
        background: active ? "var(--bg-tertiary)" : "transparent",
      }}
      onMouseEnter={(e) => {
        if (!active) e.currentTarget.style.color = "var(--text-secondary)";
      }}
      onMouseLeave={(e) => {
        if (!active) e.currentTarget.style.color = "var(--text-muted)";
      }}
    >
      {label}
      {active && (
        <span
          className="absolute bottom-0 left-1/2 -translate-x-1/2 h-[2px] rounded-full"
          style={{ width: "60%", background: accent ? "#10b981" : "var(--accent)" }}
        />
      )}
    </button>
  );
}
