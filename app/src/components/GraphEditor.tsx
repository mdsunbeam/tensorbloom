import { useCallback, useRef, useEffect, useState, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  BackgroundVariant,
  useReactFlow,
  useViewport,
  useStore,
  type ReactFlowInstance,
} from "@xyflow/react";
import { useGraphStore, type NodeData } from "../store/graph-store";
import { useTrainingStore } from "../store/training-store";
import { CATEGORY_COLORS } from "../lib/registry";
import { inferShapesClient } from "../lib/shape-inference";
import { isValidConnection as checkValid } from "../lib/connection-validation";
import { endDrag, getDragState } from "./DragOverlay";
import TensorBloomNode from "../nodes/TensorBloomNode";
import ShapeEdge from "./ShapeEdge";
import ContextMenu from "./ContextMenu";
import type { Node, Edge, Connection } from "@xyflow/react";
import type { ConnectFromContext } from "./SpotlightDialog";

const nodeTypes = { tensorLabNode: TensorBloomNode };

// Loaded models open at a zoom where node text is readable, from the top, rather than shrunk to fit
const READABLE_ZOOM = 0.75;
const VIEW_PAD = 40;
const edgeTypes = { smoothstep: ShapeEdge };

function BatchToggleButton() {
  const shapeDisplayBatch = useGraphStore((s) => s.shapeDisplayBatch);
  const toggleShapeDisplayBatch = useGraphStore((s) => s.toggleShapeDisplayBatch);
  const batchSize = useTrainingStore((s) => s.config.batchSize);
  const label = shapeDisplayBatch ? `N=${batchSize}` : "N=1";
  return (
    <div className="absolute bottom-2.5 left-14 z-10">
      <button
        onClick={toggleShapeDisplayBatch}
        className="px-2.5 py-1 text-[10px] font-mono font-medium rounded-lg"
        style={{
          background: shapeDisplayBatch ? "var(--accent-subtle)" : "var(--bg-secondary)",
          color: shapeDisplayBatch ? "var(--accent-text)" : "var(--text-muted)",
          border: `1px solid ${shapeDisplayBatch ? "var(--accent)" : "var(--border)"}`,
          boxShadow: "var(--node-shadow)",
        }}
        title={`Shape batch dimension: ${label}. Click to toggle.`}
      >
        {label}
      </button>
    </div>
  );
}

const MINIMAP_W = 168;
const MINIMAP_H = 112;
const MINIMAP_PAD = 50;

function CustomMinimap({ visible }: { visible: boolean }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef(false);
  const nodes = useGraphStore((s) => s.nodes);
  const { setCenter, setViewport, getViewport } = useReactFlow();
  const viewport = useViewport();
  const rfWidth = useStore((s) => s.width);
  const rfHeight = useStore((s) => s.height);
  const [hovered, setHovered] = useState(false);

  const bounds = useMemo(() => {
    if (nodes.length === 0) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of nodes) {
      const w = n.measured?.width ?? 180;
      const h = n.measured?.height ?? 80;
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y);
      maxX = Math.max(maxX, n.position.x + w);
      maxY = Math.max(maxY, n.position.y + h);
    }
    return {
      minX: minX - MINIMAP_PAD,
      minY: minY - MINIMAP_PAD,
      width: Math.max(1, maxX - minX + MINIMAP_PAD * 2),
      height: Math.max(1, maxY - minY + MINIMAP_PAD * 2),
    };
  }, [nodes]);

  const scale = bounds ? Math.min(MINIMAP_W / bounds.width, MINIMAP_H / bounds.height) : 1;
  const offsetX = bounds ? (MINIMAP_W - bounds.width * scale) / 2 : 0;
  const offsetY = bounds ? (MINIMAP_H - bounds.height * scale) / 2 : 0;

  const toFlowCoords = useCallback((clientX: number, clientY: number) => {
    if (!svgRef.current || !bounds) return null;
    const rect = svgRef.current.getBoundingClientRect();
    return {
      x: (clientX - rect.left - offsetX) / scale + bounds.minX,
      y: (clientY - rect.top - offsetY) / scale + bounds.minY,
    };
  }, [bounds, scale, offsetX, offsetY]);

  const handlePointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    e.preventDefault();
    svgRef.current?.setPointerCapture(e.pointerId);
    const pt = toFlowCoords(e.clientX, e.clientY);
    if (pt) setCenter(pt.x, pt.y, { zoom: getViewport().zoom, duration: 150 });
    dragRef.current = true;
  }, [toFlowCoords, setCenter, getViewport]);

  const handlePointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragRef.current) return;
    const pt = toFlowCoords(e.clientX, e.clientY);
    if (pt) {
      const z = getViewport().zoom;
      setViewport({ x: -pt.x * z + rfWidth / 2, y: -pt.y * z + rfHeight / 2, zoom: z });
    }
  }, [toFlowCoords, getViewport, setViewport, rfWidth, rfHeight]);

  const handlePointerUp = useCallback(() => { dragRef.current = false; }, []);

  // Scroll-to-zoom
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !visible) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const vp = getViewport();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.15, Math.min(2.5, vp.zoom * factor));
      const cx = rfWidth / 2;
      const cy = rfHeight / 2;
      setViewport({
        x: cx - (cx - vp.x) * (newZoom / vp.zoom),
        y: cy - (cy - vp.y) * (newZoom / vp.zoom),
        zoom: newZoom,
      });
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, [visible, getViewport, setViewport, rfWidth, rfHeight]);

  if (!visible || !bounds) return null;

  // Viewport indicator
  const flowLeft = -viewport.x / viewport.zoom;
  const flowTop = -viewport.y / viewport.zoom;

  // Nothing to navigate to when the whole model is already on screen
  const allOnScreen =
    flowLeft <= bounds.minX + MINIMAP_PAD &&
    flowTop <= bounds.minY + MINIMAP_PAD &&
    flowLeft + rfWidth / viewport.zoom >= bounds.minX + bounds.width - MINIMAP_PAD &&
    flowTop + rfHeight / viewport.zoom >= bounds.minY + bounds.height - MINIMAP_PAD;
  if (allOnScreen) return null;
  const vpX = (flowLeft - bounds.minX) * scale + offsetX;
  const vpY = (flowTop - bounds.minY) * scale + offsetY;
  const vpW = (rfWidth / viewport.zoom) * scale;
  const vpH = (rfHeight / viewport.zoom) * scale;

  return (
    <div
      className="absolute bottom-3 right-3 z-10"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: MINIMAP_W,
        height: MINIMAP_H,
        background: "var(--bg-secondary)",
        border: "1px solid var(--border)",
        borderRadius: 12,
        boxShadow: hovered
          ? "0 2px 6px rgba(0,0,0,0.15), 0 8px 24px rgba(0,0,0,0.12)"
          : "0 1px 3px rgba(0,0,0,0.12), 0 4px 16px rgba(0,0,0,0.08)",
        opacity: hovered ? 1 : 0.85,
        transition: "opacity 0.2s ease, box-shadow 0.2s ease",
        overflow: "hidden",
      }}
    >
      <svg
        ref={svgRef}
        width={MINIMAP_W}
        height={MINIMAP_H}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        style={{ cursor: "pointer", display: "block" }}
      >
        {/* Nodes */}
        {nodes.map((node) => {
          const nw = Math.max(3, (node.measured?.width ?? 180) * scale);
          const nh = Math.max(2, (node.measured?.height ?? 80) * scale);
          const nx = (node.position.x - bounds.minX) * scale + offsetX;
          const ny = (node.position.y - bounds.minY) * scale + offsetY;
          const c = CATEGORY_COLORS[(node.data as NodeData).category] ?? "var(--text-muted)";
          const sel = !!node.selected;
          return (
            <g key={node.id}>
              {sel && (
                <rect
                  x={nx - 2} y={ny - 2}
                  width={nw + 4} height={nh + 4}
                  rx={4} ry={4}
                  fill="none" style={{ stroke: c }} strokeWidth={1.5} strokeOpacity={0.3}
                />
              )}
              <rect
                x={nx} y={ny}
                width={nw} height={nh}
                rx={3} ry={3}
                style={{ fill: c }}
                fillOpacity={sel ? 0.85 : 0.55}
              />
            </g>
          );
        })}
        {/* Mask — dims area outside viewport */}
        <path
          d={`M0,0H${MINIMAP_W}V${MINIMAP_H}H0Z M${vpX},${vpY}h${vpW}v${vpH}h${-vpW}Z`}
          fillRule="evenodd"
          fill="var(--minimap-mask)"
          pointerEvents="none"
        />
        {/* Viewport border */}
        <rect
          x={vpX} y={vpY}
          width={vpW} height={vpH}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={1.5}
          strokeOpacity={0.5}
          rx={2}
          pointerEvents="none"
        />
      </svg>
    </div>
  );
}

interface GraphEditorProps {
  onShowSpotlight: (connectFrom?: ConnectFromContext) => void;
  minimapVisible: boolean;
}

export default function GraphEditor({ onShowSpotlight, minimapVisible }: GraphEditorProps) {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const onNodesChange = useGraphStore((s) => s.onNodesChange);
  const onEdgesChange = useGraphStore((s) => s.onEdgesChange);
  const onConnectStore = useGraphStore((s) => s.onConnect);
  const onConnect = useCallback((params: Parameters<typeof onConnectStore>[0]) => {
    connectSucceededRef.current = true;
    onConnectStore(params);
  }, [onConnectStore]);
  const addNode = useGraphStore((s) => s.addNode);
  const setSelectedNodeId = useGraphStore((s) => s.setSelectedNodeId);
  const trainingStatus = useTrainingStore((s) => s.status);
  const reactFlowInstance = useRef<ReactFlowInstance<Node<NodeData>> | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; nodeId: string | null } | null>(null);
  const prevNodeCount = useRef(nodes.length);
  const pendingConnectRef = useRef<{ nodeId: string; handleId: string } | null>(null);
  const connectSucceededRef = useRef(false);

  // Animate edges during training
  const displayEdges = useMemo(() => {
    if (trainingStatus !== "running") return edges;
    return edges.map((e) => ({ ...e, animated: true }));
  }, [edges, trainingStatus]);

  // Debounced shape inference
  useEffect(() => {
    const timer = setTimeout(() => {
      if (nodes.length === 0) {
        useGraphStore.getState().setShapeMap({});
        return;
      }
      const { shapes, errors } = inferShapesClient(nodes, edges);
      useGraphStore.getState().setShapeMap(shapes);
      useGraphStore.getState().setShapeErrors(errors);
    }, 200);
    return () => clearTimeout(timer);
  }, [nodes, edges]);

  // Warn if graph is very large
  useEffect(() => {
    if (nodes.length > 200) {
      console.warn(`[TensorBloom] Graph has ${nodes.length} nodes. Consider using sub-graphs or custom blocks for better performance.`);
    }
  }, [nodes.length]);

  // Show the whole model if it fits at a readable zoom; otherwise show its top at that zoom
  const fitReadable = useCallback(() => {
    const rf = reactFlowInstance.current;
    const el = wrapperRef.current;
    if (!rf || !el) return;
    const ns = rf.getNodes();
    if (ns.length === 0) return;
    const b = rf.getNodesBounds(ns);
    const { width: W, height: H } = el.getBoundingClientRect();
    const fit = Math.min((W - VIEW_PAD * 2) / b.width, (H - VIEW_PAD * 2) / b.height);
    if (fit >= READABLE_ZOOM) {
      rf.fitView({ padding: 0.15, maxZoom: 1, duration: 300 });
      return;
    }
    const z = READABLE_ZOOM;
    const x = b.width * z <= W - VIEW_PAD * 2 ? W / 2 - (b.x + b.width / 2) * z : VIEW_PAD - b.x * z;
    const y = b.height * z <= H - VIEW_PAD * 2 ? H / 2 - (b.y + b.height / 2) * z : VIEW_PAD - b.y * z;
    rf.setViewport({ x, y, zoom: z }, { duration: 300 });
  }, []);

  // Re-frame when nodes are bulk-loaded
  useEffect(() => {
    const jumped = nodes.length > 0 && prevNodeCount.current === 0;
    const bulkLoad = Math.abs(nodes.length - prevNodeCount.current) > 3;
    prevNodeCount.current = nodes.length;
    if (jumped || bulkLoad) setTimeout(fitReadable, 50);
  }, [nodes.length, fitReadable]);

  // Handle drop from custom drag overlay
  const onMouseUp = useCallback((e: React.MouseEvent) => {
    const drag = getDragState();
    if (!drag || !drag.active) return;
    const dropped = endDrag();
    if (!dropped || !reactFlowInstance.current) return;
    const raw = reactFlowInstance.current.screenToFlowPosition({ x: e.clientX, y: e.clientY });
    const position = { x: Math.round(raw.x / 20) * 20, y: Math.round(raw.y / 20) * 20 };
    addNode(dropped.moduleType, position);
  }, [addNode]);

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node<NodeData>) => {
    setSelectedNodeId(node.id);
  }, [setSelectedNodeId]);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setCtxMenu(null);
  }, [setSelectedNodeId]);

  const validateConnection = useCallback((conn: Edge | Connection) => {
    const c: Connection = {
      source: conn.source,
      target: conn.target,
      sourceHandle: conn.sourceHandle ?? null,
      targetHandle: conn.targetHandle ?? null,
    };
    const { nodes, edges } = useGraphStore.getState();
    return checkValid(c, nodes, edges);
  }, []);

  // Track connection start for quick-add
  const onConnectStart = useCallback((_: unknown, params: { nodeId: string | null; handleId: string | null; handleType: string | null }) => {
    if (params.handleType === "source" && params.nodeId && params.handleId) {
      pendingConnectRef.current = { nodeId: params.nodeId, handleId: params.handleId };
    }
  }, []);

  // Quick-add: when drag from output handle ends on empty canvas
  const onConnectEnd = useCallback((_event: MouseEvent | TouchEvent) => {
    const pending = pendingConnectRef.current;
    pendingConnectRef.current = null;
    if (!pending) return;

    // If onConnect already fired, the connection succeeded — don't open spotlight
    if (connectSucceededRef.current) {
      connectSucceededRef.current = false;
      return;
    }
    connectSucceededRef.current = false;

    // Connection dropped on empty canvas — open spotlight with connect context
    const sourceNode = useGraphStore.getState().nodes.find((n) => n.id === pending.nodeId);
    if (!sourceNode) return;

    onShowSpotlight({
      sourceNodeId: pending.nodeId,
      sourceHandleId: pending.handleId,
      sourceLabel: sourceNode.data.label,
    });
  }, [onShowSpotlight]);

  const onNodeContextMenu = useCallback((e: React.MouseEvent, node: Node<NodeData>) => {
    e.preventDefault();
    setSelectedNodeId(node.id);
    setCtxMenu({ x: e.clientX, y: e.clientY, nodeId: node.id });
  }, [setSelectedNodeId]);

  const onPaneContextMenu = useCallback((e: MouseEvent | React.MouseEvent) => {
    e.preventDefault();
    if ("clientX" in e) {
      setCtxMenu({ x: e.clientX, y: e.clientY, nodeId: null });
    }
  }, []);

  return (
    <div
      ref={wrapperRef}
      className="w-full h-full relative"
      onMouseUp={onMouseUp}
    >
      <ReactFlow
        nodes={nodes} edges={displayEdges}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}
        onNodeClick={onNodeClick} onPaneClick={onPaneClick}
        onNodeContextMenu={onNodeContextMenu}
        onPaneContextMenu={onPaneContextMenu}
        onConnectStart={onConnectStart}
        onConnectEnd={onConnectEnd}
        isValidConnection={validateConnection}
        onInit={(instance) => {
          reactFlowInstance.current = instance as ReactFlowInstance<Node<NodeData>>;
          // A graph restored from autosave is already in the store; frame it once its nodes are measured
          setTimeout(fitReadable, 100);
        }}
        nodeTypes={nodeTypes} edgeTypes={edgeTypes}
        snapToGrid snapGrid={[20, 20]}
        minZoom={0.2}
        selectionKeyCode="Shift"
        multiSelectionKeyCode={["Shift", "Control", "Meta"]}
        panOnDrag
        panOnScroll
        defaultEdgeOptions={{
          type: "smoothstep",
          animated: false,
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: "var(--edge-color)" },
        }}
        proOptions={{ hideAttribution: true }}
        deleteKeyCode="Delete"
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1.2} color="var(--bg-dot)" />
        <Controls showInteractive={false} />
      </ReactFlow>

      {/* Custom minimap */}
      <CustomMinimap visible={minimapVisible} />

      {/* Context menu */}
      {ctxMenu && (
        <ContextMenu
          x={ctxMenu.x} y={ctxMenu.y} nodeId={ctxMenu.nodeId}
          onClose={() => setCtxMenu(null)}
          onShowSpotlight={() => onShowSpotlight()}
        />
      )}

      {/* Batch size display toggle */}
      <BatchToggleButton />

    </div>
  );
}
