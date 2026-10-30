import { useMemo, useState, useCallback, useRef } from "react";
import { useGraphStore } from "../store/graph-store";
import { decompileGraph } from "../lib/decompiler";
import { downloadCode } from "../lib/project";
import { sidecarCall } from "../lib/sidecar";
import { autoLayout } from "../lib/auto-layout";
import { useHistoryStore } from "../store/history-store";
import { MODULE_REGISTRY } from "../lib/registry";
import { toast } from "./Toast";
import type { NodeCategory } from "../lib/types";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import Button from "./ui/Button";

type Mode = "preview" | "edit";

export default function CodePanel() {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const [copied, setCopied] = useState(false);
  const [mode, setMode] = useState<Mode>("preview");
  const [editCode, setEditCode] = useState("");
  const [compiling, setCompiling] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { code, errors } = useMemo(
    () => decompileGraph(nodes, edges),
    [nodes, edges],
  );

  const handleCopy = useCallback(async () => {
    const text = mode === "edit" ? editCode : code;
    if (!text) return;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [code, editCode, mode]);

  const handleDownload = useCallback(() => {
    const text = mode === "edit" ? editCode : code;
    if (!text) return;
    downloadCode(text);
  }, [code, editCode, mode]);

  const handleEdit = useCallback(() => {
    setEditCode(code);
    setMode("edit");
    setTimeout(() => textareaRef.current?.focus(), 50);
  }, [code]);

  const handleCancel = useCallback(() => {
    setMode("preview");
    setEditCode("");
  }, []);

  const handleApply = useCallback(async () => {
    if (!editCode.trim()) return;
    setCompiling(true);
    try {
      const result = await sidecarCall<{
        nodes: Array<{ id: string; type: string; moduleType?: string; label: string; position: { x: number; y: number }; params: Record<string, unknown> }>;
        edges: Array<{ id: string; source: string; target: string; sourceHandle: string; targetHandle: string }>;
      }>("compile", { code: editCode });

      const newNodes: Node<NodeData>[] = result.nodes.map((n) => {
        const mt = n.moduleType ?? "Identity";
        const def = MODULE_REGISTRY[mt];
        return {
          id: n.id, type: "tensorLabNode", position: n.position,
          data: { moduleType: mt, label: n.label, category: (def?.category ?? "linear") as NodeCategory, params: n.params },
        };
      });
      const newEdges: Edge[] = result.edges.map((e) => ({ ...e, type: "smoothstep" }));

      useHistoryStore.getState().pushSnapshot();
      useGraphStore.getState().loadGraph(autoLayout(newNodes, newEdges), newEdges);
      setMode("preview");
      setEditCode("");
      toast.success("Code applied to graph");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to compile code");
    } finally {
      setCompiling(false);
    }
  }, [editCode]);

  if (!code && mode === "preview") {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2.5" style={{ background: "var(--bg-tertiary)" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
            </svg>
          </div>
          <div className="text-xs font-medium mb-1" style={{ color: "var(--text-muted)" }}>No code yet</div>
          <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
            Add layers to the graph and PyTorch code will be generated here in real time
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 flex-shrink-0" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
            {mode === "edit" ? "Edit Code" : "Generated PyTorch"}
          </span>
          {mode === "edit" && (
            <span className="text-[11px] px-1.5 py-0.5 rounded" style={{ background: "var(--accent-subtle)", color: "var(--accent-text)" }}>
              editing
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {mode === "preview" ? (
            <>
              <PanelButton onClick={handleEdit}>Edit</PanelButton>
              <PanelButton onClick={handleCopy}>{copied ? "Copied!" : "Copy"}</PanelButton>
              <PanelButton onClick={handleDownload}>Export .py</PanelButton>
            </>
          ) : (
            <>
              <PanelButton onClick={handleApply} accent disabled={compiling}>
                {compiling ? "Compiling..." : "Apply to Graph"}
              </PanelButton>
              <PanelButton onClick={handleCancel}>Cancel</PanelButton>
              <PanelButton onClick={handleCopy}>{copied ? "Copied!" : "Copy"}</PanelButton>
            </>
          )}
        </div>
      </div>

      {/* Errors */}
      {mode === "preview" && errors.length > 0 && (
        <div className="px-3 py-2 text-[11px]" style={{ background: "var(--danger-subtle)", color: "var(--danger)", borderBottom: "1px solid var(--border)" }}>
          {errors.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}

      {/* Content */}
      {mode === "edit" ? (
        <textarea
          ref={textareaRef}
          value={editCode}
          onChange={(e) => setEditCode(e.target.value)}
          spellCheck={false}
          className="flex-1 p-3 text-[11px] leading-[1.6] font-mono resize-none outline-none"
          style={{ background: "transparent", color: "var(--text-primary)", border: "none" }}
        />
      ) : (
        <div className="flex-1 overflow-auto p-3">
          <pre className="text-[11px] leading-[1.6] font-mono whitespace-pre" style={{ color: "var(--text-primary)" }}>
            {highlightPython(code)}
          </pre>
        </div>
      )}
    </div>
  );
}

function PanelButton({ onClick, children, accent, disabled }: {
  onClick: () => void; children: React.ReactNode; accent?: boolean; disabled?: boolean;
}) {
  return (
    <Button size="sm" variant={accent ? "primary" : "secondary"} onClick={onClick} disabled={disabled} className="!h-6 !px-2">
      {children}
    </Button>
  );
}

function highlightPython(code: string): React.ReactNode[] {
  return code.split("\n").map((line, i) => {
    const parts: React.ReactNode[] = [];
    let rest = line;
    let key = 0;
    const kwPattern = /\b(import|from|class|def|self|return|super|True|False|None|as|if|else|for|in|while|pass)\b/g;
    const strPattern = /'[^']*'|"[^"]*"/g;
    const commentIdx = rest.indexOf("#");
    let comment = "";
    if (commentIdx >= 0) { comment = rest.slice(commentIdx); rest = rest.slice(0, commentIdx); }
    const tokens: Array<{ start: number; end: number; type: "kw" | "str" | "num" }> = [];
    let m: RegExpExecArray | null;
    while ((m = kwPattern.exec(rest)) !== null) tokens.push({ start: m.index, end: m.index + m[0].length, type: "kw" });
    while ((m = strPattern.exec(rest)) !== null) tokens.push({ start: m.index, end: m.index + m[0].length, type: "str" });
    const numPattern = /\b\d+\.?\d*(?:e[+-]?\d+)?\b/g;
    while ((m = numPattern.exec(rest)) !== null) tokens.push({ start: m.index, end: m.index + m[0].length, type: "num" });
    tokens.sort((a, b) => a.start - b.start);
    const filtered: typeof tokens = [];
    let lastEnd = 0;
    for (const t of tokens) { if (t.start >= lastEnd) { filtered.push(t); lastEnd = t.end; } }
    let cursor = 0;
    for (const t of filtered) {
      if (t.start > cursor) parts.push(<span key={key++}>{rest.slice(cursor, t.start)}</span>);
      const color = t.type === "kw" ? "var(--accent-text)" : t.type === "str" ? "#10b981" : "#f59e0b";
      parts.push(<span key={key++} style={{ color }}>{rest.slice(t.start, t.end)}</span>);
      cursor = t.end;
    }
    if (cursor < rest.length) parts.push(<span key={key++}>{rest.slice(cursor)}</span>);
    if (comment) parts.push(<span key={key++} style={{ color: "var(--text-muted)" }}>{comment}</span>);
    return <div key={i} className="min-h-[1.6em]">{parts.length > 0 ? parts : "\u00A0"}</div>;
  });
}
