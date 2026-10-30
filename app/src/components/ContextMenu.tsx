import { useEffect, useRef } from "react";
import { useGraphStore } from "../store/graph-store";
import { useHistoryStore } from "../store/history-store";
import { alignLeft, alignCenter, alignTop, distributeVertically } from "../lib/align";
import { groupSelection } from "../lib/document-actions";

interface ContextMenuProps {
  x: number;
  y: number;
  nodeId: string | null;
  /** Set when the menu was opened on a group's frame or card */
  groupId?: string | null;
  onClose: () => void;
  onShowSpotlight: () => void;
}

export default function ContextMenu({ x, y, nodeId, groupId, onClose, onShowSpotlight }: ContextMenuProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as HTMLElement)) onClose();
    };
    const handleEsc = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("mousedown", handleClick);
    window.addEventListener("keydown", handleEsc);
    return () => {
      window.removeEventListener("mousedown", handleClick);
      window.removeEventListener("keydown", handleEsc);
    };
  }, [onClose]);

  const selectedIds = new Set(
    useGraphStore.getState().nodes.filter((n) => n.selected).map((n) => n.id),
  );
  if (nodeId) selectedIds.add(nodeId);
  const multiSelected = selectedIds.size > 1;

  const deleteNode = () => {
    if (nodeId) {
      useGraphStore.getState().setSelectedNodeId(nodeId);
      useHistoryStore.getState().pushSnapshot();
      useGraphStore.getState().deleteSelected();
    }
    onClose();
  };

  const duplicateNode = () => {
    if (nodeId) {
      useGraphStore.getState().setSelectedNodeId(nodeId);
      useHistoryStore.getState().pushSnapshot();
      useGraphStore.getState().duplicateSelected();
    }
    onClose();
  };

  const copyNode = () => {
    if (nodeId) useGraphStore.getState().setSelectedNodeId(nodeId);
    useGraphStore.getState().copySelected();
    onClose();
  };

  const doAlign = (fn: typeof alignLeft) => {
    useHistoryStore.getState().pushSnapshot();
    const { nodes } = useGraphStore.getState();
    const { edges, groups } = useGraphStore.getState();
    useGraphStore.getState().loadGraph(fn(nodes, selectedIds), edges, groups);
    onClose();
  };

  const group = groupId ? useGraphStore.getState().groups.find((g) => g.id === groupId) : undefined;
  const groupAction = (fn: () => void) => () => { useHistoryStore.getState().pushSnapshot(); fn(); onClose(); };

  return (
    <div
      ref={ref}
      className="fixed z-50 w-48 rounded-lg shadow-xl overflow-hidden py-1"
      style={{ left: x, top: y, background: "var(--bg-secondary)", border: "1px solid var(--border)" }}
    >
      {group ? (
        <>
          <MenuItem label="Rename" onClick={() => { useGraphStore.getState().setRenamingGroupId(group.id); onClose(); }} />
          <MenuItem label={group.collapsed ? "Expand" : "Collapse"} onClick={groupAction(() => useGraphStore.getState().setGroupCollapsed(group.id, !group.collapsed))} />
          <MenuDivider />
          <MenuItem label="Ungroup" shortcut="Ctrl+Shift+G" onClick={groupAction(() => useGraphStore.getState().ungroup(group.id))} />
          <MenuItem label="Delete Group and Layers" danger onClick={groupAction(() => useGraphStore.getState().deleteMultiple(group.nodeIds))} />
        </>
      ) : nodeId ? (
        <>
          <MenuItem label="Copy" shortcut="Ctrl+C" onClick={copyNode} />
          <MenuItem label="Duplicate" shortcut="Ctrl+D" onClick={duplicateNode} />
          <MenuItem label="Delete" shortcut="Del" onClick={deleteNode} danger />
          {multiSelected && (
            <>
              <MenuDivider />
              <MenuItem label="Align Left" onClick={() => doAlign(alignLeft)} />
              <MenuItem label="Align Center" onClick={() => doAlign(alignCenter)} />
              <MenuItem label="Align Top" onClick={() => doAlign(alignTop)} />
              <MenuItem label="Distribute Vertically" onClick={() => doAlign(distributeVertically)} />
            </>
          )}
          {multiSelected && (
            <>
              <MenuDivider />
              <MenuItem label="Group Selection" shortcut="Ctrl+G" onClick={() => { groupSelection(); onClose(); }} />
            </>
          )}
          <MenuDivider />
          <MenuItem label="Add Layer..." shortcut="Ctrl+K" onClick={() => { onShowSpotlight(); onClose(); }} />
        </>
      ) : (
        <>
          <MenuItem label="Add Layer..." shortcut="Ctrl+K" onClick={() => { onShowSpotlight(); onClose(); }} />
          <MenuItem label="Paste" shortcut="Ctrl+V" onClick={() => {
            useHistoryStore.getState().pushSnapshot();
            useGraphStore.getState().paste();
            onClose();
          }} disabled={!useGraphStore.getState().clipboard} />
          <MenuItem label="Select All" shortcut="Ctrl+A" onClick={() => {
            useGraphStore.getState().selectAll();
            onClose();
          }} />
        </>
      )}
    </div>
  );
}

function MenuItem({ label, shortcut, onClick, danger, disabled }: {
  label: string; shortcut?: string; onClick: () => void; danger?: boolean; disabled?: boolean;
}) {
  return (
    <button
      onClick={disabled ? undefined : onClick}
      className="flex items-center w-full px-3 py-1.5 text-xs transition-colors"
      style={{
        color: disabled ? "var(--text-muted)" : danger ? "var(--danger)" : "var(--text-primary)",
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? "default" : "pointer",
      }}
      onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.background = "var(--bg-tertiary)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
    >
      <span className="flex-1 text-left">{label}</span>
      {shortcut && <span className="text-[11px] ml-3" style={{ color: "var(--text-muted)" }}>{shortcut}</span>}
    </button>
  );
}

function MenuDivider() {
  return <div className="my-1 mx-2" style={{ borderTop: "1px solid var(--border)" }} />;
}
