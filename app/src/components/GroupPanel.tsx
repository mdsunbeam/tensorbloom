import { useEffect, useMemo, useState } from "react";
import { useGraphStore } from "../store/graph-store";
import { useHistoryStore } from "../store/history-store";
import { formatParamCount } from "../lib/model-summary";
import { quickParamCount } from "../lib/node-summary";
import { focusNode } from "../lib/canvas-focus";
import { CATEGORY_COLORS } from "../lib/registry";
import type { LayerGroup } from "../lib/groups";
import Button from "./ui/Button";

/** The Properties panel for a selected group: its name, its layers, and what can be done with it */
export default function GroupPanel({ group }: { group: LayerGroup }) {
  const nodes = useGraphStore((s) => s.nodes);
  const [name, setName] = useState(group.name);
  useEffect(() => setName(group.name), [group.id, group.name]);

  const members = useMemo(() => {
    const byId = new Map(nodes.map((n) => [n.id, n]));
    return group.nodeIds.map((id) => byId.get(id)).filter((n): n is NonNullable<typeof n> => !!n);
  }, [nodes, group.nodeIds]);
  const params = members.reduce((sum, n) => sum + quickParamCount(n.data.moduleType, n.data.params), 0);

  const act = (fn: () => void) => () => { useHistoryStore.getState().pushSnapshot(); fn(); };
  const commitName = () => {
    if (name.trim() && name.trim() !== group.name) act(() => useGraphStore.getState().renameGroup(group.id, name))();
    else setName(group.name);
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="px-4 py-3 border-b border-line">
        <label htmlFor="group-name" className="block text-[11px] font-medium mb-1 text-fg-secondary">Group name</label>
        <input
          id="group-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={commitName}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") { setName(group.name); (e.target as HTMLInputElement).blur(); } }}
          className="w-full px-2.5 h-8 text-xs rounded-lg bg-surface-raised border border-line text-fg outline-none focus:border-accent"
        />
        <div className="mt-2 text-[11px] text-fg-muted tabular-nums">
          {members.length} layer{members.length === 1 ? "" : "s"}{params > 0 && <> · {formatParamCount(params)} params</>}
        </div>
      </div>

      <div className="px-4 py-3 border-b border-line flex flex-wrap gap-2">
        <Button size="sm" onClick={act(() => useGraphStore.getState().setGroupCollapsed(group.id, !group.collapsed))}>
          {group.collapsed ? "Expand" : "Collapse"}
        </Button>
        <Button size="sm" onClick={act(() => useGraphStore.getState().ungroup(group.id))} title="Ctrl+Shift+G">Ungroup</Button>
      </div>

      <div className="px-4 py-3 flex-1">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-fg-muted mb-1.5">Layers</div>
        <ul className="space-y-0.5">
          {members.map((n) => (
            <li key={n.id}>
              <button
                type="button"
                onClick={() => focusNode(n.id)}
                className="w-full flex items-center gap-2 px-2 py-1 rounded-md text-left text-xs hover:bg-surface-raised"
              >
                <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: CATEGORY_COLORS[n.data.category] }} aria-hidden="true" />
                <span className="truncate text-fg">{n.data.label}</span>
                <span className="ml-auto flex-shrink-0 text-[11px] text-fg-muted">{n.data.moduleType}</span>
              </button>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-[11px] leading-relaxed text-fg-muted">
          Groups only organize the canvas. The model, its code and its training are the same with or without them.
        </p>
      </div>

      <div className="px-4 py-3 border-t border-line">
        <Button variant="danger" size="sm" block onClick={act(() => useGraphStore.getState().deleteMultiple(group.nodeIds))}>
          Delete group and its layers
        </Button>
      </div>
    </div>
  );
}
