import type { AutoSaveData } from "../lib/auto-save";

function timeAgo(timestamp: number): string {
  const min = Math.round((Date.now() - timestamp) / 60_000);
  if (min < 1) return "just now";
  if (min < 60) return `${min} min ago`;
  const h = Math.round(min / 60);
  if (h < 24) return `${h} hour${h === 1 ? "" : "s"} ago`;
  const d = Math.round(h / 24);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}

/** Offers back unsaved work from the last session, at the top of the canvas */
export default function RecoveryBanner({ data, onRestore, onDiscard }: {
  data: AutoSaveData;
  onRestore: () => void;
  onDiscard: () => void;
}) {
  return (
    <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 pl-4 pr-2 py-2 rounded-xl text-xs"
      role="status"
      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", boxShadow: "var(--node-shadow-hover)" }}>
      <span className="whitespace-nowrap" style={{ color: "var(--text-secondary)" }}>
        Unsaved work from your last session: <span className="font-semibold" style={{ color: "var(--text-primary)" }}>{data.projectName}</span>, {data.nodes.length} layer{data.nodes.length === 1 ? "" : "s"}, {timeAgo(data.timestamp)}
      </span>
      <button onClick={onDiscard} className="px-2.5 py-1 rounded-lg font-medium" style={{ color: "var(--text-secondary)", border: "1px solid var(--border)" }}>
        Discard
      </button>
      <button onClick={onRestore} className="px-2.5 py-1 rounded-lg font-semibold text-white" style={{ background: "var(--accent)" }}>
        Restore
      </button>
    </div>
  );
}
