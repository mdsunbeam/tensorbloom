import type { NodeCategory } from "./types";

// Line icons in the Lucide style (24 × 24, 2 px stroke), one per layer category,
// so a layer's kind doesn't depend on telling its color apart
const PATHS: Record<NodeCategory, React.ReactNode> = {
  io: <><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" /><polyline points="10 17 15 12 10 7" /><line x1="15" y1="12" x2="3" y2="12" /></>,
  data: <><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" /></>,
  linear: <><circle cx="5" cy="6" r="2" /><circle cx="5" cy="18" r="2" /><circle cx="19" cy="6" r="2" /><circle cx="19" cy="18" r="2" /><path d="M7 6h10M7 18h10M6.5 7.5l11 9M6.5 16.5l11-9" /></>,
  conv: <><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M3 15h18M9 3v18M15 3v18" /></>,
  pool: <><polyline points="4 14 10 14 10 20" /><polyline points="20 10 14 10 14 4" /><line x1="14" y1="10" x2="21" y2="3" /><line x1="3" y1="21" x2="10" y2="14" /></>,
  norm: <><path d="M2 20h20" /><path d="M3 20c4 0 5-15 9-15s5 15 9 15" /></>,
  activation: <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />,
  recurrent: <><polyline points="17 1 21 5 17 9" /><path d="M3 11V9a4 4 0 0 1 4-4h14" /><polyline points="7 23 3 19 7 15" /><path d="M21 13v2a4 4 0 0 1-4 4H3" /></>,
  transformer: <><path d="M11 3l1.8 4.9L17.7 9.7l-4.9 1.8L11 16.4l-1.8-4.9L4.3 9.7l4.9-1.8z" /><path d="M18.5 14.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z" /></>,
  dropout: <><path d="M10.1 2.18a9.9 9.9 0 0 1 3.8 0" /><path d="M17.6 3.71a10 10 0 0 1 2.69 2.7" /><path d="M21.82 10.1a9.9 9.9 0 0 1 0 3.8" /><path d="M20.29 17.6a10 10 0 0 1-2.7 2.69" /><path d="M13.9 21.82a9.9 9.9 0 0 1-3.8 0" /><path d="M6.4 20.29a10 10 0 0 1-2.69-2.7" /><path d="M2.18 13.9a9.9 9.9 0 0 1 0-3.8" /><path d="M3.71 6.4a10 10 0 0 1 2.7-2.69" /></>,
  embedding: <><line x1="4" y1="9" x2="20" y2="9" /><line x1="4" y1="15" x2="20" y2="15" /><line x1="10" y1="3" x2="8" y2="21" /><line x1="16" y1="3" x2="14" y2="21" /></>,
  loss: <><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" /></>,
  reshape: <><polyline points="15 3 21 3 21 9" /><polyline points="9 21 3 21 3 15" /><line x1="21" y1="3" x2="14" y2="10" /><line x1="3" y1="21" x2="10" y2="14" /></>,
  math: <path d="M18 7V4H6l6 8-6 8h12v-3" />,
  vision: <><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="2" /><path d="M21 15l-3.1-3.1a2 2 0 0 0-2.8 0L6 21" /></>,
};

export function CategoryIcon({ category, size = 14, strokeWidth = 2, className = "", style }: {
  category: NodeCategory;
  size?: number;
  strokeWidth?: number;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth}
      strokeLinecap="round" strokeLinejoin="round" className={className} style={style} aria-hidden="true">
      {PATHS[category] ?? PATHS.math}
    </svg>
  );
}

/** A see-through version of a color, also for colors given as var(--…) */
export function tint(color: string, percent: number): string {
  return `color-mix(in srgb, ${color} ${percent}%, transparent)`;
}

/** Square badge with the category's icon, used on nodes and in panels */
export function CategoryBadge({ category, color, size = 20, className = "" }: {
  category: NodeCategory;
  color: string;
  size?: number;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center justify-center flex-shrink-0 ${className}`}
      style={{ width: size, height: size, borderRadius: Math.round(size / 4), background: tint(color, 18), color }}
    >
      <CategoryIcon category={category} size={Math.round(size * 0.65)} strokeWidth={2.2} />
    </span>
  );
}
