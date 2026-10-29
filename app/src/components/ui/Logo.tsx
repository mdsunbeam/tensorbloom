/** The TensorBloom mark: eight petals around a center. Drawn in currentColor. */
export function LogoMark({ size = 16, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 36 36" fill="currentColor" className={className} aria-hidden="true">
      <g transform="translate(18,18)">
        <circle cx="0" cy="-10" r="5" />
        <circle cx="7" cy="-7" r="5" />
        <circle cx="10" cy="0" r="5" />
        <circle cx="7" cy="7" r="5" />
        <circle cx="0" cy="10" r="5" />
        <circle cx="-7" cy="7" r="5" />
        <circle cx="-10" cy="0" r="5" />
        <circle cx="-7" cy="-7" r="5" />
      </g>
    </svg>
  );
}

/** "tensorbloom" wordmark, with "tensor" in the quieter color */
export function Wordmark({ className = "" }: { className?: string }) {
  return (
    <span className={`font-bold tracking-[-0.02em] ${className}`} aria-label="TensorBloom">
      <span className="text-fg-muted" aria-hidden="true">tensor</span>
      <span className="text-fg" aria-hidden="true">bloom</span>
    </span>
  );
}
