import { forwardRef } from "react";

type Variant = "primary" | "secondary" | "danger" | "ghost";
type Size = "sm" | "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  // The one main action in a panel or dialog
  primary: "bg-accent text-white hover:bg-accent-hover border border-transparent",
  // Everything else people can do there
  secondary: "bg-transparent text-fg-secondary border border-line hover:bg-surface-raised hover:text-fg",
  // Destructive: quiet until hovered, so it never competes with the main action
  danger: "bg-transparent text-fg-secondary border border-line hover:text-danger hover:border-danger hover:bg-danger-subtle",
  // Toolbar and inline actions
  ghost: "bg-transparent text-fg-secondary border border-transparent hover:bg-surface-raised hover:text-fg",
};

const SIZES: Record<Size, string> = {
  sm: "h-7 px-2.5 text-[11px] gap-1.5",
  md: "h-8 px-3.5 text-xs gap-2",
  lg: "h-10 px-4 text-sm gap-2",
};

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
  /** Stretch to the container's width */
  block?: boolean;
};

/** The app's button. Pick the variant by role, not by look: one primary per panel or dialog. */
const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", block = false, className = "", type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={`inline-flex items-center justify-center rounded-lg font-medium whitespace-nowrap transition-colors disabled:opacity-50 disabled:pointer-events-none ${
        variant === "primary" ? "font-semibold" : ""
      } ${VARIANTS[variant]} ${SIZES[size]} ${block ? "w-full" : ""} ${className}`}
      {...rest}
    />
  );
});

export default Button;

/** Square button for an icon. `label` is required: it's the accessible name and the tooltip. */
export const IconButton = forwardRef<HTMLButtonElement, Omit<ButtonProps, "children" | "size"> & { label: string; children: React.ReactNode; size?: "sm" | "md" }>(
  function IconButton({ label, variant = "ghost", size = "md", className = "", type = "button", ...rest }, ref) {
    return (
      <button
        ref={ref}
        type={type}
        aria-label={label}
        title={label}
        className={`inline-flex items-center justify-center rounded-md transition-colors disabled:opacity-50 disabled:pointer-events-none ${VARIANTS[variant]} ${
          size === "sm" ? "w-6 h-6" : "w-7 h-7"
        } ${className}`}
        {...rest}
      />
    );
  },
);
