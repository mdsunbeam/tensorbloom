/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      // Theme tokens from index.css, so components can use hover:, focus-visible: and disabled: classes
      colors: {
        accent: { DEFAULT: "var(--accent)", hover: "var(--accent-hover)", subtle: "var(--accent-subtle)", text: "var(--accent-text)" },
        danger: { DEFAULT: "var(--danger)", subtle: "var(--danger-subtle)" },
        warning: { DEFAULT: "var(--warning)", subtle: "var(--warning-subtle)" },
        success: { DEFAULT: "var(--success)", subtle: "var(--success-subtle)" },
        surface: { DEFAULT: "var(--bg-secondary)", sunken: "var(--bg-primary)", raised: "var(--bg-tertiary)", elevated: "var(--bg-elevated)" },
        fg: { DEFAULT: "var(--text-primary)", secondary: "var(--text-secondary)", muted: "var(--text-muted)" },
        line: { DEFAULT: "var(--border)", subtle: "var(--border-subtle)" },
      },
    },
  },
  plugins: [],
};
