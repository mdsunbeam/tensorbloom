/**
 * Where TensorBloom is running. The desktop app (Tauri) can train, import code and export, because
 * it runs Python. The web version runs in a plain browser: building, shape checking and code
 * generation all work there, and anything that needs Python points to the desktop app.
 */
export const IS_DESKTOP = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
export const IS_WEB = !IS_DESKTOP;
