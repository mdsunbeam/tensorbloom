import { useState, useEffect, useCallback } from "react";

interface PythonEnvStatus {
  ready: boolean;
  python_path: string;
  venv_path: string;
  error: string | null;
  stage: string;
}

interface PythonSetupProps {
  onReady: () => void;
}

export default function PythonSetup({ onReady }: PythonSetupProps) {
  const [status, setStatus] = useState<PythonEnvStatus | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installLog, setInstallLog] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Check on mount
  useEffect(() => {
    checkEnv();
  }, []);

  const checkEnv = useCallback(async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const result = await invoke<PythonEnvStatus>("check_python_env");
      setStatus(result);
      if (result.ready) {
        onReady();
      }
    } catch (e) {
      // Not running in Tauri (browser dev mode)
      setError("Not running in Tauri desktop app. Use `npx tauri dev` to launch.");
    }
  }, [onReady]);

  const handleInstall = useCallback(async () => {
    setInstalling(true);
    setError(null);
    setInstallLog("Creating Python virtual environment...\nThis may take a few minutes (downloading PyTorch ~2GB)...\n");

    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const result = await invoke<PythonEnvStatus>("setup_python_env");
      setStatus(result);
      setInstallLog((prev) => prev + "\nInstallation complete!");

      if (result.ready) {
        setTimeout(onReady, 1000);
      } else if (result.error) {
        setError(result.error);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setInstallLog((prev) => prev + `\nError: ${msg}`);
    } finally {
      setInstalling(false);
    }
  }, [onReady]);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center"
      style={{ background: "var(--bg-primary)" }}
    >
      <div
        className="w-[480px] rounded-xl overflow-hidden"
        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", boxShadow: "0 8px 32px rgba(0,0,0,0.3)" }}
      >
        {/* Header */}
        <div className="px-6 pt-6 pb-4 text-center">
          <svg width="64" height="64" viewBox="0 0 120 120" fill="none" className="mx-auto mb-3">
            <g transform="translate(60,60)">
              <circle cx="0" cy="-34" r="15" fill="var(--text-primary)"/>
              <circle cx="24" cy="-24" r="15" fill="var(--text-primary)"/>
              <circle cx="34" cy="0" r="15" fill="var(--text-primary)"/>
              <circle cx="24" cy="24" r="15" fill="var(--text-primary)"/>
              <circle cx="0" cy="34" r="15" fill="var(--text-primary)"/>
              <circle cx="-24" cy="24" r="15" fill="var(--text-primary)"/>
              <circle cx="-34" cy="0" r="15" fill="var(--text-primary)"/>
              <circle cx="-24" cy="-24" r="15" fill="var(--text-primary)"/>
            </g>
          </svg>
          <div className="text-lg font-bold" style={{ color: "var(--text-primary)", fontFamily: "'Outfit', 'Inter', sans-serif", letterSpacing: "-0.4px" }}>
            <span style={{ color: "var(--text-muted)" }}>tensor</span>
            <span>bloom</span>
          </div>
          <div className="text-[10px] font-medium uppercase tracking-[3px] mt-1" style={{ color: "var(--text-muted)" }}>
            Setup
          </div>
          <div className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
            A Python environment with PyTorch is required for training and code compilation
          </div>
        </div>

        {/* Status */}
        <div className="px-6 pb-4">
          {status && (
            <div className="rounded-lg p-3 mb-3 text-xs" style={{ background: "var(--bg-tertiary)" }}>
              <div className="flex items-center gap-2 mb-1">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ background: status.ready ? "var(--success)" : "var(--warning)" }}
                />
                <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                  {status.ready ? "Environment Ready" : "Setup Required"}
                </span>
              </div>
              <div style={{ color: "var(--text-muted)" }}>
                {status.stage === "not_installed" && "No Python environment found. Click Install to set up."}
                {status.stage === "deps_missing" && "Python venv exists but PyTorch is not installed."}
                {status.ready && `PyTorch installed at ${status.venv_path}`}
              </div>
            </div>
          )}

          {/* Install log */}
          {installLog && (
            <div
              className="rounded-lg p-3 mb-3 text-[11px] font-mono max-h-[140px] overflow-y-auto whitespace-pre-wrap"
              style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)" }}
            >
              {installLog}
              {installing && <span className="animate-pulse">|</span>}
            </div>
          )}

          {/* Error */}
          {error && (
            <div
              className="rounded-lg p-3 mb-3 text-xs"
              style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}
            >
              {error}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="px-6 pb-6 flex gap-3">
          {!status?.ready && (
            <button
              onClick={handleInstall}
              disabled={installing}
              className="flex-1 py-2.5 rounded-lg text-sm font-semibold text-white transition-colors"
              style={{
                background: installing ? "var(--text-muted)" : "var(--accent)",
                opacity: installing ? 0.7 : 1,
              }}
            >
              {installing ? "Installing... (this takes a few minutes)" : "Install Python Environment"}
            </button>
          )}
          {status?.ready && (
            <button
              onClick={onReady}
              className="flex-1 py-2.5 rounded-lg text-sm font-semibold text-white"
              style={{ background: "var(--accent)" }}
            >
              Continue to TensorBloom
            </button>
          )}
        </div>

        {/* Info */}
        <div className="px-6 pb-4 text-[10px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
          This will create a local Python virtual environment (~3 GB) in your app data directory.
          Requires Python 3.9+ to be installed on your system.
          {status?.venv_path && (
            <div className="mt-1 font-mono truncate">{status.venv_path}</div>
          )}
        </div>
      </div>
    </div>
  );
}
