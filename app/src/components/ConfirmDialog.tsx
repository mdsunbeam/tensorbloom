import { useEffect, useRef } from "react";
import { create } from "zustand";
import { useEscapeKey } from "../lib/use-escape-key";

export type SaveChoice = "save" | "discard" | "cancel";

interface Request {
  projectName: string;
  action: string;
  resolve: (choice: SaveChoice) => void;
}

const useConfirmStore = create<{ request: Request | null }>(() => ({ request: null }));

/** Asks whether to save the project before an action that would replace or close it */
export function askToSaveChanges(projectName: string, action: string): Promise<SaveChoice> {
  return new Promise((resolve) => {
    useConfirmStore.getState().request?.resolve("cancel");
    useConfirmStore.setState({ request: { projectName, action, resolve } });
  });
}

export default function ConfirmDialog() {
  const request = useConfirmStore((s) => s.request);
  const saveRef = useRef<HTMLButtonElement>(null);

  const answer = (choice: SaveChoice) => {
    request?.resolve(choice);
    useConfirmStore.setState({ request: null });
  };

  useEscapeKey(() => answer("cancel"), { enabled: request !== null });
  useEffect(() => { if (request) saveRef.current?.focus(); }, [request]);

  if (!request) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" style={{ background: "rgba(0,0,0,0.5)" }} onClick={() => answer("cancel")}>
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        className="w-[380px] rounded-xl p-5"
        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", boxShadow: "0 8px 40px rgba(0,0,0,0.4)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="confirm-title" className="text-sm font-semibold mb-1.5" style={{ color: "var(--text-primary)" }}>
          Save changes to &ldquo;{request.projectName}&rdquo;?
        </h2>
        <p className="text-xs leading-relaxed mb-5" style={{ color: "var(--text-secondary)" }}>
          You have unsaved changes. If you don&rsquo;t save before {request.action}, they&rsquo;ll be lost.
        </p>
        <div className="flex items-center gap-2">
          <button
            onClick={() => answer("discard")}
            className="px-3 py-1.5 rounded-lg text-xs font-medium"
            style={{ color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            Don&rsquo;t save
          </button>
          <div className="flex-1" />
          <button
            onClick={() => answer("cancel")}
            className="px-3 py-1.5 rounded-lg text-xs font-medium"
            style={{ color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            Cancel
          </button>
          <button
            ref={saveRef}
            onClick={() => answer("save")}
            className="px-3.5 py-1.5 rounded-lg text-xs font-semibold text-white"
            style={{ background: "var(--accent)" }}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
