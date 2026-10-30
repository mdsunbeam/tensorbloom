import { useRef } from "react";
import { create } from "zustand";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";

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
  // Keep the last question on screen while the dialog closes, so the text doesn't blank out
  const shown = useRef(request);
  if (request) shown.current = request;

  const answer = (choice: SaveChoice) => {
    request?.resolve(choice);
    useConfirmStore.setState({ request: null });
  };

  return (
    <Dialog
      open={request !== null}
      onClose={() => answer("cancel")}
      role="alertdialog"
      onTop
      width={400}
      initialFocus={saveRef}
      title={<>Save changes to &ldquo;{shown.current?.projectName}&rdquo;?</>}
      footer={
        <>
          <Button onClick={() => answer("discard")} className="mr-auto">Don&rsquo;t save</Button>
          <Button onClick={() => answer("cancel")}>Cancel</Button>
          <Button ref={saveRef} variant="primary" onClick={() => answer("save")}>Save</Button>
        </>
      }
    >
      <p className="px-5 py-4 text-xs leading-relaxed text-fg-secondary">
        You have unsaved changes. If you don&rsquo;t save before {shown.current?.action}, they&rsquo;ll be lost.
      </p>
    </Dialog>
  );
}
