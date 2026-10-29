import Dialog from "./ui/Dialog";
import Button from "./ui/Button";

const STEPS: { title: string; body: React.ReactNode }[] = [
  {
    title: "Start from a model, or from your data",
    body: <>Open a template from <b>Insert › From Template</b>, or add a <b>Data</b> node and pick a dataset in Properties. Then drag layers in from the left, or press <Kbd>Ctrl+K</Kbd> to search for one.</>,
  },
  {
    title: "Connect the layers",
    body: <>Drag from a layer's bottom handle to the next one's top. Each edge carries a tensor, and its shape shows when you hover it. A red edge means the next layer can't take that shape: the toolbar counts these problems and offers fixes.</>,
  },
  {
    title: "Set up training",
    body: <>Select a layer to change its parameters (the <b>?</b> next to each explains it). The <b>Training</b> tab holds the optimizer, epochs and loss. Templates fill these in for you.</>,
  },
  {
    title: "Train",
    body: <>Press <b>Train</b> or <Kbd>Ctrl+Enter</Kbd>. The <b>Charts</b> tab follows the loss and accuracy as it runs, and sums up the result when it's done.</>,
  },
  {
    title: "Use the result",
    body: <>Export the trained model as ONNX or TorchScript with <Kbd>Ctrl+E</Kbd>, or copy the generated PyTorch code from the <b>Code</b> tab. Save the project with <Kbd>Ctrl+S</Kbd> to keep the graph and its settings.</>,
  },
];

export default function GettingStartedDialog({ open, onClose, onShowTemplates }: {
  open: boolean;
  onClose: () => void;
  onShowTemplates: () => void;
}) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      width={520}
      title="Getting started"
      description="From an empty canvas to a trained model in five steps"
      footer={
        <>
          <Button onClick={onClose}>Close</Button>
          <Button variant="primary" onClick={() => { onClose(); onShowTemplates(); }}>Browse templates</Button>
        </>
      }
    >
      <ol className="p-5 space-y-4">
        {STEPS.map((s, i) => (
          <li key={s.title} className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-semibold bg-accent-subtle text-accent-text tabular-nums">
              {i + 1}
            </span>
            <div className="min-w-0">
              <div className="text-xs font-semibold text-fg mb-0.5">{s.title}</div>
              <p className="text-xs leading-relaxed text-fg-secondary">{s.body}</p>
            </div>
          </li>
        ))}
      </ol>
    </Dialog>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return <kbd className="px-1 py-px rounded bg-surface-raised border border-line text-[11px] font-sans text-fg">{children}</kbd>;
}
