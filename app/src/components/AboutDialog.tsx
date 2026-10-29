import { version } from "../../package.json";
import { getCachedEnvironment } from "../lib/sidecar";
import { usePythonStore } from "../store/python-store";
import { openExternal } from "../lib/open-external";
import { LINKS } from "../lib/links";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";
import { LogoMark, Wordmark } from "./ui/Logo";

export default function AboutDialog({ open, onClose, gpuInfo }: { open: boolean; onClose: () => void; gpuInfo: string | null }) {
  const pythonStatus = usePythonStore((s) => s.status);
  // Known once Python has started, i.e. after the first run or code import this session
  const env = getCachedEnvironment();

  const rows: [string, string][] = [
    ["Version", version],
    ["Python", env?.pythonVersion || (pythonStatus === "ready" ? "Installed" : "Not set up")],
    ["PyTorch", env?.torchVersion || "–"],
    ["CUDA", env ? (env.cudaVersion ?? "None") : "–"],
    ["GPU", env?.gpuName ?? gpuInfo ?? "–"],
  ];

  return (
    <Dialog
      open={open}
      onClose={onClose}
      width={380}
      title="About TensorBloom"
      footer={
        <>
          <Button onClick={() => void openExternal(LINKS.releases)}>Release notes</Button>
          <Button variant="primary" onClick={onClose}>Close</Button>
        </>
      }
    >
      <div className="px-5 pt-5 pb-4 text-center">
        <LogoMark size={40} className="mx-auto mb-2 text-accent-text" />
        <Wordmark className="text-lg" />
        <p className="text-xs text-fg-secondary mt-1">Build neural networks visually, and train them with PyTorch.</p>
      </div>
      <dl className="mx-5 mb-5 rounded-lg border border-line divide-y divide-line text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 px-3 py-2">
            <dt className="text-fg-secondary">{k}</dt>
            <dd className="text-fg font-medium tabular-nums text-right truncate" title={v}>{v}</dd>
          </div>
        ))}
      </dl>
    </Dialog>
  );
}
