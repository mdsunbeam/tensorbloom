import { version } from "../../package.json";
import { getCachedEnvironment } from "../lib/sidecar";
import { usePythonStore, gpuUpgradeAvailable } from "../store/python-store";
import { openExternal } from "../lib/open-external";
import { LINKS, feedbackUrl } from "../lib/links";
import { osName, usageAvailable, useUsageStore } from "../lib/usage";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";
import { LogoMark, Wordmark } from "./ui/Logo";

export default function AboutDialog({ open, onClose, gpuInfo }: { open: boolean; onClose: () => void; gpuInfo: string | null }) {
  const pythonStatus = usePythonStore((s) => s.status);
  const consent = useUsageStore((s) => s.consent);
  const setConsent = useUsageStore((s) => s.setConsent);
  const offerGpu = usePythonStore(gpuUpgradeAvailable);
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
          <Button onClick={() => void openExternal(feedbackUrl(version, osName()))}>Send feedback</Button>
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
      {offerGpu && (
        <div className="mx-5 mb-5 -mt-2 flex items-center justify-between gap-3 rounded-lg bg-surface-sunken px-3 py-2.5 text-xs">
          <p className="text-fg-secondary">PyTorch is using the CPU. Add GPU support (CUDA, about 3 GB) to train on your NVIDIA GPU.</p>
          <Button size="sm" variant="primary" onClick={() => { onClose(); void usePythonStore.getState().installGpu(); }}>Add GPU support</Button>
        </div>
      )}
      {usageAvailable && (
        <label className="mx-5 mb-5 flex items-start gap-2 text-xs cursor-pointer">
          <input type="checkbox" className="mt-0.5" checked={consent === "granted"}
            onChange={(e) => setConsent(e.target.checked ? "granted" : "denied")} />
          <span className="text-fg-secondary">
            Share anonymous usage events to help improve TensorBloom.{" "}
            <button type="button" className="underline hover:text-fg" onClick={() => void openExternal(LINKS.privacy)}>What's sent</button>
          </span>
        </label>
      )}
    </Dialog>
  );
}
