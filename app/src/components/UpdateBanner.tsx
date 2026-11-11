import { useUpdateStore } from "../store/update-store";
import { useTrainingStore } from "../store/training-store";
import { openExternal } from "../lib/open-external";
import { LINKS } from "../lib/links";
import Button from "./ui/Button";

/** Offers a new version once the startup check finds one. Placed at the top of the canvas by App. */
export default function UpdateBanner() {
  const { status, version, progress, dismissed, install, dismiss } = useUpdateStore();
  const training = useTrainingStore((s) => s.status === "running" || s.status === "starting");

  const busy = status === "downloading" || status === "installing";
  if (dismissed && !busy) return null;
  if (status !== "available" && !busy) return null;

  return (
    <div className="pointer-events-auto flex items-center gap-3 pl-4 pr-2 py-2 rounded-xl text-xs"
      role="status"
      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", boxShadow: "var(--node-shadow-hover)" }}>
      <span className="whitespace-nowrap" style={{ color: "var(--text-secondary)" }}>
        {status === "downloading"
          ? <>Downloading TensorBloom {version}{progress != null ? ` (${Math.round(progress * 100)}%)` : "…"}</>
          : status === "installing"
            ? <>Installing TensorBloom {version}. The app will restart.</>
            : <>
                <span className="font-semibold" style={{ color: "var(--text-primary)" }}>TensorBloom {version}</span> is available.
                {training && " You can update after training finishes."}
              </>}
      </span>
      {!busy && (
        <>
          <Button size="sm" variant="ghost" onClick={() => void openExternal(LINKS.releases)}>What's new</Button>
          <Button size="sm" onClick={dismiss}>Later</Button>
          <Button size="sm" variant="primary" disabled={training} onClick={() => void install()}>Update and restart</Button>
        </>
      )}
    </div>
  );
}
