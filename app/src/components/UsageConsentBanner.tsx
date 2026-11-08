import { usageAvailable, useUsageStore } from "../lib/usage";
import { openExternal } from "../lib/open-external";
import { LINKS } from "../lib/links";
import Button from "./ui/Button";

/** Asks once whether to share anonymous usage events. Placed at the top of the canvas by App. */
export default function UsageConsentBanner() {
  const consent = useUsageStore((s) => s.consent);
  const setConsent = useUsageStore((s) => s.setConsent);
  if (!usageAvailable || consent !== "unset") return null;

  return (
    <div className="pointer-events-auto flex items-center gap-3 pl-4 pr-2 py-2 rounded-xl text-xs max-w-full"
      role="status"
      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", boxShadow: "var(--node-shadow-hover)" }}>
      <span style={{ color: "var(--text-secondary)" }}>
        <span className="font-semibold" style={{ color: "var(--text-primary)" }}>Help improve TensorBloom?</span>{" "}
        Share anonymous usage events, such as when a training run finishes. Your models, code, and data are never sent.
      </span>
      <Button size="sm" variant="ghost" onClick={() => void openExternal(LINKS.privacy)}>What's sent</Button>
      <Button size="sm" onClick={() => setConsent("denied")}>No thanks</Button>
      <Button size="sm" variant="primary" onClick={() => setConsent("granted")}>Share</Button>
    </div>
  );
}
