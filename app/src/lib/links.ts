/** Where TensorBloom lives on the web */
export const LINKS = {
  docs: "https://tensorbloom.org/docs",
  pytorchReference: "https://pytorch.org/docs/stable/nn.html",
  newIssue: "https://github.com/mdsunbeam/tensorbloom/issues/new",
  releases: "https://github.com/mdsunbeam/tensorbloom/releases/latest",
};

/** The OS as people write it, for the feedback form */
export function osName(ua: string = typeof navigator !== "undefined" ? navigator.userAgent : ""): string {
  if (/Windows/i.test(ua)) return "Windows";
  if (/Mac/i.test(ua)) return "macOS";
  if (/Linux|X11/i.test(ua)) return "Linux";
  return "Other";
}

/** The feedback form, with this copy's version and OS filled in */
export function feedbackUrl(version: string, os: string): string {
  const q = new URLSearchParams({ template: "feedback.yml", version, os });
  return `https://github.com/mdsunbeam/tensorbloom/issues/new?${q}`;
}
