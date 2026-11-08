/** Where TensorBloom lives on the web */
export const LINKS = {
  docs: "https://tensorbloom.org/docs",
  pytorchReference: "https://pytorch.org/docs/stable/nn.html",
  newIssue: "https://github.com/mdsunbeam/tensorbloom/issues/new",
  releases: "https://github.com/mdsunbeam/tensorbloom/releases/latest",
  privacy: "https://github.com/mdsunbeam/tensorbloom/blob/main/docs/guides/privacy.md",
};

/** The feedback form, with this copy's version and OS filled in */
export function feedbackUrl(version: string, os: string): string {
  const q = new URLSearchParams({ template: "feedback.yml", version, os });
  return `https://github.com/mdsunbeam/tensorbloom/issues/new?${q}`;
}
