import { describe, it, expect } from "vitest";
import { explainTrainingError } from "./training-errors";

describe("explainTrainingError", () => {
  it("names the in_features value to use for a Linear mismatch", () => {
    const hint = explainTrainingError("mat1 and mat2 shapes cannot be multiplied (64x3136 and 784x512)");
    expect(hint).toContain("gets 3136 features");
    expect(hint).toContain("in_features is 784");
  });

  it("suggests a smaller batch for out-of-memory errors", () => {
    expect(explainTrainingError("CUDA out of memory. Tried to allocate 2.00 GiB")).toContain("smaller batch size");
  });

  it("points at the last layer when labels exceed the outputs", () => {
    expect(explainTrainingError("Target 12 is out of bounds.")).toContain("number of classes");
  });

  it("names the channel count for a convolution mismatch", () => {
    const hint = explainTrainingError(
      "Given groups=1, weight of size [32, 3, 3, 3], expected input[64, 1, 28, 28] to have 3 channels, but got 1 channels instead",
    );
    expect(hint).toContain("in_channels to 1");
  });

  it("names the missing package", () => {
    expect(explainTrainingError("No module named 'torchaudio.datasets'")).toContain('"torchaudio"');
  });

  it("recognizes a failed download", () => {
    expect(explainTrainingError("<urlopen error [Errno -3] Temporary failure in name resolution>")).toContain("internet connection");
  });

  it("returns null for errors it doesn't know", () => {
    expect(explainTrainingError("something unusual happened")).toBeNull();
  });
});
