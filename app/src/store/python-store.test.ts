import { describe, it, expect } from "vitest";
import { packageFromDownloadLine } from "./python-store";

describe("packageFromDownloadLine", () => {
  it("names the package from a PyTorch index URL", () => {
    expect(packageFromDownloadLine(
      "Downloading https://download.pytorch.org/whl/cu126/torch-2.9.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl (2.5 GB)",
    )).toBe("torch");
  });

  it("names the package from a PyPI file name", () => {
    expect(packageFromDownloadLine("Downloading nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux_2_27_x86_64.whl (393.1 MB)"))
      .toBe("nvidia_cublas_cu12");
  });

  it("skips metadata downloads and other lines", () => {
    expect(packageFromDownloadLine("  Downloading numpy-2.5.4-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (6.6 kB)")).toBeNull();
    expect(packageFromDownloadLine("Collecting torch")).toBeNull();
  });
});
