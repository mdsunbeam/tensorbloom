import { describe, it, expect } from "vitest";
import { gpuUpgradeAvailable } from "./python-store";

describe("gpuUpgradeAvailable", () => {
  const cpuWithGpu = { status: "ready" as const, torchCuda: false, cudaUpgradeSupported: true, nvidiaGpu: "NVIDIA GeForce RTX 4070" };

  it("offers the CUDA build when PyTorch is the CPU build and there's an NVIDIA GPU", () => {
    expect(gpuUpgradeAvailable(cpuWithGpu)).toBe(true);
  });

  it("doesn't offer it once PyTorch can use CUDA", () => {
    expect(gpuUpgradeAvailable({ ...cpuWithGpu, torchCuda: true })).toBe(false);
  });

  it("doesn't offer it without an NVIDIA GPU, on macOS, or before setup", () => {
    expect(gpuUpgradeAvailable({ ...cpuWithGpu, nvidiaGpu: null })).toBe(false);
    expect(gpuUpgradeAvailable({ ...cpuWithGpu, cudaUpgradeSupported: false })).toBe(false);
    expect(gpuUpgradeAvailable({ ...cpuWithGpu, status: "missing" })).toBe(false);
    expect(gpuUpgradeAvailable({ ...cpuWithGpu, status: "installing" })).toBe(false);
  });
});
