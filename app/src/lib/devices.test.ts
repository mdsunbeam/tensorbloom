import { describe, expect, it } from "vitest";
import { deviceOptions } from "./devices";

describe("deviceOptions", () => {
  it("offers MPS on a Mac and CUDA elsewhere", () => {
    expect(deviceOptions("auto", true)).toEqual(["auto", "cpu", "mps"]);
    expect(deviceOptions("auto", false)).toEqual(["auto", "cpu", "cuda:0", "cuda:1"]);
  });

  it("keeps a device chosen on another computer", () => {
    expect(deviceOptions("cuda:0", true)).toEqual(["auto", "cpu", "mps", "cuda:0"]);
  });
});
