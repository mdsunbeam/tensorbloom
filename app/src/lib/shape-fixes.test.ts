import { describe, it, expect } from "vitest";
import { shapeErrorFix, shapeErrorSummary } from "./shape-fixes";
import type { ShapeError } from "./shape-inference";

const linearError: ShapeError = {
  message: "Dimension mismatch: input last dim is 3072, but in_features=784",
  type: "root",
  sourceNodeId: "fc1",
  expectedShape: [1, 784],
  receivedShape: [1, 3072],
  paramHint: "in_features",
};

describe("shapeErrorFix", () => {
  it("sets a size parameter to what the layer receives", () => {
    expect(shapeErrorFix(linearError, { in_features: 784 })).toEqual({ param: "in_features", from: 784, to: 3072 });
    expect(shapeErrorFix({ ...linearError, paramHint: "in_channels", receivedShape: [1, 16, 8, 8] }, { in_channels: 3 }))
      .toEqual({ param: "in_channels", from: 3, to: 16 });
  });

  it("has nothing to offer for other errors", () => {
    expect(shapeErrorFix({ ...linearError, type: "cascaded" }, { in_features: 784 })).toBeNull();
    expect(shapeErrorFix({ ...linearError, paramHint: "kernel_size" }, {})).toBeNull();
    expect(shapeErrorFix({ ...linearError, paramHint: undefined }, {})).toBeNull();
    expect(shapeErrorFix(linearError, { in_features: 3072 })).toBeNull();
  });
});

describe("shapeErrorSummary", () => {
  it("compares the shapes when both are known", () => {
    expect(shapeErrorSummary(linearError, (s) => `[${s.join(", ")}]`)).toBe("Gets [1, 3072], expects [1, 784]");
  });
  it("falls back to the message", () => {
    expect(shapeErrorSummary({ ...linearError, expectedShape: undefined }, String)).toBe(linearError.message);
  });
});
