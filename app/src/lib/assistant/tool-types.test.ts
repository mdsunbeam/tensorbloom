import { describe, it, expect } from "vitest";
import { validateInput, closestNames } from "./tool-types";

const schema = {
  type: "object",
  properties: { run_number: { type: "integer" }, mode: { type: "string", enum: ["a", "b"] }, ids: { type: "array", items: { type: "string" } } },
  required: ["run_number"],
};

describe("validateInput", () => {
  it("accepts input that matches", () => {
    expect(validateInput(schema, { run_number: 3, mode: "a", ids: ["x"] })).toBeNull();
  });
  it("names the field that's wrong", () => {
    expect(validateInput(schema, {})).toBe("input.run_number is required");
    expect(validateInput(schema, { run_number: 1.5 })).toBe("input.run_number must be an integer");
    expect(validateInput(schema, { run_number: 1, mode: "c" })).toMatch(/input.mode must be one of/);
    expect(validateInput(schema, { run_number: 1, ids: [3] })).toBe("input.ids[0] must be a string");
    expect(validateInput(schema, { run_number: 1, extra: true })).toMatch(/input.extra isn't a known field/);
  });
});

describe("closestNames", () => {
  it("suggests the layer that was probably meant", () => {
    expect(closestNames("conv2d", ["Linear", "Conv2d", "Conv1d", "ReLU"])[0]).toBe("Conv2d");
    expect(closestNames("Convolution2D", ["Linear", "Conv2d", "ReLU"])[0]).toBe("Conv2d");
  });
});

describe("validateInput with several types", () => {
  it("accepts any of them", () => {
    const s = { type: "object", properties: { clip: { type: ["number", "null"] } } };
    expect(validateInput(s, { clip: null })).toBeNull();
    expect(validateInput(s, { clip: 1.5 })).toBeNull();
    expect(validateInput(s, { clip: "x" })).toBe("input.clip must be a number, or input.clip must be null");
  });
});
