import { describe, it, expect } from "vitest";
import { parseBlocks } from "./Markdown";

describe("parseBlocks", () => {
  it("splits paragraphs, lists and code", () => {
    const blocks = parseBlocks("First line\nsame paragraph\n\n- one\n- two\n\n1. a\n2. b\n\n```python\nx = 1\n```");
    expect(blocks).toEqual([
      { kind: "p", text: "First line same paragraph" },
      { kind: "ul", items: ["one", "two"] },
      { kind: "ol", items: ["a", "b"] },
      { kind: "code", lang: "python", text: "x = 1" },
    ]);
  });
  it("keeps an unclosed code block", () => {
    expect(parseBlocks("```\nx")).toEqual([{ kind: "code", lang: "", text: "x" }]);
  });
});
