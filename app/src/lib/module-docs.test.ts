import { describe, expect, it } from "vitest";
import { moduleDocs } from "./module-docs";
import { MODULE_REGISTRY } from "./registry";
import onDisk from "../../../docs/guides/modules.md?raw";

describe("module reference", () => {
  it("lists every module in the registry", () => {
    const md = moduleDocs();
    for (const def of Object.values(MODULE_REGISTRY)) expect(md).toContain(`| **${def.label}** |`);
  });

  it("matches docs/guides/modules.md (run `npm run docs:modules` after changing the registry)", () => {
    expect(onDisk).toBe(moduleDocs());
  });
});
