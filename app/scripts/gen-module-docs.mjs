// Writes docs/guides/modules.md from the layer registry. Usage (in app/): npm run docs:modules
import { build } from "esbuild";
import { writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";
import { tmpdir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url));
const outfile = resolve(tmpdir(), `tensorbloom-module-docs-${process.pid}.mjs`);
await build({ entryPoints: [resolve(here, "../src/lib/module-docs.ts")], bundle: true, format: "esm", platform: "node", outfile, logLevel: "error" });
const { moduleDocs } = await import(pathToFileURL(outfile).href);
const target = resolve(here, "../../docs/guides/modules.md");
writeFileSync(target, moduleDocs());
console.log(`Wrote ${target}`);
