#!/usr/bin/env node
// Downloads the uv binary that TensorBloom bundles, so the app can install its own Python and
// PyTorch on a clean machine. Tauri bundles it from src-tauri/binaries/uv-<target triple>.
//
// The target comes from TAURI_ENV_TARGET_TRIPLE (set by `tauri build/dev`, including with
// --target), then the first argument, then the host Rust toolchain. Each archive is checked
// against the SHA-256 pinned below. To update uv, change UV_VERSION and the hashes, which
// are in the uv-<target>.<ext>.sha256 files on the release page.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { chmodSync, copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const UV_VERSION = "0.11.32";

const SHA256 = {
  "x86_64-unknown-linux-gnu": "aab924fd522efd06f1c5f3b93a243864fc453132c94b2dc49f1371b528a4b967",
  "aarch64-unknown-linux-gnu": "4d4fa08d95b06642e5800df6a22bd71455f23f988269e18da2847971d8c0bf31",
  "aarch64-apple-darwin": "ed336d0ba49db8ef89b2b41fffa372ce63bd032f22a56f001c265891aec32829",
  "x86_64-apple-darwin": "77f5ca26c0de20e992a3677a174fe1121ee25c36f9b1434a863f75bf077a05eb",
  "x86_64-pc-windows-msvc": "acfde570451cfdb8689fa159a138ee805ba4e241c466432750302c86254b0984",
  "aarch64-pc-windows-msvc": "a7427ea0440bb826b6716d1837ff3d173b8e7d496cb09ee8f456b4e023a2fdcd",
};

const binDir = join(dirname(fileURLToPath(import.meta.url)), "..", "src-tauri", "binaries");

function hostTriple() {
  try {
    const out = execFileSync("rustc", ["-vV"], { encoding: "utf8" });
    const host = /^host: (\S+)/m.exec(out)?.[1];
    if (host) return host;
  } catch { /* no Rust toolchain; guess from Node */ }
  const arch = process.arch === "arm64" ? "aarch64" : "x86_64";
  if (process.platform === "darwin") return `${arch}-apple-darwin`;
  if (process.platform === "win32") return `${arch}-pc-windows-msvc`;
  return `${arch}-unknown-linux-gnu`;
}

async function download(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return Buffer.from(await res.arrayBuffer());
  } catch (e) {
    // Node's fetch ignores HTTPS_PROXY; curl doesn't, and ships with every supported OS
    console.warn(`fetch-uv: direct download failed (${e.message}), trying curl`);
    return execFileSync("curl", ["-fsSL", url], { maxBuffer: 256 * 1024 * 1024 });
  }
}

async function main() {
  const triple = process.env.TAURI_ENV_TARGET_TRIPLE || process.argv[2] || hostTriple();
  const sha = SHA256[triple];
  if (!sha) throw new Error(`No uv build is pinned for ${triple}. Add it to SHA256 in ${fileURLToPath(import.meta.url)}.`);

  const windows = triple.includes("windows");
  const exe = windows ? ".exe" : "";
  const dest = join(binDir, `uv-${triple}${exe}`);
  const stamp = join(binDir, `.uv-${triple}.version`);
  if (existsSync(dest) && existsSync(stamp) && readFileSync(stamp, "utf8").trim() === `${UV_VERSION} ${sha}`) {
    return;
  }

  const asset = `uv-${triple}.${windows ? "zip" : "tar.gz"}`;
  const url = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${asset}`;
  console.log(`fetch-uv: downloading uv ${UV_VERSION} for ${triple}`);
  const archive = await download(url);
  const actual = createHash("sha256").update(archive).digest("hex");
  if (actual !== sha) throw new Error(`fetch-uv: ${asset} has SHA-256 ${actual}, expected ${sha}`);

  const work = mkdtempSync(join(tmpdir(), "tensorbloom-uv-"));
  try {
    const archivePath = join(work, asset);
    writeFileSync(archivePath, archive);
    try {
      // bsdtar (macOS, Windows 10+) reads zip as well as tar.gz
      execFileSync("tar", ["-xf", archivePath, "-C", work], { stdio: "pipe" });
    } catch (e) {
      // GNU tar can't read zip, which matters only when bundling for Windows from Linux
      if (!windows) throw e;
      execFileSync("unzip", ["-q", archivePath, "-d", work]);
    }
    // The tarballs hold uv-<triple>/uv; the zips hold uv.exe at the top
    const extracted = [join(work, `uv-${triple}`, `uv${exe}`), join(work, `uv${exe}`)].find(existsSync);
    if (!extracted) throw new Error(`fetch-uv: no uv${exe} in ${asset}`);
    mkdirSync(binDir, { recursive: true });
    copyFileSync(extracted, dest);
    if (!windows) chmodSync(dest, 0o755);
    writeFileSync(stamp, `${UV_VERSION} ${sha}\n`);
    console.log(`fetch-uv: wrote ${dest}`);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error(e.message ?? e);
  process.exit(1);
});
