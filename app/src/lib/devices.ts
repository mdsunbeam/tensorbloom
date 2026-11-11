/** Training devices offered in the Training panel */

export function isMac(): boolean {
  if (typeof navigator === "undefined") return false;
  return /Mac/i.test(navigator.platform || navigator.userAgent);
}

/** Apple Silicon Macs train on the GPU through MPS; everywhere else the GPU is CUDA */
export function baseDevices(mac: boolean = isMac()): string[] {
  return mac ? ["auto", "cpu", "mps"] : ["auto", "cpu", "cuda:0", "cuda:1"];
}

/** The device choices, keeping a saved choice from another computer visible */
export function deviceOptions(current: string, mac: boolean = isMac()): string[] {
  const base = baseDevices(mac);
  return base.includes(current) ? base : [...base, current];
}
