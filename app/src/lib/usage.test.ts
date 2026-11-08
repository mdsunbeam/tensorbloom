import { describe, expect, it, vi } from "vitest";
import { buildPayload, deviceFamily, osFamily, track, usageAvailable } from "./usage";

describe("usage events", () => {
  it("send nothing in a build without an endpoint", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    expect(usageAvailable).toBe(false);
    track({ name: "try_it_opened", props: { first: true } });
    expect(fetchSpy).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it("carry only the documented fields", () => {
    const p = buildPayload(
      { name: "training_finished", props: { first: true, stopped_early: false, epochs: 3, seconds: 41, device: "mps" } },
      { installId: "abc", firstSeenAt: 0 },
      9 * 60_000,
    );
    expect(Object.keys(p).sort()).toEqual(["app_version", "event", "install_id", "minutes_since_install", "os", "props"]);
    expect(p.minutes_since_install).toBe(9);
    expect(p.event).toBe("training_finished");
  });

  it("reduce the OS and device to a family", () => {
    expect(osFamily("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)")).toBe("macos");
    expect(osFamily("Mozilla/5.0 (Windows NT 10.0; Win64; x64)")).toBe("windows");
    expect(osFamily("Mozilla/5.0 (X11; Linux x86_64)")).toBe("linux");
    expect(deviceFamily("cuda:1")).toBe("cuda");
    expect(deviceFamily("mps")).toBe("mps");
    expect(deviceFamily("auto")).toBe("auto");
  });
});
