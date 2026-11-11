import { useMemo, useState } from "react";
import { useGraphStore } from "../store/graph-store";
import {
  buildDiagramScene, sceneToSvg, sceneToPdf, DEFAULT_DIAGRAM_OPTIONS,
  type DiagramOptions, type DiagramProjection, type DiagramPalette, type DiagramLabelMode,
} from "../lib/diagram-3d";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";
import Segmented from "./ui/Segmented";

interface DiagramExportDialogProps {
  open: boolean;
  onClose: () => void;
}

type FileFormat = "svg" | "pdf" | "png";

const OPTIONS_KEY = "tensorbloom.diagramOptions";

function loadOptions(): DiagramOptions {
  try {
    const raw = localStorage.getItem(OPTIONS_KEY);
    if (raw) return { ...DEFAULT_DIAGRAM_OPTIONS, ...JSON.parse(raw) };
  } catch { /* storage unavailable */ }
  return DEFAULT_DIAGRAM_OPTIONS;
}

/** Draws the SVG onto a canvas at `scale` and returns the PNG bytes */
async function svgToPng(svg: string, width: number, height: number, scale: number): Promise<Uint8Array> {
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("Couldn't draw the diagram"));
      img.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Couldn't draw the diagram");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("Couldn't encode the PNG");
    return new Uint8Array(await blob.arrayBuffer());
  } finally {
    URL.revokeObjectURL(url);
  }
}

function Check({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="flex items-start gap-2 text-xs text-fg cursor-pointer select-none" title={hint}>
      <input type="checkbox" className="mt-0.5 accent-[var(--accent)]" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

const fieldLabel = "block text-[11px] font-medium mb-1.5 text-fg-secondary";

export default function DiagramExportDialog({ open, onClose }: DiagramExportDialogProps) {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const shapeMap = useGraphStore((s) => s.shapeMap);
  const groups = useGraphStore((s) => s.groups);
  const projectName = useGraphStore((s) => s.projectName);

  const [opts, setOpts] = useState<DiagramOptions>(loadOptions);
  const [format, setFormat] = useState<FileFormat>("pdf");
  const [pngScale, setPngScale] = useState<"2" | "4">("4");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const set = <K extends keyof DiagramOptions>(key: K, value: DiagramOptions[K]) => {
    const next = { ...opts, [key]: value };
    setOpts(next);
    setResult(null);
    try { localStorage.setItem(OPTIONS_KEY, JSON.stringify(next)); } catch { /* storage unavailable */ }
  };

  const scene = useMemo(() => buildDiagramScene(nodes, edges, shapeMap, groups, opts), [nodes, edges, shapeMap, groups, opts]);
  const svg = useMemo(() => sceneToSvg(scene), [scene]);
  const empty = scene.blockCount === 0;

  const handleSave = async () => {
    setError(null);
    setResult(null);
    setBusy(true);
    const base = (projectName || "model").replace(/\s+/g, "_").toLowerCase() + "_diagram";
    const filename = `${base}.${format}`;
    try {
      const data: string | Uint8Array = format === "svg" ? svg
        : format === "pdf" ? sceneToPdf(scene)
        : await svgToPng(svg, scene.width, scene.height, Number(pngScale));

      let savedTo: string | null = null;
      let inTauri = false;
      try {
        const { save } = await import("@tauri-apps/plugin-dialog");
        inTauri = "__TAURI_INTERNALS__" in window;
        if (inTauri) {
          const path = await save({
            defaultPath: filename,
            filters: [{ name: format.toUpperCase(), extensions: [format] }],
            title: "Export Diagram",
          });
          if (!path) { setBusy(false); return; }
          const fs = await import("@tauri-apps/plugin-fs");
          if (typeof data === "string") await fs.writeTextFile(path, data);
          else await fs.writeFile(path, data);
          savedTo = path;
        }
      } catch (e) {
        if (inTauri) throw e;
      }
      if (!savedTo) {
        const type = format === "svg" ? "image/svg+xml" : format === "pdf" ? "application/pdf" : "image/png";
        const url = URL.createObjectURL(new Blob([data as BlobPart], { type }));
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        savedTo = filename;
      }
      setResult(`Saved ${savedTo}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed");
    } finally {
      setBusy(false);
    }
  };

  const copySvg = async () => {
    try {
      await navigator.clipboard.writeText(svg);
      setResult("Copied the SVG to the clipboard");
    } catch {
      setError("Couldn't copy to the clipboard");
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      width={980}
      title="Export diagram"
      description="A 3D figure of the model for papers and slides, as vector PDF, SVG or PNG"
      footer={
        <>
          <Segmented<FileFormat> label="File format" size="sm" value={format} onChange={(v) => { setFormat(v); setResult(null); }}
            options={[
              { value: "pdf", label: "PDF", title: "Vector, for LaTeX (\\includegraphics) and Word" },
              { value: "svg", label: "SVG", title: "Vector, editable in Inkscape, Illustrator or Figma" },
              { value: "png", label: "PNG", title: "Image, for slides and posters" },
            ]} />
          {format === "png" && (
            <Segmented<"2" | "4"> label="PNG resolution" size="sm" value={pngScale} onChange={setPngScale}
              options={[
                { value: "2", label: "2×", title: `${Math.round(scene.width * 2)} × ${Math.round(scene.height * 2)} px` },
                { value: "4", label: "4×", title: `${Math.round(scene.width * 4)} × ${Math.round(scene.height * 4)} px` },
              ]} />
          )}
          <Button onClick={copySvg} disabled={empty}>Copy SVG</Button>
          <div className="flex-1" />
          <Button onClick={onClose}>{result ? "Done" : "Cancel"}</Button>
          <Button variant="primary" onClick={handleSave} disabled={busy || empty}>{busy ? "Saving…" : `Save ${format.toUpperCase()}`}</Button>
        </>
      }
    >
      <div className="flex gap-4 p-5 min-h-0">
        <div
          className="flex-1 min-w-0 rounded-lg border border-line overflow-auto flex items-center justify-center p-3"
          style={{
            height: 460,
            background: opts.background === "white" ? "#ffffff"
              : "repeating-conic-gradient(#e4e4e7 0% 25%, #ffffff 0% 50%) 50% / 16px 16px",
          }}
        >
          {empty ? (
            <div className="text-xs text-fg-muted">Add layers to the canvas to draw a diagram.</div>
          ) : (
            <div
              role="img"
              aria-label="Diagram preview"
              className="[&>svg]:max-w-full [&>svg]:max-h-[436px] [&>svg]:h-auto [&>svg]:w-auto"
              dangerouslySetInnerHTML={{ __html: svg }}
            />
          )}
        </div>

        <div className="w-[260px] flex-shrink-0 space-y-4 overflow-y-auto" style={{ maxHeight: 460 }}>
          <div>
            <div className={fieldLabel}>View</div>
            <Segmented<DiagramProjection> label="View" size="sm" value={opts.projection} onChange={(v) => set("projection", v)}
              options={[{ value: "oblique", label: "3D" }, { value: "isometric", label: "Isometric" }, { value: "flat", label: "Flat" }]} />
          </div>
          <div>
            <div className={fieldLabel}>Colors</div>
            <Segmented<DiagramPalette> label="Colors" size="sm" value={opts.palette} onChange={(v) => set("palette", v)}
              options={[{ value: "paper", label: "Paper" }, { value: "tensorbloom", label: "App" }, { value: "grayscale", label: "Gray" }]} />
          </div>
          <div>
            <div className={fieldLabel}>Layer names</div>
            <Segmented<DiagramLabelMode> label="Layer names" size="sm" value={opts.labels} onChange={(v) => set("labels", v)}
              options={[{ value: "type", label: "Type" }, { value: "name", label: "Node name" }, { value: "none", label: "None" }]} />
          </div>
          <div className="space-y-2">
            <div className={fieldLabel}>Show</div>
            <Check label="Tensor shapes" checked={opts.showShapes} onChange={(v) => set("showShapes", v)} />
            <Check label="Legend" checked={opts.showLegend} onChange={(v) => set("showLegend", v)} />
            <Check label="White background" checked={opts.background === "white"} onChange={(v) => set("background", v ? "white" : "transparent")} />
          </div>
          <div className="space-y-2">
            <div className={fieldLabel}>Simplify</div>
            <Check label="Fold BN, activations and pooling into the layer before" checked={opts.foldLayers} onChange={(v) => set("foldLayers", v)}
              hint="Draws them as colored bands on the block, the way most papers do" />
            <Check label="Stack repeated layers (×N)" checked={opts.stackRepeats} onChange={(v) => set("stackRepeats", v)} />
            <Check label="Hide Flatten and reshapes" checked={opts.hideReshape} onChange={(v) => set("hideReshape", v)} />
            <Check label="Draw collapsed groups as one block" checked={opts.collapseGroups} onChange={(v) => set("collapseGroups", v)} />
          </div>
          {error && <div role="alert" className="px-3 py-2 rounded-lg text-xs bg-danger-subtle text-danger">{error}</div>}
          {result && <div role="status" className="px-3 py-2 rounded-lg text-xs bg-success-subtle text-success break-all">{result}</div>}
        </div>
      </div>
    </Dialog>
  );
}
