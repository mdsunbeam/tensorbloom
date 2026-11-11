import { describe, it, expect } from "vitest";
import { addLayer, insertAfter, removeLayers, setParams, connect, disconnect, checkParams, createNode, groupLayers, GraphOpError, type GraphDoc } from "./graph-ops";

function chain(): GraphDoc {
  let doc: GraphDoc = { nodes: [], edges: [], groups: [] };
  const a = addLayer(doc, { moduleType: "Input", params: { shape: "[1, 784]" }, position: { x: 0, y: 0 }, id: "a" }); doc = a.doc;
  const b = addLayer(doc, { moduleType: "Linear", params: { in_features: 784, out_features: 10 }, position: { x: 0, y: 120 }, id: "b" }); doc = b.doc;
  const c = addLayer(doc, { moduleType: "Output", position: { x: 0, y: 240 }, id: "c" }); doc = c.doc;
  doc = connect(doc, { from: "a", to: "b" });
  return connect(doc, { from: "b", to: "c" });
}

const links = (doc: GraphDoc) => doc.edges.map((e) => `${e.source}>${e.target}`).sort();

describe("settings", () => {
  it("fills in defaults and keeps the given settings", () => {
    const n = createNode({ moduleType: "Dropout", params: { p: 0.3 }, position: { x: 0, y: 0 } });
    expect(n.data.params).toMatchObject({ p: 0.3, inplace: false });
  });
  it("rejects settings a layer doesn't have, naming the ones it does", () => {
    expect(() => checkParams("Dropout", { rate: 0.3 })).toThrow(/Dropout has no setting "rate". Its settings: p, inplace/);
  });
  it("rejects values of the wrong type", () => {
    expect(() => checkParams("Linear", { out_features: 1.5 })).toThrow(GraphOpError);
    expect(() => checkParams("Linear", { bias: "yes" })).toThrow(/must be true or false/);
  });
  it("rejects unknown layer types", () => {
    expect(() => createNode({ moduleType: "Linearr", position: { x: 0, y: 0 } })).toThrow(/isn't a layer type/);
  });
});

describe("insertAfter", () => {
  it("puts the layer between a layer and what it fed, and makes room below", () => {
    const r = insertAfter(chain(), "b", { moduleType: "Dropout" });
    expect(links(r.doc)).toEqual(["a>b", `b>${r.id}`, `${r.id}>c`].sort());
    const pos = (id: string) => r.doc.nodes.find((n) => n.id === id)!.position.y;
    expect(pos(r.id)).toBe(240);
    expect(pos("c")).toBe(360);
    expect(pos("a")).toBe(0);
  });
});

describe("removeLayers", () => {
  it("bridges a removed layer in a chain", () => {
    const r = insertAfter(chain(), "b", { moduleType: "Dropout" });
    const doc = removeLayers(r.doc, [r.id], { bridge: true });
    expect(links(doc)).toEqual(["a>b", "b>c"]);
  });
  it("drops connections when not bridging, and empties groups", () => {
    const g = groupLayers(chain(), ["b", "c"], "Head");
    const doc = removeLayers(g.doc, ["b", "c"]);
    expect(links(doc)).toEqual([]);
    expect(doc.groups).toEqual([]);
  });
  it("fails on an unknown id", () => {
    expect(() => removeLayers(chain(), ["zz"])).toThrow(/no layer zz/);
  });
});

describe("connect", () => {
  it("refuses a taken input unless asked to replace it", () => {
    const doc = addLayer(chain(), { moduleType: "Linear", position: { x: 300, y: 0 }, id: "d" }).doc;
    expect(() => connect(doc, { from: "d", to: "c" })).toThrow(/already connected to b/);
    expect(links(connect(doc, { from: "d", to: "c", replace: true }))).toEqual(["a>b", "d>c"]);
  });
  it("refuses loops and unknown handles", () => {
    expect(() => connect(chain(), { from: "c", to: "a" })).toThrow();
    expect(() => connect(chain(), { from: "b", to: "a" })).toThrow(/loop/);
    const doc = addLayer(chain(), { moduleType: "Add", position: { x: 0, y: 0 }, id: "add" }).doc;
    expect(() => connect(doc, { from: "a", to: "add", toHandle: "q" })).toThrow(/no input "q". Its inputs:/);
  });
  it("uses the next free input on a layer with several", () => {
    let doc = addLayer(chain(), { moduleType: "Add", position: { x: 0, y: 0 }, id: "add" }).doc;
    doc = connect(doc, { from: "a", to: "add" });
    doc = connect(doc, { from: "b", to: "add" });
    expect(new Set(doc.edges.filter((e) => e.target === "add").map((e) => e.targetHandle)).size).toBe(2);
  });
  it("disconnects", () => {
    expect(links(disconnect(chain(), { from: "a", to: "b" }))).toEqual(["b>c"]);
    expect(() => disconnect(chain(), { from: "a", to: "c" })).toThrow(/isn't connected/);
  });
});

describe("setParams", () => {
  it("changes only the given settings", () => {
    const doc = setParams(chain(), "b", { out_features: 32 });
    expect(doc.nodes.find((n) => n.id === "b")!.data.params).toMatchObject({ in_features: 784, out_features: 32 });
  });
});
