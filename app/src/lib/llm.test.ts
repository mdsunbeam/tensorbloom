import { describe, it, expect } from "vitest";
import {
  buildLLMGraph, countLLMParams, activeParams, estimateMemory, fitFor, kvCacheBytes, weightBytes, type LLMSpec,
} from "./llm";
import { computeModelSummary } from "./model-summary";
import { inferShapesClient as inferShapes } from "./shape-inference";

function spec(overrides: Partial<LLMSpec>): LLMSpec {
  const s: LLMSpec = {
    family: "llama", name: "test", vocab_size: 128256, d_model: 4096, n_layers: 32, n_heads: 32, n_kv_heads: 8,
    head_dim: 128, ffn_dim: 14336, mlp: "swiglu", norm: "rmsnorm", norm_eps: 1e-5, rope_theta: 500000,
    learned_positions: false, max_context: 131072, tie_embeddings: false, bias: "none", n_experts: 0,
    experts_per_token: 0, params: 0, warnings: [], ...overrides,
  };
  return { ...s, params: overrides.params ?? countLLMParams(s) };
}

const llama8b = spec({});
const qwenHalfB = spec({ vocab_size: 151936, d_model: 896, n_layers: 24, n_heads: 14, n_kv_heads: 2, head_dim: 64, ffn_dim: 4864, tie_embeddings: true, bias: "qkv", max_context: 32768 });
const mixtral = spec({ vocab_size: 32000, n_experts: 8, experts_per_token: 2, max_context: 32768 });

describe("countLLMParams", () => {
  it("matches published parameter counts", () => {
    expect(llama8b.params).toBe(8_030_261_248);
    expect(qwenHalfB.params).toBe(494_032_768);
    expect(mixtral.params).toBe(46_702_792_704);
    const gpt2 = spec({ vocab_size: 50257, d_model: 768, n_layers: 12, n_heads: 12, n_kv_heads: 12, head_dim: 64, ffn_dim: 3072, mlp: "gelu", norm: "layernorm", bias: "all", tie_embeddings: true, learned_positions: true, max_context: 1024, rope_theta: 0 });
    expect(gpt2.params).toBe(124_439_808);
  });

  it("counts a mixture of experts' active parameters per token", () => {
    // Mixtral 8x7B runs about 12.9B parameters per token
    expect(activeParams(mixtral) / 1e9).toBeCloseTo(12.9, 1);
    expect(activeParams(llama8b)).toBe(llama8b.params);
  });
});

describe("memory estimates", () => {
  it("sizes weights by precision, keeping the embedding and head at 16 bits when quantized", () => {
    expect(weightBytes(llama8b, "bf16")).toBe(llama8b.params * 2);
    const nf4 = weightBytes(llama8b, "nf4") / 1024 ** 3;
    expect(nf4).toBeGreaterThan(4.5);
    expect(nf4).toBeLessThan(6);
  });

  it("sizes the KV cache from key/value heads, not query heads", () => {
    // Llama 3.1 8B: 128 KB per token
    expect(kvCacheBytes(llama8b, 1)).toBe(2 * 32 * 8 * 128 * 2);
    expect(kvCacheBytes(llama8b, 8192)).toBe(1024 ** 3);
  });

  it("puts an 8B model's QLoRA under 16 GB and its full fine-tune far above", () => {
    const qlora = estimateMemory(llama8b, "lora", "nf4", 2048)!;
    expect(qlora.total / 1024 ** 3).toBeLessThan(16);
    const full = estimateMemory(llama8b, "full", "bf16", 2048)!;
    expect(full.total / 1024 ** 3).toBeGreaterThan(120);
    expect(estimateMemory(llama8b, "full", "nf4", 2048)).toBeNull();
  });

  it("rates the fit against the memory available", () => {
    expect(fitFor(10, 100)).toBe("fits");
    expect(fitFor(90, 100)).toBe("tight");
    expect(fitFor(101, 100)).toBe("too-big");
    expect(fitFor(1, null)).toBe("unknown");
  });
});

describe("buildLLMGraph", () => {
  it("draws tokens through every block to logits, with the blocks in one collapsed group", () => {
    const { nodes, edges, groups } = buildLLMGraph(qwenHalfB, 128);
    expect(nodes.filter((n) => n.data.moduleType === "DecoderBlock")).toHaveLength(24);
    expect(edges).toHaveLength(nodes.length - 1);
    expect(groups).toHaveLength(1);
    expect(groups[0].nodeIds).toHaveLength(24);
    expect(groups[0].collapsed).toBe(true);

    const { shapes, errors } = inferShapes(nodes, edges);
    expect(errors).toEqual({});
    const out = nodes.find((n) => n.data.moduleType === "Linear")!;
    expect(shapes[out.id].shape).toEqual([1, 128, 151936]);
  });

  it("counts the same parameters as the spec, apart from a tied head", () => {
    const untied = buildLLMGraph(llama8b, 128);
    expect(computeModelSummary(untied.nodes).totalParams).toBe(llama8b.params);
    const tied = buildLLMGraph(qwenHalfB, 128);
    expect(computeModelSummary(tied.nodes).totalParams).toBe(qwenHalfB.params + qwenHalfB.vocab_size * qwenHalfB.d_model);
  });
});
