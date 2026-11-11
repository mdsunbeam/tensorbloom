/**
 * Source of DecoderBlock, the Llama-style LLM block, for generated code.
 *
 * The sidecar runs the same source (DECODER_BLOCK_SOURCE in python/tensorbloom_llm.py), so exported
 * code matches what trained. python/test_llm.py checks the two copies are identical.
 */
export const DECODER_BLOCK_SOURCE = `class DecoderMLP(nn.Module):
    """Feed-forward part of a decoder block: gated (SwiGLU/GeGLU) or plain GELU."""

    def __init__(self, d_model, ffn_dim, mlp="swiglu", bias=False):
        super().__init__()
        self.gated = mlp in ("swiglu", "geglu")
        self.act = nn.SiLU() if mlp == "swiglu" else nn.GELU(approximate="tanh")
        if self.gated:
            self.gate_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)

    def forward(self, x):
        h = self.act(self.gate_proj(x)) * self.up_proj(x) if self.gated else self.act(self.up_proj(x))
        return self.down_proj(h)


class DecoderBlock(nn.Module):
    """Pre-norm LLM decoder block, as in Llama, Mistral, Qwen and Gemma.

    x -> norm -> causal self-attention (rotary positions, grouped-query heads) -> + x
      -> norm -> MLP (gated, plain, or a mixture of experts)                  -> + x
    """

    def __init__(self, d_model=512, n_heads=8, n_kv_heads=8, head_dim=0, ffn_dim=1408,
                 mlp="swiglu", norm="rmsnorm", norm_eps=1e-5, rope_theta=10000.0,
                 bias="none", n_experts=0, experts_per_token=2):
        super().__init__()
        if n_heads % n_kv_heads:
            raise ValueError(f"n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = head_dim or d_model // n_heads
        self.rope_theta = rope_theta
        self.experts_per_token = experts_per_token
        norm_cls = nn.RMSNorm if norm == "rmsnorm" else nn.LayerNorm
        self.attn_norm = norm_cls(d_model, eps=norm_eps)
        self.mlp_norm = norm_cls(d_model, eps=norm_eps)
        qkv_bias = bias in ("qkv", "all")
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias == "all")
        if n_experts:
            self.router = nn.Linear(d_model, n_experts, bias=False)
            self.experts = nn.ModuleList(DecoderMLP(d_model, ffn_dim, mlp, bias == "all") for _ in range(n_experts))
        else:
            self.mlp = DecoderMLP(d_model, ffn_dim, mlp, bias == "all")

    def rotate(self, x):
        """Rotary position embedding: turn each pair of features by an angle set by the position."""
        t, d = x.size(-2), x.size(-1)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        angles = torch.outer(torch.arange(t, device=x.device, dtype=torch.float32), inv_freq)
        angles = torch.cat([angles, angles], dim=-1)
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return x * angles.cos().to(x.dtype) + torch.cat([-x2, x1], dim=-1) * angles.sin().to(x.dtype)

    def attention(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.rope_theta:
            q, k = self.rotate(q), self.rotate(k)
        if self.n_kv_heads != self.n_heads:
            # Grouped-query attention: each key/value head serves several query heads
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))

    def feed_forward(self, x):
        if not hasattr(self, "router"):
            return self.mlp(x)
        # Mixture of experts: each token goes to its top-k experts, weighted by the router
        weights, chosen = self.router(x).softmax(dim=-1).topk(self.experts_per_token, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            hit = chosen == i
            tokens = hit.any(dim=-1)
            if tokens.any():
                w = (weights * hit).sum(dim=-1, keepdim=True)[tokens]
                out[tokens] = out[tokens] + expert(x[tokens]) * w
        return out

    def forward(self, x):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.mlp_norm(x))
        return x.squeeze(0) if squeeze else x
`;
