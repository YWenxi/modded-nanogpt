# Sparse attention gate (context-based no-op for attention heads)

Date: 2026-10-04

## Why this is needed

A head that has nothing useful to contribute for a token must still output something. Transformers
typically solve this by dumping attention mass onto the BOS token (the "attention sink") so the
output is near-zero and harmless. Two problems in this speedrun's setting:

- With sliding-window attention, BOS is frequently outside the window, so the sink is unavailable.
- RoPE makes any position-based sink depend on relative distance, so a learned sink does not
  behave uniformly as the window moves.

The sparse attention gate (added in record `records/082325_SparseAttnGate/` by @classiclarryd)
solves this distance-invariantly: instead of faking a no-op via a sink token, the head multiplies
its own output by ~0 per token. Background and plots in the author's blog post:
https://medium.com/@larry36d/modulating-attention-scores-cc0bcd853f06

## Mechanism

Setup at train_gpt.py:597-599:

```python
self.attn_gate_dim = 12
self.attn_gate = CastedLinear(self.attn_gate_dim, num_heads)  # 12 -> 6
self.attn_gate.weight.detach().zero_()
```

Applied to the attention output at train_gpt.py:614, where `y` is `(B, T, num_heads, head_dim)`:

```python
y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_dim])).view(B, T, self.num_heads, 1)
```

1. `x[..., :12]` reads only 12 of the 768 residual-stream channels (the "sparse" part); the record
   author found 12 active dims sufficient.
2. A tiny 12 -> num_heads linear produces one logit per head per token; negligible cost.
3. `sigmoid` squashes to (0, 1); broadcast-multiply over head_dim scales each head's output.
   Gate ~1 = normal output; gate ~0 = head silenced for that token (context-based no-op).
4. Zero-init means sigmoid(0) = 0.5 at start: all heads begin at half strength and learn when to
   open or close.

## Verification

From `records/082325_SparseAttnGate/README.md` (H100 record, 2025-08-23): estimated impact ~50
fewer training steps with a slight per-step time increase; iterations reduced 1750 -> 1695.
Validated with 14 runs: mean val loss 3.2787 vs 3.28 target (p=0.0059), mean time 168.75s.

## Caveats

- The gate reads the *first* 12 channels of the normed residual stream; this couples the
  mechanism to the learned representation using those channels — an emergent property, not a
  guaranteed one.
- Follow-on negative result from the same record: removing the computed value entirely for the
  first/last 3 layers (relying only on value embeddings) cost ~0.015 loss, more than the speedup
  was worth — the gate is the cheap way to silence attention, deleting submodules is not.
