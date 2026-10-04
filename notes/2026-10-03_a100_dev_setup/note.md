# A100 development setup (flex attention + bf16)

Date: 2026-10-03
Hardware: 8x NVIDIA A100-SXM4-80GB (compute capability 8.0), torch 2.10.0+cu128

## Why this is needed

`master` (the current 39.9s record code) cannot run on A100:

- Attention is a patched **FlashAttention-3** build (`devenpzak/flash-attn3-12864`), downloaded from
  the HF Hub at import time. FA3 is Hopper-only (sm90) and the Hub repo is no longer publicly
  accessible (404/401).
- The training path is full-stack **FP8** (`torch._scaled_mm`, `float8_e4m3fn`), which needs
  sm89/sm90 tensor cores. A100 has none.

The fallback is the last pre-FA3 record: flex_attention + bf16, which runs fine on sm80.

## Setup (already applied on branch `dev`)

```bash
cd /root/modded-nanogpt
git config --global --add safe.directory /root/modded-nanogpt   # if git complains about ownership
git stash push -m "docker env customization" -- Dockerfile      # only if Dockerfile is dirty
git checkout -b dev 12dfed7        # 2025-08-23, SparseAttnGate record: last flex-attention tree
```

No source patch is required for bf16. This revision has a built-in switch at `train_gpt.py:670`:

```python
use_fp8 = not os.environ.get("DISABLE_FP8", False)
```

`DISABLE_FP8=1` routes the lm_head loss (the only fp8 consumer in this revision) to a bf16
`F.linear`. `run.sh` on `dev` already has it baked in:

```bash
DISABLE_FP8=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Smoke test procedure (verified working)

```bash
sed -e 's/num_iterations = 1695/num_iterations = 20/' \
    -e 's/val_tokens = 10485760/val_tokens = 2097152/' train_gpt.py > smoke_gpt.py
DISABLE_FP8=1 torchrun --standalone --nproc_per_node=8 smoke_gpt.py
```

Observed 2026-10-03 on this box: exit 0 on all ranks, val_loss 10.83 -> 5.90 after 20 steps,
~235 ms/step, peak 33.6 GiB allocated / 50 GiB reserved per GPU. No torch-2.10 API issues
(`flex_attention`, `BlockMask.from_kv_blocks`, triton Muon kernels all work on sm80).

## Running

```bash
bash run.sh        # full 1695-step recipe, ~15-20 min on 8xA100
```

## Caveats

- Wall-time is not comparable to the record table (that is 8xH100). Same algorithm, same 3.28
  val-loss target; bf16 loss is strictly more precise than the fp8 loss the record used.
- Hyperparameters (1695 steps, lr/window schedules) are unchanged from the record commit.
- `logs/` accumulates run logs; each run also prints its code and environment into its log.

## Switching back

```bash
git checkout master
git stash pop      # restores the Dockerfile customization
```
