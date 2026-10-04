# A100 vs H100: result alignment of the flex+bf16 fallback

Date: 2026-10-03
Hardware/Environment: 8x A100-SXM4-80GB, torch 2.10.0+cu128; branch `dev` @ 12dfed7, DISABLE_FP8=1

## Why

Quantify how the A100 fallback (flex attention + bf16, see 2026-10-03_a100_dev_setup.md)
compares to the official 8xH100 records whose recipes it runs.

## Track 1 (small, target 3.28) — official: records/082325_SparseAttnGate (14 runs)

| | Official H100 | Ours (2 runs) |
|---|---|---|
| Val loss | mean 3.2787 ± 0.0016, range 3.2760-3.2815 | 3.2764, 3.2727 |
| Train time | 168.8s mean | 420.7s / 422.4s (2.50x) |

Both A100 runs beat the official mean; 3.2727 beats all 14 official runs. Likely cause: the
record computes the lm_head loss in fp8; our bf16 loss removes that noise. Official record
itself had 4/14 runs above 3.28 and passes on t-test (p=0.0059).

## Track 2 (medium, target 2.92) — official: records/042225_GPT2Medium_Record8

| | Official H100 | Ours |
|---|---|---|
| Final val loss | 2.919750 | 2.920107 (+0.00036) |
| Train time | 1469s (24.50 min) | 3782s (63.0 min, 635 ms/step, 2.58x) |

Loss delta is within single-seed noise (track-1 spread was ±0.0016); ours lands 0.0001 above
target, official 0.0003 below. compiled_autograd was disabled on A100 (torch 2.10 flex
limitation) — perf-only, does not change the math.

## Conclusion

- Observed slowdown 2.50-2.58x, better than the raw bf16 compute ratio (A100 ~312 vs H100
  ~989 TFLOPS, 3.2x): the H100 runs are partly comms/overhead-bound at these model sizes.
- Numerics reproduce the official trajectories: track 1 slightly better (bf16 head), track 2
  within seed noise. The fallback is a valid dev platform for algorithm work.

## Artifacts

- a100_records/2026-10-03_FlexBF16Baseline/ (scripts, logs, plots, 2 full runs)
- a100_records/2026-10-03_FlexBF16Medium/ (script, logs, plot, 1 full run)

## Caveats

- Wall times are never comparable across hardware; only loss trajectories and step counts are.
- Official records are judged by multi-seed t-test; our sample is 2 runs (track 1) and 1 run
  (track 2). A clean medium-track pass likely needs another seed or a few extra steps.
