# Modded-NanoGPT on A100

This repo is a fork of the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt)
adapted to run on **8x NVIDIA A100-SXM4-80GB** (the upstream records target 8x H100).

The A100 fallback config is: FlexAttention + bf16 with `DISABLE_FP8=1` (A100 has no FP8
support). The upstream training algorithm, data pipeline, and loss targets are unchanged.
See [README_legacy.md](README_legacy.md) for the original upstream README with the full
H100 world-record history, and [notes/](notes/) for setup and experiment writeups.

## Record tracker

### Track 1: GPT-2 small (target 3.28)

| # | Record time | Date | Title | Description | Record |
| - | - | --- | --- | --- | --- |
| 1 | 7.01 / 7.04 minutes (2 runs) | 2026-10-03 | FlexAttention + bf16 baseline | First full runs on 8xA100. Val 3.2764 / 3.2727 vs 3.28 target (beats the official H100 mean of 3.2787; 2.5x slower than H100's 168.8s) | [a100_records/2026-10-03_FlexBF16Baseline](a100_records/2026-10-03_FlexBF16Baseline) |
| 2 | — (target not reached) | 2026-10-04 | AdamW optimizer swap | Muon replaced with distributed AdamW (DistAdamW) on all hidden matrices. Val 3.6017 vs Muon baseline 3.2727 at the same 1695 steps, 417s train. Not a record attempt — ablation only | [a100_records/2026-10-04_AdamW](a100_records/2026-10-04_AdamW) |

### Track 2: GPT-2 Medium (target 2.92)

| # | Record time | Date | Title | Description | Record |
| - | - | --- | --- | --- | --- |
| 1 | 63.04 minutes | 2026-10-03 | FlexAttention + bf16 baseline | Medium track (350M) on 8xA100. Val 2.9201 vs 2.92 target | [a100_records/2026-10-03_FlexBF16Medium](a100_records/2026-10-03_FlexBF16Medium) |

Each record directory contains the exact standalone training script used, its `run.sh`
launcher, run logs, and loss-curve plots.

## Running

```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py 8   # downloads the first ~800M training tokens
./run.sh                             # trains the current train_gpt.py on 8 GPUs
```

To reproduce a specific record, run its self-contained launcher, e.g.:

```bash
bash a100_records/2026-10-04_AdamW/run.sh
```

**Note: torch.compile adds around 5 minutes of latency the first time you run the code.**

## Layout

- `train_gpt.py` / `train_gpt_medium.py` — current dev scripts (small / medium track)
- `a100_records/` — dated record snapshots produced on this machine
- `records/` — upstream H100 world-record history (from the original repo)
- `notes/` — setup records and experiment findings
- `data/` — FineWeb token data loaders
