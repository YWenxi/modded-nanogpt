# Optimizer evolution in the NanoGPT speedrun (tracks 1-3)

Date: 2026-10-04
Source: `records/track_1_short/`, `records/track_2_medium/`, `records/track_3_optimization/` and
the record-tracker table in `README.md`, all read on the **master** branch (the `dev` branch
carries an older records snapshot). Wallclock figures are on 8xH100 unless noted.

## Why this is needed

The speedrun's optimizer went through ~2.5 years of continuous iteration: AdamW -> Muon ->
distributed/sharded Muon+Adam -> NorMuon -> the unified ANVIL stack. This note is the reference
map for diving into the current optimizer design: what was tried, in what order, what each change
measured, and what explicitly failed. Local `train_gpt.py` on `dev` corresponds to roughly the
2025-08-23 record (#31-era: Muon + DistAdam, FP8 lm head, 1695 steps).

## Big picture (track 1, GPT-2 small, target <= 3.28 FineWeb val loss)

| era | optimizer story | wallclock |
|---|---|---|
| 2024-06 | tuned AdamW baseline (lr 0.0018, betas (0.9,0.95), trapezoidal) | 31.4 min |
| 2024-10 | Muon introduced; SOAP sample-eff record; DistributedMuon | 24.9 -> 13.1 min |
| 2024-10-29 | 4-optimizer shootout: Muon > SOAP > Shampoo > Adam; zero wd optimal | -- |
| 2024-11 | momentum warmup, U-net + doubled lr | 8.2 -> 7.2 min |
| 2025-01 | batched Muon, Adam eps 1e-10, lr floor 0.1 | ~3 min |
| 2025-05/07 | reduce_scatter+all_gather inside both optimizers (DistAdam), Triton NS | 2.82 min |
| 2025-09/10 | custom sizing/batching, Polar Express replaces NS, NorMuon, lr fixes | 2.31 min |
| 2025-11/12 | cautious WD, batch-size schedule, gates/scalars moved Muon->Adam, fp32 Adam state, retie lm head | 2.07 min |
| 2026-01 | mantissa-sidecar bf16, module-level betas, interleaved comms, unified zero-copy optimizer | 1.75 min |
| 2026-02/08 | every-other-step Adam, sparse bigram grads, paired-head Muon, tail averaging | ~1.3 min |
| 2026-08-30 | ANVIL2 (record #92): twin-rail momentum, 6 quintic spectral maps, CUDA-graphed optimizer | **0.665 min** |

## Track 1 chronology — algorithm & hyperparameter changes

### 2024: Muon arrives

- **2024-06-06_AdamW (#2, 31.4 min, @kellerjordan0)**: tuned AdamW baseline. lr 0.0018,
  warmup 250, warmdown 2000 (trapezoidal), betas (0.9, 0.95), batch 2^19.
- **2024-10-04, Muon introduced (#3, 24.9 min, @kellerjordan0, @jxbz)**: SGD-Nesterov-momentum
  whose update is orthogonalized by a quintic Newton-Schulz (NS) iteration in bf16, coefficients
  (3.4445, -4.7750, 2.0315) chosen for max slope at zero (non-convergent; lands in ~0.68-1.13).
  Muon handles 2D hidden matrices; embeddings/head/scalars stay on Adam. Claimed ~1.5x sample
  efficiency vs Adam, <2% wallclock overhead. From CIFAR-10 speedrunning + Bernstein & Newhouse
  (arXiv:2409.20325).
- **2024-10-09_SOAP (@vyasnikhil96)**: SOAP (arXiv:2409.11321) on the body, lr ~0.0018,
  betas (.95,.95), precondition_frequency 10. Sample-efficiency record (3.15B tokens) but not
  wallclock.
- **2024-10-10_Muon (#4, 22.3 min)**: Muon lr 0.02 effective (0.1x0.0036 with
  `max(rows,cols)**0.5` RMS-matching scale), momentum 0.95, NS 5 steps; AdamW head lr 0.0036.
- **2024-10-17_DistributedMuon (#6, 13.1 min)**: NS work distributed round-robin over ranks
  (`i % world_size == rank`), results shared. Same hyperparameters.
- **2024-10-29_Optimizers**: shootout on the then-record (~20 tuning attempts each; embed/head
  always Adam, all wd 0, trapezoidal): Adam lr 0.0018 betas (0.9,0.95); DistributedShampoo
  lr 0.0018 betas (0.95,0.95) eps 1e-12 AdamGrafting; SOAP lr 0.0018 betas (.95,.95);
  **Muon lr 0.02 momentum 0.95 wins**. Failures: Sophia, Lion, ScheduleFree lose to Adam;
  AdEmaMix only marginally better; Shampoo eps >1e-8 hurts.
- **2024-11-03_UntieEmbed (#8)**: separate Adam lrs — wte 0.3, lm_head 0.002; Muon body 0.02/0.95.
- **2024-11-06_ShortcutsTweaks (#9, 8.2 min)**: **Muon momentum warmup** 0.85->0.95 over 500
  steps; learnable scalars on Adam lr 0.02.
- **2024-11-10_UNetDoubleLr (#11, 7.2 min)**: doubled lr everywhere — Muon 0.04, wte 0.6,
  lm_head 0.008, scalars Adam 0.04.

### 2025 H1: consolidation + systems

- **2025-01-13_Fp8LmHead (#19)**: lr cooldown floor 0.0 -> 0.1.
- **2025-01-16_Sub3Min (#20, 2.99 min)**: **batched Muon** — QKV stacked (4, hdim, dim), one NS
  pass over the batch dim (net 1-2 s). **Adam eps 1e-8 -> 1e-10** because zero-init head gives
  sub-eps gradients (+0.0014 val, -10 steps). Adam: head 0.008, embed 0.6, scalars 0.04,
  betas (0.8,0.95); Muon lr 0.05.
- **2025-01-26_BatchSize (#21)**: batch 524K->393K tokens (critical batch size, arXiv:2410.21676);
  FP8 head scales retuned.
- **2025-05-24/25/30 (FasterReduce, EvenFasterReduce, noallreduce; #22-24)**: all_reduce replaced
  by **reduce_scatter + all_gather inside both optimizers** (@vagrawal). Muon: RS grads, each
  rank runs momentum+NS on its shard, all-gather params. **DistAdam** created: per-rank Adam
  state slices, bias corrections per slice.
- **2025-07-12_BosAlign (#26)**: cooldown_frac 0.4->0.45, lr floor 0.1->0.05, 1750 steps.
- **2025-07-18_TritonMuon (#27, 2.82 min, @byronxu99)**: Triton NS kernels — symmetric A=X@X.T
  computed as one triangle + mirrored store; fused B=b*A+c*A@A; buffer swapping. MLP matrices
  made same shape (24 total, divisible by 8 GPUs) for Muon grouping. Momentum warmup 300 steps.

### 2025 H2: second-order refinements

- **2025-09-11_VectSigmoidBFloat16 (#32)**: vectorized Muon loops; bf16 optimizer state.
- **2025-09-23_MuonCustomSizing (#36, @classiclarryd)**: attn/MLP weights in identical shapes so
  both share reduce_scatter calls; 4 staged RS groups with NS of group k overlapped with RS of
  group k+1; leading with small params worth +0.2 s.
- **2025-09-29_PolarExpress (#38, @varunneal)**: **NS replaced by Polar Express**
  (arXiv:2505.16932), num_iters 5, muon lr 0.06, safety_factor 1.02. -10 steps, same step time.
  Honest caveat: paper's gains not fully replicable on the speedrun.
- **2025-09-30_CustomBatching (#39)**: **Adam updates only every other step** (grads accumulate
  2 steps), Muon every step; Muon momentum cooldown over last 50 steps; Adam beta1 0.8->0.7.
- **2025-10-04_Backout (#40)**: Adam beta1 0.7->0.65; Adam shard size 8/GPU beats 7 or 9 (odd but
  measured, 1.8 s).
- **2025-10-24_NorMuon (#41, @li_zichong, arXiv:2510.05491)**: after Polar Express, per-neuron
  mean-square update `v_mean` tracked by an EMA (beta2 0.95, fp32); each neuron's update rescaled
  by 1/sqrt(EMA), then whole matrix rescaled back to its pre-normalization norm. NorMuon on
  hidden+gate params lr 0.06, momentum 0.95, beta2 0.95.
- **2025-10-27_FixMuonLR (#42, @varunneal)**: **Muon lr halved 0.06 -> 0.03** ("~twice as high
  as it should be", partly NorMuon's second-order effect); MLP up-proj lr_mul 2.0 per
  "effective lr ~ sqrt(output_dim)"; NorMuon normalization columnwise (output-dim) better than
  rowwise. -30 steps.
- **2025-10-31_AdamSyncGradientHook (#44, @akash5474)**: DistAdam reduce-scatter moved into
  backward hooks; step() walks params in reverse order to overlap comms with backward. -0.7 s.
- **2025-11-10_CautiousWD (#43, @varunneal)**: **Cautious Weight Decay** (arXiv:2510.12402) on
  NorMuon: decoupled WD only where sign(update . param) indicates growth. wd 1.2 on the lr
  schedule (eff 0.036); scheduled WD worth 10-15 steps over unscheduled. CWD has *higher* val
  loss most of training and only wins during cooldown; params end <20% of baseline mean-square.
- **2025-11-29_BatchSizeSchedule (#46, @varunneal)**: batch ramps 8->16->24 (x2048x8 tokens) at
  33%/66% with lr max scaling 1.0 -> 1.51 -> 1.93 ((16/8)^0.6, (24/8)^0.6); cooldown_frac 0.55;
  NorMuon lr 0.023. -65 steps. Prior batch schedules had failed "for some time".
- **2025-12-10_SALambdaOnWeights (#47)**: sa_lambda folded into QKV weights (QK-norm makes it
  exact); DistAdam hook warmup fix. Failures: lambda into W_O or into attn-gate output — worse.
- **2025-12-11_NorMuonOptimsAndFixes (#48, @ChrisJMcCormick)**: NorMuon axis heuristic fixed for
  gates/heads (smear gate had been constrained to +/-1 — biggest single win); weight layouts
  transposed, accidentally doubling attn/mlp lr via the tall-matrix heuristic — kept (lr muls
  ended attn 2.0, MLP-in 2.0, MLP-out 4.0); compiled variance-norm/CWD helpers with 0-D CPU
  tensors to avoid recompiles. Failed: per-head Muon, RoPE/NoPE-split Muon, 8x(768x768) MLP
  chunks — all faster but hurt learning.
- **2025-12-18_CautiousWDAdam (#50)**: CWD extended to Adam params (skip scalars), wd 0.005.
- **2025-12-19_RetieLMHead (#51)**: re-tied embed/lm_head (reverses #8) to cut DistAdam comms —
  unexpectedly also -55 steps; FP8 scales retuned to fix CWD-induced variance.
- **2025-12-21_SmoothedScalars (#52)**: smear gate Muon->Adam lr_mul 0.01; scalars get own
  DistAdam with betas (0.9,0.99); scalar updates frozen 40 steps at schedule transitions;
  all_reduce (replicated) path for params <1024 values. Failed: grad clipping at transitions,
  per-scalar lrs/betas.
- **2025-12-22_MultiTokenPrediction (#53)**: untie embed/lm_head at 2/3 of training; the 75x
  embed lr_mul hypothesis retired ("untied embedding weights did not benefit from a lr
  multiplier"); CWD mask `>=` -> `>`.
- **2025-12-31_GatesToCompiledAdam (#56)**: attn+VE gates Muon->Adam as pre-stacked banks;
  **fp32 Adam state faster *and* lower loss than bf16** (3.2777 vs 3.2786, p=0.0036); compiled
  fused Adam update with fused CWD mask. Failed: freezing gates with scalars (variance).

### 2026: unification and ANVIL

- **2026-01-04_MixedPrecisionInterweavedOptimizer (#57)**: MLP/attn weights bf16 + **uint16
  mantissa sidecar** in Muon (updates accumulate in effective fp32; without it bf16 updates are
  "substantially detrimental"). **Module-level Adam betas**: lm_head/embed (0.5,0.95) — sparse
  rare-token gradients shouldn't drag the first moment; scalars/gates (0.9,0.99); x0_lambdas
  (0.65,0.95); value_embed (0.75,0.95). Interleaved Muon<->Adam step (Adam launches Muon's RS,
  Muon runs Polar Express between Adam's comms) — near-zero comm downtime.
- **2026-01-18_UnifiedOptimizers (#61)**: **`NorMuonAndAdam`, single unified zero-copy optimizer**
  — no param groups, per-parameter config table, explicit scatter_order/work_order instead of
  hooks (work order suggested by Claude Opus; consistent -0.0001 loss). Attn bank (40,768,768),
  MLP bank (24,3072,768) balanced 5/3 matrices per GPU. Tied-embed grads combined manually;
  moments all-gathered + resharded at untie. Failed: blending Muon+Adam updates on matrices.
- **2026-01-19_BigramHashEmbedding (#62)**: bigram table on Adam; every-other-step Adam means
  alternating 31/37 ms steps.
- **2026-02-03_VeTuned / 2026-01-30_VeFused (#65)**: 5 VEs fused into one parameter
  (`view(5, vocab, -1)[:, input]`, ~0.8 s); grad_scale=2 for 1<->8 GPU grad-magnitude
  consistency. Record #63 (@photon_mz): VE pattern [.12...012] -> [.01...345], -25 steps.
- **2026-02-06_SparseBigramGradient (#71)**: sparse gradient exchange for bigram table
  (all_to_all_single, ~60-80% sparsity). Sparse row-level Adam tried, needs re-tuning.
- **2026-02-10_ShortWindow (#72)**: **lr floor 0.1 -> 0.15** ("avoids stalling" in cooldown).
- **2026-02-16_FlattenForward (#74)**: transposed Polar Express (XTX algebra) for tall matrices;
  Nesterov momentum fused into Polar Express. Candid note: mathematically-neutral kernel changes
  measurably move val loss both directions.
- **2026-03-22_VarlenMaxDocs (#78)**: NorMuon variance EMA beta2 0.95 -> 0.9 (from nanochat).
- **2026-04-08_PairedHeadMuon (#80)**: Polar Express per **head-pair** on K/V (paired-head
  attention) — works where per-head Muon failed.
- **2026-08-30_ANVIL2 (#92, @DevenPzak) — current record: 39.914 s, val 3.27731 (n=18)**.
  Swapping in record #89's optimizer costs +21.1 millinats at equal wall; the ANVIL algorithm
  itself is worth +4.16 s of the 33.98 s win.

## The current stack (record #92, refactored into `track_1_short/optim/anvil.py`)

**One class `AnvilAndAdam`** (not a torch Optimizer): per-param config table, explicit
scatter/work comms orders, two modes (sharded = RS + shard update + AG; replicated = one flat
all_reduce per dtype + redundant update).

**ANVIL (projection matrices: qk_bank, vo_bank, mlp_bank; bf16 + sharded)**
- lr 0.023, momentum 0.95 (Nesterov), beta2 0.9 (lane-energy EMA), weight_decay 2.25.
- **Twin-rail velocity**: fast rail beta scheduled (240-step warmup 0.85->0.93, 50-step cooldown
  to 0.85), slow rail constant 0.98; until step 514 fast-only, then blend 0.4385 fast +
  0.5615 slow.
- Whitening: six re-derived quintic spectral maps (successor of Polar Express/NS; envelope
  [0.9951, 1.00041], tail gain 643.13) on bf16, norm by 1.05*||X||_F + 1e-6; Triton XTX/XXT
  symmetric kernels; split-baddbmm for >1024-row banks.
- Per-lane energy equalization (NorMuon-style), sign-aligned cautious decoupled WD gated on the
  slow rail's sign, **uint16 mantissa sidecar** for exact fp32 commits on bf16.
- lr multipliers: shape multiplier max(1, rows/cols)^0.5; c_proj 2x; frozen 0.

**Adam (everything else; steps only on odd steps)**
- Defaults lr 0.008, eps 1e-10, wd 0.005; compiled fused update with bias correction and fused
  cautious-WD mask `(update . p) > 0`; fp32 state.
- lm_head/embed: sharded, betas (0.5,0.95), wd_mul 150; tied until the extension stage (grads
  combined by transpose-add; moments transferred at split).
- value_embeds: sharded, betas (0.75,0.95), lr_mul 70, wd_mul 5; n-gram cadence (every 4th step
  from step 336; betas squared, wd_mul 10); row-sparse gradient exchange.
- scalars (replicated, (0.9,0.99), lr_mul 5), smear_gate ((0.9,0.99), lr_mul 0.01),
  post/resid_lambdas ((0.9,0.95)), MUDD params ((0.9,0.99), lr_mul 0.25).

**N-gram table** (84.6M x 768, not a model param): row-compacted Adam — beta1 = 0, one fp32
second moment per row, beta2 0.95, lr_mul 70; every 4th step from 336; untouched rows exactly 0
update; missed beta2 decays replayed lazily.

**Schedules**: 5-stage batch/window/seq (batch 8->16->24->20->8 x2048x8 tokens; 1194 steps);
lr stage multipliers 1.0 -> 1.52 -> 1.73; linear cooldown over last 80% to LR_FLOOR 0.30
(README says 0.20 but code ships 0.30 — flagged in schedule.py).

**Terminal weight ships (inside the timed loop)**: tail-EMA (window 298, blend 0.65) on
embed+lm_head; tail-avg replacing vo/mlp banks; bank tail-blend with Richardson correction;
then **decontraction** — every shipped weight rescaled to its pre-ship Frobenius norm.

## Explicitly failed optimizer experiments (the graveyard)

- Sophia, Lion, AdamWScheduleFree: lose to Adam (2024-10-29). AdEmaMix: marginal.
- Shampoo eps >1e-8 hurts (2024-10-29).
- Polar Express paper gains not fully replicable here (2025-09-29).
- Per-head Muon, RoPE/NoPE-split Muon, square-chunk MLP Muon: faster but hurt learning
  (2025-12-11). Per-head-*pair* later worked (2026-04-08).
- Grad clipping at schedule transitions; per-scalar custom lrs/betas (2025-12-21).
- Freezing gates alongside scalars: one lucky run, variance too high (2025-12-31).
- Blending Muon+Adam updates on matrices: nothing promising (2026-01-18).
- torch.Embedding(sparse=True): metadata overhead, breaks CUDA graphs (2026-01-26).
- Sparse row-level Adam for bigrams: needs re-tuning (2026-02-06; later realized as
  row-compacted Adam in ANVIL2).
- nanochat-style single fused NorMuon helper: slower here (2026-02-16).
- Pre-multiplying post-attn/post-MLP lambdas into weights: hurts loss (2025-12-10, 2026-02-16).
- Simple linear FP8-scale schedule: failed (2025-12-19).

## Surprising / worth remembering

- Adam shard size 8/GPU measurably beats 7 or 9 (2025-10-04).
- NorMuon's 0.06 lr was ~2x too high; halving worked *and* the accidental layout-transpose
  doubling worked; later lr_mul tuning had "surprisingly little effect" (2025-10-27, 2025-12-11).
- Wrong-axis NorMuon normalization had been constraining the smear gate to +/-1 (2025-12-11).
- CWD looks worse for most of training and only wins during cooldown (2025-11-10).
- fp32 Adam state is faster *and* lower loss than bf16 (2025-12-31).
- Re-tying embed/lm_head reduced step count, against expectations (2025-12-19).
- A comms work order suggested by Claude Opus beat the hand-designed hook scheme and
  inexplicably lowered loss (2026-01-18).

## Track 2 (GPT-2 Medium 350M, target <= 2.92) — optimizer changes

Baseline (#2, 2025-01-18, 29.3 min): Muon lr 0.025 momentum 0.95 Nesterov wd 0, NS 5 steps;
Adam head 0.003 / embed 0.3 / scalars 0.015, betas (0.8,0.95), eps 1e-10; cooldown_frac 0.4;
7500 steps.

- **2025-02-08_WeightDecay (#3, 28.1 min)**: decoupled **wd 0.01 in Muon**; 7150 steps.
- **2025-02-14_OptCoeffs (#4, 27.7 min, @leloykun)**: per-iteration NS coefficients instead of
  constant (4.0848,-6.8946,2.9270), (3.9505,-6.3029,2.6377), ... ; 7050 steps.
- **2025-03-06_LongerCooldown (#5, 27.2 min)**: cooldown_frac 0.4->0.6; 6950 steps.
- **2025-03-25_ArchOptTweaks (#6, 25.95 min)**: **wd_mul 2.0 on MLP matrices**; 6710 steps.
- **2025-04-16_Record7 (25.29 min)**: sharded mixed-precision Muon (bf16 weights + mantissa
  correction) — the ancestor of the 2026-01 track-1 sidecar; 6450 steps.
- **2025-04-22_Record8 (24.50 min)**: cooldown_frac 0.6->0.7; cubic window schedule; 5960 steps.
- **2025-09-16_Snoo (#12, 23.28 min, @dominikkallusky)**: **Snoo (Sparse Nesterov Outer
  Optimizer)** — every k=28 steps the displacement becomes the gradient of an outer Nesterov
  step (outer lr 0.68, momentum 0.37); Muon lr 0.025->0.027; 5640 steps.
- **2025-09-17_UpdateSmoothing (#13, 23.14 min, @acutkosky)**: **EMA of Muon update**,
  final_update = EMA(NS(EMA(grads))), weight 0.5 decaying to 0.2 over 3000 steps; Muon lr 0.03;
  lr floor 0.01x peak. Ablation without smoothing misses target (p=0.80).
- **2025-09-30_SmoothedSnooMedium (#14, 23.08 min)**: Snoo around smoothed Muon (constant 0.2);
  5590 steps.
- **2025-12-31_BulkSmallTrackTransfer (#18, 17.35 min, @classiclarryd)**: ports the track-1
  stack — NorMuon lr 0.015 (vs 0.023), beta2 0.95, wd 1.2 + CWD; DistAdam lr 0.004,
  betas (0.8,0.95), wd 0.005; batch ramp; cooldown 0.70; split embed/head at 2/12. Snoo/EMA
  wrappers *not* carried over; hparams flagged as first guesses.

## Track 3 (`records/track_3_optimization/`) — the pure optimizer benchmark

Holds everything except the optimizer fixed: same FineWeb data, batch 524,288 tokens/step,
simplified GPT-2-small arch (12L/768, plain 1024-ctx causal attention, RMSNorm gains and biases
restored, no VE/skip lambdas/Triton), one fwd-bwd per step. Metric: **steps to <= 3.28**
(wallclock irrelevant). Validity: one-sided z-test, (3.28 - avg) * sqrt(n) >= 0.004.

Baseline (`train_gpt_simple.py`, = result #36, 3250 steps): AdamW embed lr 0.7 / head 0.004 /
1-D 0.015, betas (0.8,0.95), eps 1e-10, wd 0.001; Muon lr 0.025 wd 0.05 mu 0.95 with **12 NS
iterations of the convergent cubic coeffs (2, -1.5, 0.5)** (wallclock doesn't matter here);
cooldown_frac 0.7.

World-record progression (selection): 3600 (Muon+aux Adam) -> 3325 (MuonH, @kaiyue-wen) ->
3250 (NorMuon + u/w floor, @kumarkrishna) -> 3225 (Contra-Muon) -> 3150 (SOAP-Muon on MLP,
@Sam_Acqua) -> 3030 (PowerCool + Contra/Soft-Muon interp) -> 2990 (radial brake) -> 2930
(Aurora row-balanced polar) -> 2875 (Circuit-Muon V/O coupling) -> 2750 (SOAP-Muon on ALL
hidden matrices, freq 1) -> 2720 (final-step EMA blend) -> **2690 (current WR, #46, @ypwang61:
#45 + per-output-row u/w floor + Cautious WD 0.025)** — 20.8% faster than the 3250-step tuned
baseline.

Active techniques in WR #46: Muon core; SOAP preconditioning before Muon orthogonalization on
all hidden matrices (beta2 0.90, trust gate on attn.proj); per-row u/w floor 0.3825 with radius
pin; radial brake x0.5 on outward radial component + post-pin CWD 0.025; PowerCool lr (power
1.2); momentum 0.85->0.95 warmup 300, 0.95->0.85 cooldown 200; EMA-Nesterov wrapper (gamma .99);
split beta2s on 1-D Adam (gains .99, biases .997/.9965); depth-scaled mlp.fc init; Rademacher
gain init .125; tail-EMA readout (0.4*theta + 0.6*EMA, excluding token embedding).

Reference points: plain Adam 5625 steps (undertuned); AdamH 4875; Shampoo 4100; PSGD-Kron 3375;
Muon wd .025 3375; NorMuon 3250; SpectralDescent (= Muon with mu=0) 8225 — momentum matters a lot.

## Cross-cutting themes (for the design dive)

1. **Division of labor is stable**: spectral/orthogonalized updates (Muon-family) for 2D hidden
   matrices; Adam for embeddings, head, gates, scalars. Every attempt to blur this (blend
   Muon+Adam, per-head Muon, matrices on Adam) failed or was neutral.
2. **The whitening kernel is a replaceable part**: quintic NS -> tuned per-step coeffs -> Polar
   Express -> six re-derived ANVIL maps; each swap needed lr retuning (lr interacts strongly
   with the map's output scale).
3. **Momentum has its own schedule**: warmup 0.85->0.95 (300 steps), cooldown at the end,
   twin-rail fast+slow blending in ANVIL; EMA-of-update variants (UpdateSmoothing, Snoo) on
   track 2.
4. **Weight decay came back cautiously**: zero wd optimal in 2024; wd 0.01 (track 2) then
   cautious/masked wd 1.2 scheduled with lr (track 1) — it only pays off during cooldown.
5. **Comms and optimizer are one system**: RS+AG inside step, hooks/work orders, every-other-step
   and every-4th-step Adam cadences, sparse row exchange, fp32-vs-bf16 state, mantissa sidecars.
   Many "optimizer" records are really comms-overlap records.
6. **Endgame matters**: lr floor (0.1 -> 0.15 -> 0.30), tail-EMA/tail-avg weight shipping,
   decontraction — several records are won in the last 10% of steps.
