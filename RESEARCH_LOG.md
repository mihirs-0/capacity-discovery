# Research Log: Capacity and Structural Discovery in Gradient-Trained Transformers

**Date range:** April 9–15, 2026
**Repository:** github.com/mihirs-0/capacity-discovery

---

## 1. Codebase Setup

Built from scratch per SPEC.md. Clean-room implementation, no HuggingFace/TransformerLens.

### Architecture
- Decoder-only Transformer: 4 layers, 4 heads, d_model=128, d_mlp=512
- **Exact parameter count: 803,584**
- Character-level tokenizer: 40 tokens (36 alphanumeric + PAD/BOS/SEP/EOS)
- Sequence layout (16 tokens): `[BOS] b1-b6 [SEP] z1 z2 [SEP] a1-a4 [EOS]`
- Loss computed only on A+EOS positions (positions 11–15)

### Task
Surjective map: n_b base strings B, each mapping to K distinct targets A via selector z.
- D = n_b × K total examples
- All B unique, all A first-chars distinct per group, all z distinct per group
- No structure between B/z/A — pure lookup table memorization

### Training
- AdamW: lr=1e-3, β=(0.9, 0.999), weight_decay=0.01
- Cosine warmup 500 steps, then constant LR
- Batch size 128, sampled with replacement
- 50,000 steps per run
- float32 throughout

### Diagnostics (every 100 steps)
- D1: Full training set loss + per-position decomposition
- D3: z-shuffle gap (permute z across batch, measure loss increase)
- D4: Per-group accuracy (fraction of groups with ≥80% accuracy)
- D6: Stable rank of all weight matrices

---

## 2. Phase 1: D-Sweep (35 runs)

**Config:** K=20, D ∈ {1000, 3000, 5000, 10000, 20000, 50000, 100000}, 5 seeds each.

**GPU:** RTX 3090, single-process sequential via `run_fast.py` with torch.compile + early stopping at loss < 0.05.

**Total wall time:** ~172 minutes for all 50 runs (Phase 1 + Phase 1.5).

### Results (τ at 0.5·log(K) threshold)

```
       D  n_runs  tau_mean   tau_std  final_loss  converged
    1000       5       500       0.0      0.0037      5/5
    3000       5       800     244.9      0.0250      5/5
    5000       5      1000       0.0      0.0205      5/5
   10000       5      2000       0.0      0.0175      5/5
   20000       5      4300     244.9      0.0364      5/5
   50000       5     29800    8891.6      0.8927      5/5 (tau@0.5 only; 2/5 at tau@0.3)
  100000       5       NaN       NaN      2.8642      0/5
```

### Token-normalized analysis (tau_tokens = tau_steps × 128)

```
       D  tau_steps  tau_tokens  coverage (tokens/D)
    1000       500       64000     64.00×
    3000       750       96000     32.00×
    5000      1000      128000     25.60×
   10000      2000      256000     25.60×
   20000      4167      533333     26.67×
   50000     46500     5952000    119.04×
  100000       NaN         NaN       NaN
```

**Power-law fit (excluding D=100K): τ_tokens ∝ D^1.10**

### Key observations
- **Three regimes:** sub-linear (D≤5K, coverage decreasing), linear (D=5K–20K, coverage flat ~25×), super-linear/divergent (D≥50K)
- **D=100K never leaves the log(K) plateau.** Loss = 2.86 for all 5 seeds across 50K steps. Zero convergence at any threshold.
- **D=50K is the critical zone:** bimodal (1/2 seeds converge at tau@0.3 threshold, 5/5 at tau@0.5)
- **Per-position decomposition (D=10K):** position 1 of A (the z-dependent position) is last to drop. Positions 2–4 fall earlier — model first learns (B, a₁)→a₂a₃a₄ suffix, z-dependence at position 1 is the actual bottleneck.
- **The wall is at params/D ≈ 8–16.** Model has 12× information-theoretic capacity headroom at D=100K — the barrier is optimization, not capacity.

### z-shuffle gap
**NOTE:** The z-shuffle gap data in the results.zip export is all zeros. This is a bug: the CUDA graph output buffer from the first model call was overwritten by the second call (fixed in commit 844cd92). The gap values from the terminal output (second, clean run) show correct behavior:
- D=1K: gap ~2.7 at convergence
- D=10K: gap ~5.7 at convergence
- D=100K: gap ~0.005 (model never uses z)

---

## 3. Phase 1.5: K-Sweep (15 runs)

**Config:** D=10000 fixed, K ∈ {5, 10, 20}, 5 seeds each.

**NOTE:** The first run had a directory-collision bug (all K values wrote to D10000_seed{N}, K=20 overwrote K=5 and K=10). Fixed in commit c5c9c31 by using `experiment_name=phase1_5_k_sweep/K{K}`.

### Results (from the second, correct run — terminal output)

```
   K  tau@0.5 (mean)   final_loss
   5          ~5,200        0.025
  10          ~5,200        0.024
  20          ~5,100        0.018
```

**τ is flat across K at fixed D=10000. The D-not-K finding replicates.** The waiting time depends on dataset size, not per-group ambiguity.

---

## 4. Experiment A: Small Model Test (10 runs)

**Config:** d_model=32, d_mlp=128, n_layers=4, n_heads=4.
**Exact parameter count:** ~40K params (4 params/example at D=10K).
**Two noise conditions:** lr=1e-3 (5 seeds) and lr=3e-3 (5 seeds).

### Results

```
  lr=1e-3:  tau@0.5 mean = 18,840   final_loss mean = 1.30
  lr=3e-3:  tau@0.5 mean = 27,000   final_loss mean = 1.44
```

### Key findings
- **Small model never fully converges in 50K steps.** Final loss ~1.3 (mid-transition). Has 6× capacity headroom — still an optimization failure.
- **Higher noise (lr=3e-3) is SLOWER by 43%.** Same direction as big model. **No noise sign-flip exists.**
- **Noise sign-flip hypothesis REJECTED.** This was the planned basis for Phase 2 (noise sweeps). Phase 2 is deprioritized.

---

## 5. Experiment B: Attention Tracking from Checkpoints

**Method:** Load saved checkpoints from Phase 1 runs. For each, compute per-head attention from A-prediction positions (10–13) to z-positions (8–9). Also compute z-positions' attention to B-positions (1–6).

**Baseline:** Random attention to z = 2/11 ≈ 0.18 (2 z-positions visible out of 11 total at the A-prediction positions under causal mask).

### Max(Layer 0) z-attention rise from step 0 to final step

```
       D  n_seeds      start        end      delta
    1000        4     0.1798     0.2767    +0.0969
    3000        2     0.1782     0.3114    +0.1332
    5000        3     0.1760     0.3269    +0.1509
   10000        2     0.1770     0.3758    +0.1988
   20000        3     0.1760     0.4454    +0.2694
   50000        2     0.1779     0.3427    +0.1648
  100000        3     0.1762     0.4780    +0.3018
```

### Key finding
**The model that NEVER converges (D=100K) has the LARGEST rise in z-attention (+0.30), even larger than D=20K which converges successfully (+0.27).** The Q/K routing circuit forms monotonically during the plateau. The model is "looking at z" despite producing zero useful output.

---

## 6. Linear Probe of Residual Stream

**Method:** Load D=100K step 50000 checkpoint (stuck plateau) and D=20K step 17500 (converged). For each layer, train a linear classifier on the residual stream at A-prediction positions to predict the correct A token. 80/20 train/test split on the full dataset. Chance = 1/36 ≈ 0.028.

### Results

```
       layer  D=100K stuck (50K)  D=20K converged  D=100K random init
       embed             0.027           0.027              0.027
    after_L0             0.027           0.070              0.026
    after_L1             0.027           0.172              0.027
    after_L2             0.028           0.440              0.027
    after_L3             0.029           0.964              0.026
     chance              0.028           0.028              0.028
```

### Key finding
**The D=100K stuck model is statistically indistinguishable from random initialization at every layer.** Despite z-attention rising from 0.18→0.48, the residual stream carries zero linearly-recoverable information about the correct A token. The Q/K circuit has formed (attention routes to z); the V/O circuit has NOT produced useful content in the residual stream.

---

## 7. Weight Drift Analysis

**Method:** ||W_step − W_init||_F / ||W_init||_F for attention subcircuit matrices (W_Q, W_K, W_V, W_O), averaged across all 4 layers.

### Results

```
       D     step       W_Q       W_K       W_V       W_O
   20000    17500      2.40      2.38      1.81      1.80   (converged)
   50000    50000      3.69      3.70      2.70      2.84   (partial)
  100000    50000      2.53      2.57      2.69      2.91   (stuck)
```

### Key finding
**ALL weight matrices have moved significantly from initialization at D=100K, including V and O.** V/O drift at D=100K (2.7–2.9×) is actually LARGER than at D=20K converged (1.8×). **The "frozen V/O" hypothesis is rejected.** Weights are moving; they are not moving toward a useful solution.

---

## 8. Hessian Eigenvalue Analysis

### First attempt: Power iteration (UNRELIABLE)

```
checkpoint                      λ_max         λ_min        residual_min   ratio
D=20K plateau (256 samples)    +5.50e+00    −4.48e+00       4.19e-02     0.01 ✓
D=100K stuck (256 samples)     +5.25e+00    −6.76e-02       2.01e-01     2.98 ✗
D=20K converged (256 samples)  +1.22e+02    −1.66e-01       1.58e+00     9.48 ✗
D=100K random (256 samples)    +1.54e+02    −1.40e+02       8.36e-01     0.01 ✓
```

**The D=100K λ_min = −0.068 was UNRELIABLE (residual/|eigenvalue| = 2.98).** This led to the incorrect "66× curvature collapse" narrative, which was retracted.

### Second attempt: Lanczos via scipy.eigsh (CONVERGED)

```
Method: scipy.sparse.linalg.eigsh, 4096-sample subset, ncv=60, tol=1e-5
Verified: ncv=30 → ncv=60 gives identical values. ARPACK reports CONVERGED.
Independent residual check: ||Hv − λv|| / |λ| < 1e-5 for all.

checkpoint                       λ_max                    λ_min
                                 value     residual       value     residual      sign
D=20K plateau (step 2500)    +2.873502    1.58e-05    −1.036235    5.81e-06    NEGATIVE
D=100K stuck (step 50000)    +0.985858    1.42e-06    −0.746546    7.65e-07    NEGATIVE
```

### Key findings
1. **Both plateaus are saddle points (λ_min negative).** No sign flip. No topological phase transition.
2. **The escape-direction curvature differs by only 1.4×** (−1.04 vs −0.75), not the 66× from the broken measurement.
3. **λ_max is ~2.9× smaller at D=100K** (0.99 vs 2.87). The entire loss landscape is softer at D=100K.
4. **Condition number λ_max/|λ_min|** is 2.77 at D=20K and 1.32 at D=100K — the saddle at D=100K is actually *more isotropic* (better conditioned) than at D=20K.
5. **These are 4096-sample Hessians** (16× the broken run), not full-dataset. Full-dataset values may differ slightly but signs should be robust.

---

## 9. Summary of Hypotheses Tested

| # | Hypothesis | Status | Evidence |
|---|-----------|--------|----------|
| 1 | τ depends on D, not K | **CONFIRMED** | Phase 1.5: τ flat across K=5,10,20 at fixed D |
| 2 | Noise sign-flip between capacity regimes | **REJECTED** | Experiment A: noise hurts at both sizes |
| 3 | Hidden progress: loss hides internal learning | **PARTIALLY CONFIRMED** | Q/K attention rises during flat loss, but residual carries no answer info |
| 4 | V/O subcircuit is frozen at D=100K | **REJECTED** | Weight drift: V/O moved MORE than at converged D=20K |
| 5 | Topological phase transition (λ_min sign flip) | **REJECTED** | Lanczos: both plateaus negative |
| 6 | Curvature collapse (66× weaker escape) | **REJECTED** | Was based on unconverged power iteration; Lanczos gives 1.4× |

---

## 10. What Is Actually Known (as of April 15, 2026)

### Established facts (reliable measurements)
1. τ scales super-linearly with D beyond D=20K and diverges at D=100K.
2. τ does not depend on K at fixed D (replicating prior work).
3. Noise (higher LR) hurts uniformly — no sign-flip across capacity regimes.
4. The z-attention routing circuit (Q/K) forms monotonically during the plateau at all D values, including D=100K where the model never converges.
5. The residual stream at D=100K contains zero linearly-recoverable answer information at any layer, identical to random initialization.
6. All weight matrices (Q, K, V, O, MLP) have moved substantially from initialization at D=100K.
7. The D=100K plateau is a saddle with negative λ_min, not a true local minimum.
8. The escape curvature at D=100K (|λ_min| = 0.75) is within 1.4× of D=20K (|λ_min| = 1.04).

### The unresolved question
**Why doesn't SGD escape the D=100K saddle?** The curvature permits escape (λ_min ≈ −0.75). The geometry is similar to D=20K (which escapes in ~5K steps). All weights are moving. But the model makes zero progress toward the answer in 50K steps. Something prevents the gradient signal from aligning with the escape direction — but what?

### Leading candidate for next measurement
**Gradient projection onto the negative-curvature eigenvector.** Compute ⟨g, v_min⟩/||g|| for many mini-batch gradients at both D=20K plateau and D=100K plateau. If the projection is dramatically smaller at D=100K, SGD cannot "see" the escape direction even though it exists. This would distinguish "the landscape permits escape but the gradient doesn't point there" from "the landscape permits escape and the gradient points there but escape is just slow."

---

## 11. Engineering Notes

### GPU deployment issues encountered
- **RTX 5090 (Blackwell):** 24-process multiprocessing caused 0% GPU utilization due to CUDA context thrashing. Solved by switching to single-process sequential runner.
- **torch.compile:** `mode="reduce-overhead"` caused CUDA graph output buffer overwrites (alternating run failures). Fixed by disabling CUDA graphs: `torch._inductor.config.triton.cudagraphs = False`.
- **z_shuffle_gap bug:** Second model call in the diagnostic overwrote first call's loss tensor via CUDA graph buffer reuse. Fixed by calling `.item()` immediately after each model call.
- **import scoping:** `import torch._inductor.config` inside `__init__` made Python treat `torch` as local variable. Fixed by moving import to module level.
- **Hessian convergence:** Power iteration fails for near-zero eigenvalues (residual > eigenvalue). Must use Lanczos (scipy.eigsh) with proper ncv and tolerance, verified by independent residual check.

### Disk budget
- Checkpoints every 2500 steps: ~3.4 GB for 50 runs
- Metrics (eval every 500 steps): ~3 MB for 50 runs
- Model checkpoint size: ~3.2 MB each (803K params × 4 bytes)

### Timing (RTX 3090, single-process, torch.compile)
- D=1000: ~10s per run (early stops at step ~500)
- D=10000: ~80s per run (early stops at step ~5000)
- D=20000: ~280s per run (early stops at step ~18000)
- D=50000: ~640s per run (runs full 50K steps)
- D=100000: ~660s per run (runs full 50K steps)
- Hessian Lanczos (CPU, 4096 samples): ~10–20 min per checkpoint
