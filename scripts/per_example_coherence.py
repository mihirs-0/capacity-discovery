#!/usr/bin/env python3
"""Measure per-example gradient coherence at the D=10K and D=100K plateaus.

For a batch of n examples, compute:
  g_i   = ∇ L(x_i) for each example i  (per-example gradient)
  ḡ     = (1/n) Σ g_i                   (the batch gradient — what SGD sees)
  |ḡ|   = norm of the batch gradient
  rms   = √(E[‖g_i‖²])                  (rms of per-example gradient magnitude)
  coh   = |ḡ|² / E[‖g_i‖²]              (coherence: 1 if aligned, 1/n if orthogonal)
  n_eff = |ḡ|² / var(g)                 (effective-N: larger = more alignment signal)

Predictions from the gradient-trajectory story:
  - D=10K plateau: growing coherence would be consistent with the 4×
    ‖∇L‖ growth observed through steps 500..1500. We're measuring at
    a single step (1200) here so this is one point, not a trajectory.
  - D=100K stuck:  coherence should be LOWER than D=10K if the
    "per-example gradients are too incoherent to build structure"
    story is correct. If coherence is comparable but gradient magnitude
    is smaller, the story is "per-example gradients are themselves
    smaller" — different mechanism.

We use n=500 single-example backward passes per checkpoint (no vmap
tricks — just a loop, since the model and dataset are small).
"""

import json
import math
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap


def strip_prefix(state):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state.items()}


def load_model(ckpt, device):
    m = Transformer.from_config(ModelConfig()).to(device)
    s = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(strip_prefix(s))
    for p in m.parameters():
        p.requires_grad_(True)
    m.train()
    return m


def per_example_stats(model, samples: torch.Tensor, device: torch.device) -> dict:
    """Run one forward-backward per example, collect per-example gradient
    norms and the mean gradient."""
    n = samples.shape[0]
    sum_g: torch.Tensor | None = None
    per_example_sq_norms = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    for i in range(n):
        for p in model.parameters():
            p.grad = None
        x = samples[i:i + 1].to(device)
        loss, _ = model(x, x)
        loss.backward()
        g = torch.cat([p.grad.detach().flatten() for p in model.parameters()]).cpu()
        per_example_sq_norms[i] = float((g * g).sum().item())
        losses[i] = float(loss.item())
        if sum_g is None:
            sum_g = g.clone()
        else:
            sum_g += g

    mean_g = sum_g / n
    mean_sq_norm = float(per_example_sq_norms.mean())
    batch_grad_sq_norm = float((mean_g * mean_g).sum().item())
    coherence = batch_grad_sq_norm / mean_sq_norm if mean_sq_norm > 0 else float("nan")
    # Effective-n: |ḡ|² / var(g), where var = E[‖g‖²] - ‖E[g]‖²
    var_g = mean_sq_norm - batch_grad_sq_norm
    n_eff = batch_grad_sq_norm / var_g if var_g > 1e-12 else float("inf")

    return {
        "n_samples": n,
        "mean_loss": float(losses.mean()),
        "mean_per_example_sq_norm": mean_sq_norm,
        "rms_per_example_norm": float(math.sqrt(mean_sq_norm)),
        "batch_grad_norm": float(math.sqrt(batch_grad_sq_norm)),
        "coherence": coherence,
        "coherence_over_1_over_n": coherence * n,  # how many times above random
        "n_eff": n_eff,
        "var_g": var_g,
    }


def run_one(label: str, ckpt: str, D: int, K: int, n_samples: int,
            device: torch.device, sample_seed: int):
    print(f"\n=== {label} ===")
    print(f"  ckpt: {ckpt}")

    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
    rng = np.random.RandomState(sample_seed)
    idx = rng.choice(dataset.D, size=n_samples, replace=False)
    samples = dataset.data[idx]

    model = load_model(ckpt, device)
    t0 = time.time()
    stats = per_example_stats(model, samples, device)
    dt = time.time() - t0
    print(f"  {n_samples} per-example backwards in {dt:.0f}s")

    for k in ["mean_loss", "rms_per_example_norm", "batch_grad_norm",
              "coherence", "coherence_over_1_over_n", "n_eff", "var_g"]:
        v = stats[k]
        print(f"  {k:>26s} = {v:.6e}")

    stats["label"] = label
    stats["D"] = D
    stats["ckpt"] = ckpt
    stats["wall_seconds"] = dt
    del model
    return stats


def main():
    device = torch.device("cpu")
    K = 20
    n_samples = 500
    sample_seed = 42

    targets = [
        ("D=10K plateau (step 1200, before escape)",
         "results/basin_plateau_retrain/runs/D10000_seed3/checkpoints/step_1200.pt",
         10000),
        ("D=100K stuck (step 50000)",
         "results/phase1_d_sweep/runs/D100000_seed0/checkpoints/step_50000.pt",
         100000),
    ]

    results = []
    for label, ckpt, D in targets:
        r = run_one(label, ckpt, D, K, n_samples, device, sample_seed)
        results.append(r)

    # Comparison
    r10k, r100k = results
    print("\n" + "=" * 78)
    print("CROSS-D COMPARISON")
    print("=" * 78)
    print(f"{'quantity':>30}  {'D=10K':>14}  {'D=100K':>14}  {'ratio':>10}")
    print("-" * 78)
    for key, label in [
        ("mean_loss", "mean loss"),
        ("rms_per_example_norm", "RMS ‖g_i‖ (per-ex)"),
        ("batch_grad_norm", "‖ḡ‖ (batch mean)"),
        ("coherence", "coherence (|ḡ|²/E|g|²)"),
        ("coherence_over_1_over_n", "coherence × n (vs random=1)"),
        ("n_eff", "effective n"),
    ]:
        a = r10k[key]; b = r100k[key]
        ratio = a / b if b != 0 else float("inf")
        print(f"{label:>30}  {a:>14.4e}  {b:>14.4e}  {ratio:>10.3f}")
    print()

    print("Interpretation:")
    print(f"  Random-chance coherence = 1/n = 1/{n_samples} = {1/n_samples:.4e}")
    print(f"  D=10K coherence / chance = {r10k['coherence_over_1_over_n']:.3f}")
    print(f"  D=100K coherence / chance = {r100k['coherence_over_1_over_n']:.3f}")
    print()
    if r10k["rms_per_example_norm"] != 0:
        per_ex_ratio = r10k["rms_per_example_norm"] / r100k["rms_per_example_norm"]
        batch_ratio = r10k["batch_grad_norm"] / r100k["batch_grad_norm"]
        print(f"  RMS per-example norm ratio (10K / 100K): {per_ex_ratio:.2f}")
        print(f"  Batch gradient norm ratio   (10K / 100K): {batch_ratio:.2f}")
        print(f"  Coherence ratio              (10K / 100K): {r10k['coherence']/r100k['coherence']:.2f}")
        print()
        if batch_ratio / per_ex_ratio > 3:
            print("  → Most of the batch-gradient gap comes from ALIGNMENT, not per-example magnitude.")
        elif per_ex_ratio / batch_ratio > 0.7:
            print("  → Most of the batch-gradient gap comes from PER-EXAMPLE MAGNITUDE, not alignment.")
        else:
            print("  → The gap is mixed: both per-example magnitude and alignment contribute.")

    # Save
    os.makedirs("results/gradient_trajectory", exist_ok=True)
    out = {"n_samples": n_samples, "sample_seed": sample_seed, "results": results}
    path = "results/gradient_trajectory/per_example_coherence.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
