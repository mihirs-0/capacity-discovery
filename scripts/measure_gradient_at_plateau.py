#!/usr/bin/env python3
"""Measure ||∇L|| and cos(∇L, v_min) at the D=100K seed=0 step=50000 plateau.

Decides case (a) vs case (b):
  (a) ∇L ≈ 0, negative curvature basin is too narrow — true non-escapable saddle
  (b) ∇L is small but nonzero, v_min near-orthogonal to ∇L — slow descent, not a saddle

Runs on CPU to avoid contending with a concurrent MPS training job.
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

CKPT = "results/phase1_d_sweep/runs/D100000_seed0/checkpoints/step_50000.pt"
CACHE = "results/push_escape/eigenpairs.pt"
D = 100000
K = 20


def strip_compiled_prefix(state):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state.items()}


def main():
    device = torch.device("cpu")

    print(f"Loading model from {CKPT}")
    model = Transformer.from_config(ModelConfig()).to(device)
    state = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(strip_compiled_prefix(state))
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()  # no dropout here, so equivalent to eval

    print(f"Loading cached eigenpairs from {CACHE}")
    cache = torch.load(CACHE, map_location=device, weights_only=True)
    v_min = cache["v_min"].to(torch.float32)  # (N,) already unit-norm
    v_max = cache["v_max"].to(torch.float32)
    lam_min = float(cache["lambda_min"])
    lam_max = float(cache["lambda_max"])
    print(f"  λ_min = {lam_min:+.6e}   ||v_min|| = {v_min.norm().item():.8f}")
    print(f"  λ_max = {lam_max:+.6e}   ||v_max|| = {v_max.norm().item():.8f}")

    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
    full_data = dataset.data

    # ---- 1) Full-dataset gradient ----
    print("\n[1/3] Full-dataset gradient")
    for p in model.parameters():
        p.grad = None
    batch_size = 2000
    n_batches = 0
    t0 = time.time()
    for start in range(0, full_data.shape[0], batch_size):
        batch = full_data[start:start + batch_size].to(device)
        loss, _ = model(batch, batch)
        loss.backward()  # accumulate into p.grad
        n_batches += 1
    # Average across mini-batches to match loss-mean semantics
    for p in model.parameters():
        p.grad.div_(n_batches)
    grad_full = torch.cat([p.grad.flatten() for p in model.parameters()]).detach()
    print(f"  elapsed: {time.time()-t0:.1f}s  n_batches={n_batches}")
    print(f"  ||∇L||_full = {grad_full.norm().item():.6e}")
    grad_full_unit = grad_full / grad_full.norm()
    cos_vmin_full = float((grad_full_unit @ v_min).item())
    cos_vmax_full = float((grad_full_unit @ v_max).item())
    print(f"  cos(∇L_full, v_min) = {cos_vmin_full:+.6e}")
    print(f"  cos(∇L_full, v_max) = {cos_vmax_full:+.6e}")

    # ---- 2) Subsample gradient (matches Hessian Lanczos setup: 4096 random samples) ----
    print("\n[2/3] Subsample-4096 gradient (matches Hessian Lanczos setup)")
    rng = np.random.RandomState(123)
    subset_idx = rng.choice(dataset.D, size=4096, replace=False)
    subset_data = full_data[subset_idx]

    for p in model.parameters():
        p.grad = None
    n_batches = 0
    t0 = time.time()
    for start in range(0, subset_data.shape[0], batch_size):
        batch = subset_data[start:start + batch_size].to(device)
        loss, _ = model(batch, batch)
        loss.backward()
        n_batches += 1
    for p in model.parameters():
        p.grad.div_(n_batches)
    grad_sub = torch.cat([p.grad.flatten() for p in model.parameters()]).detach()
    print(f"  elapsed: {time.time()-t0:.1f}s  n_batches={n_batches}")
    print(f"  ||∇L||_subset = {grad_sub.norm().item():.6e}")
    grad_sub_unit = grad_sub / grad_sub.norm()
    cos_vmin_sub = float((grad_sub_unit @ v_min).item())
    cos_vmax_sub = float((grad_sub_unit @ v_max).item())
    print(f"  cos(∇L_subset, v_min) = {cos_vmin_sub:+.6e}")
    print(f"  cos(∇L_subset, v_max) = {cos_vmax_sub:+.6e}")

    # ---- 3) Per-parameter gradient stats — context ----
    grad_per_param_norms = []
    param_names = []
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        g = grad_full[offset:offset + n]
        grad_per_param_norms.append((name, g.norm().item()))
        param_names.append(name)
        offset += n
    grad_per_param_norms.sort(key=lambda x: -x[1])
    print("\n[3/3] Top 10 per-parameter gradient norms (full dataset):")
    for name, g in grad_per_param_norms[:10]:
        print(f"  {name:>35s}  ||g|| = {g:.6e}")

    # Parameter norm for context
    param_norm = torch.cat([p.detach().flatten() for p in model.parameters()]).norm().item()
    print(f"\n  ||θ|| = {param_norm:.6e}  (for context)")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  λ_min = {lam_min:+.6e}")
    print(f"  λ_max = {lam_max:+.6e}")
    print(f"  ||∇L||_full   = {grad_full.norm().item():.6e}")
    print(f"  ||∇L||_subset = {grad_sub.norm().item():.6e}")
    print(f"  ||θ||         = {param_norm:.6e}")
    print(f"  cos(∇L_full, v_min)   = {cos_vmin_full:+.6e}   "
          f"(→ angle {math.degrees(math.acos(max(-1, min(1, cos_vmin_full)))):>6.2f}°)")
    print(f"  cos(∇L_subset, v_min) = {cos_vmin_sub:+.6e}   "
          f"(→ angle {math.degrees(math.acos(max(-1, min(1, cos_vmin_sub)))):>6.2f}°)")
    print(f"  cos(∇L_full, v_max)   = {cos_vmax_full:+.6e}")
    print(f"  cos(∇L_subset, v_max) = {cos_vmax_sub:+.6e}")

    summary = {
        "ckpt": CKPT,
        "cache": CACHE,
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "grad_norm_full": grad_full.norm().item(),
        "grad_norm_subset": grad_sub.norm().item(),
        "param_norm": param_norm,
        "cos_vmin_full": cos_vmin_full,
        "cos_vmin_subset": cos_vmin_sub,
        "cos_vmax_full": cos_vmax_full,
        "cos_vmax_subset": cos_vmax_sub,
        "angle_vmin_full_deg": math.degrees(math.acos(max(-1, min(1, cos_vmin_full)))),
        "angle_vmin_subset_deg": math.degrees(math.acos(max(-1, min(1, cos_vmin_sub)))),
    }
    os.makedirs("results/push_escape", exist_ok=True)
    out_path = "results/push_escape/gradient_measurement.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
