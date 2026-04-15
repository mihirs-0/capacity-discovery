#!/usr/bin/env python3
"""For each of D=10K, D=20K, D=100K plateau checkpoints, compute:
  - lambda_min via Lanczos (ncv=60, tol=1e-5, max_samples=4096)
  - ||∇L|| on the full dataset
  - cos(∇L, v_min) and the angle in degrees

The question: is the "plateau is not a saddle, v_min ⊥ ∇L" finding at
D=100K universal, or specific to D=100K? If D=10K and D=20K plateaus
ALSO have nonzero ∇L near-orthogonal to v_min, the saddle interpretation
of the plateau was wrong at every D, and the published story about
"entropic stabilization of a saddle" needs a retroactive reframing.

Runs on CPU. For D=100K, reuses the cached eigenvectors from
results/push_escape/eigenpairs.pt (no need to recompute).
"""

import argparse
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
from scipy.sparse.linalg import eigsh

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap

from scripts.hessian_lanczos import (
    strip_compiled_prefix,
    get_param_list,
    hvp_full_dataset,
    make_linear_operator,
)


TARGETS = [
    # (label, ckpt_path, D, K, plateau_loss_expected, cache_path_or_None)
    ("D=10K plateau",
     "results/basin_plateau_retrain/runs/D10000_seed3/checkpoints/step_1200.pt",
     10000, 20, 2.4628, None),
    ("D=20K plateau",
     "results/phase1_d_sweep/runs/D20000_seed0/checkpoints/step_2500.pt",
     20000, 20, 2.7446, None),
    ("D=100K plateau",
     "results/phase1_d_sweep/runs/D100000_seed0/checkpoints/step_50000.pt",
     100000, 20, 2.8662, "results/push_escape/eigenpairs.pt"),
]


def compute_lambda_min_and_vmin(model, data_for_hvp, device, batch_size, ncv, tol,
                                 maxiter, seed):
    op, n_calls = make_linear_operator(model, data_for_hvp, device,
                                        batch_size=batch_size, verbose=False)
    n_params = op.shape[0]
    rng = np.random.RandomState(seed)
    v0 = rng.randn(n_params).astype(np.float64)
    t0 = time.time()
    vals, vecs = eigsh(op, k=1, which="SA", tol=tol, maxiter=maxiter, ncv=ncv,
                       v0=v0, return_eigenvectors=True)
    dt = time.time() - t0
    lam_min = float(vals[0])
    v = vecs[:, 0]
    v /= np.linalg.norm(v)
    print(f"  λ_min = {lam_min:+.6e}  ({n_calls[0]} HVPs, {dt:.0f}s)",
          flush=True)
    return lam_min, torch.from_numpy(v.astype(np.float32)), n_params


def compute_gradient(model, data, device, batch_size):
    for p in model.parameters():
        p.grad = None
    n_batches = 0
    for start in range(0, data.shape[0], batch_size):
        batch = data[start:start + batch_size].to(device)
        loss, _ = model(batch, batch)
        loss.backward()
        n_batches += 1
    for p in model.parameters():
        p.grad.div_(n_batches)
    g = torch.cat([p.grad.detach().flatten() for p in model.parameters()])
    return g, n_batches


def analyze_one(label, ckpt_path, D, K, plateau_loss_expected, cache_path,
                device, args):
    print(f"\n=== {label}  ({ckpt_path}) ===", flush=True)
    model = Transformer.from_config(ModelConfig()).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(strip_compiled_prefix(state))
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    # Full dataset (seed=D convention)
    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
    full = dataset.data
    print(f"  dataset: D={dataset.D}  K={K}")

    # Sanity-check the loss at this checkpoint
    with torch.no_grad():
        total = 0.0
        n = 0
        for s in range(0, full.shape[0], args.batch_size):
            batch = full[s:s + args.batch_size].to(device)
            loss, _ = model(batch, batch)
            total += loss.item() * batch.shape[0]
            n += batch.shape[0]
    loss_val = total / n
    print(f"  loss on full dataset: {loss_val:.4f}  "
          f"(expected ~{plateau_loss_expected:.4f})")

    # v_min: reuse cache if available
    if cache_path and os.path.exists(cache_path) and not args.recompute_cached:
        print(f"  [cache] loading v_min from {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        lam_min = float(blob["lambda_min"])
        v_min = blob["v_min"].to(torch.float32)
        n_params = int(blob["n_params"])
        print(f"  λ_min (cached) = {lam_min:+.6e}")
    else:
        print(f"  computing v_min via Lanczos (ncv={args.ncv}, tol={args.tol}, "
              f"max_samples={args.max_samples})...")
        rng_sub = np.random.RandomState(123)
        if args.max_samples is not None and args.max_samples < dataset.D:
            idx = rng_sub.choice(dataset.D, size=args.max_samples, replace=False)
            data_for_hvp = full[idx]
        else:
            data_for_hvp = full
        lam_min, v_min, n_params = compute_lambda_min_and_vmin(
            model, data_for_hvp, device, args.batch_size, args.ncv, args.tol,
            args.maxiter, args.lanczos_seed
        )
        v_min = v_min.to(torch.float32).cpu()

    # Full-dataset gradient
    print(f"  computing ∇L on full dataset...", flush=True)
    t0 = time.time()
    grad_full, n_batches = compute_gradient(model, full, device, args.batch_size)
    grad_full = grad_full.cpu().float()
    print(f"  grad elapsed: {time.time()-t0:.0f}s  n_batches={n_batches}")
    grad_norm_full = grad_full.norm().item()
    cos_full = float((grad_full / grad_norm_full) @ v_min)
    angle_full = math.degrees(math.acos(max(-1, min(1, cos_full))))

    # Subsample gradient (same subsample as Lanczos HVPs for consistency)
    rng_sub = np.random.RandomState(123)
    if args.max_samples is not None and args.max_samples < dataset.D:
        idx = rng_sub.choice(dataset.D, size=args.max_samples, replace=False)
        data_sub = full[idx]
    else:
        data_sub = full
    grad_sub, _ = compute_gradient(model, data_sub, device, args.batch_size)
    grad_sub = grad_sub.cpu().float()
    grad_norm_sub = grad_sub.norm().item()
    cos_sub = float((grad_sub / grad_norm_sub) @ v_min)
    angle_sub = math.degrees(math.acos(max(-1, min(1, cos_sub))))

    param_norm = torch.cat([p.detach().flatten() for p in model.parameters()]
                            ).norm().item()

    # Top-10 gradient contributors
    offset = 0
    per_param = []
    for name, p in model.named_parameters():
        n = p.numel()
        per_param.append((name, grad_full[offset:offset + n].norm().item()))
        offset += n
    per_param.sort(key=lambda x: -x[1])

    result = {
        "label": label,
        "ckpt": ckpt_path,
        "D": D,
        "K": K,
        "loss": loss_val,
        "lambda_min": lam_min,
        "grad_norm_full": grad_norm_full,
        "grad_norm_subset": grad_norm_sub,
        "param_norm": param_norm,
        "cos_vmin_full": cos_full,
        "cos_vmin_subset": cos_sub,
        "angle_vmin_full_deg": angle_full,
        "angle_vmin_subset_deg": angle_sub,
        "top_params": per_param[:10],
    }

    print(f"  ||∇L||_full   = {grad_norm_full:.6e}")
    print(f"  ||∇L||_subset = {grad_norm_sub:.6e}")
    print(f"  ||θ||         = {param_norm:.6e}")
    print(f"  cos(∇L_full, v_min)   = {cos_full:+.6e}  → angle {angle_full:.2f}°")
    print(f"  cos(∇L_subset, v_min) = {cos_sub:+.6e}  → angle {angle_sub:.2f}°")
    print("  top params by grad norm:")
    for name, gn in per_param[:5]:
        print(f"    {name:>35s}  {gn:.4e}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ncv", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-samples", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--lanczos-seed", type=int, default=1)
    ap.add_argument("--recompute-cached", action="store_true")
    ap.add_argument("--out", default="results/push_escape/plateau_gradient_comparison.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Lanczos: ncv={args.ncv}, tol={args.tol}, max_samples={args.max_samples}")

    results = []
    for label, ckpt, D, K, plateau_loss, cache in TARGETS:
        r = analyze_one(label, ckpt, D, K, plateau_loss, cache, device, args)
        results.append(r)

    print("\n" + "=" * 96)
    print("CROSS-D PLATEAU SUMMARY")
    print("=" * 96)
    print(f"{'D':>6}  {'loss':>7}  {'λ_min':>12}  {'||∇L||':>12}  "
          f"{'||θ||':>10}  {'cos(∇L,v_min)':>14}  {'angle':>8}")
    print("-" * 96)
    for r in results:
        print(f"{r['D']:>6}  {r['loss']:>7.4f}  {r['lambda_min']:>+12.4e}  "
              f"{r['grad_norm_full']:>12.4e}  {r['param_norm']:>10.4e}  "
              f"{r['cos_vmin_full']:>+14.4e}  {r['angle_vmin_full_deg']:>6.2f}°")
    print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
