#!/usr/bin/env python3
"""Experiment 2: is the basin anisotropic at D=10K?

Compute v_min (Hessian smallest eigenvector) at a plateau checkpoint,
v_max at the same plateau checkpoint, and perturb the *converged* D=10K
checkpoint along v_min, v_max, and 3 random unit directions.  Report the
critical epsilon for each.

If the basin is narrow along the escape direction (v_min at plateau),
  critical_eps(v_min) < critical_eps(random) < critical_eps(v_max).

Usage:
    # Assumes scripts/retrain_d10k_plateau.py has already run and produced
    # results/basin_plateau_retrain/runs/D10000_seed3/checkpoints/step_*.pt
    python scripts/basin_anisotropy.py \
        --plateau-ckpt results/basin_plateau_retrain/runs/D10000_seed3/checkpoints/step_1000.pt \
        --converged-ckpt results/phase1_d_sweep/runs/D10000_seed3/checkpoints/step_4500.pt
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
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap

# Reuse building blocks from the existing Lanczos implementation
from scripts.hessian_lanczos import (
    strip_compiled_prefix,
    get_param_list,
    flat_to_params,
    params_to_flat,
    hvp_full_dataset,
    make_linear_operator,
    compute_loss,
)
from scripts.basin_width import (
    load_model,
    flatten_params,
    set_params_from_flat,
    eval_full_loss,
)


def lanczos_one_side(op, n_params, which: str, ncv: int, tol: float,
                     maxiter: int, seed: int) -> tuple[float, np.ndarray]:
    """Run eigsh and return (eigenvalue, eigenvector) for 'LA' or 'SA'."""
    rng = np.random.RandomState(seed)
    v0 = rng.randn(n_params).astype(np.float64)
    vals, vecs = eigsh(op, k=1, which=which, tol=tol, maxiter=maxiter,
                       ncv=ncv, v0=v0, return_eigenvectors=True)
    return float(vals[0]), vecs[:, 0]


def sweep_direction(model: Transformer, theta0: torch.Tensor,
                     direction: torch.Tensor, data: torch.Tensor,
                     device: torch.device, epsilons: np.ndarray,
                     threshold: float, label: str) -> dict:
    losses = np.full(len(epsilons), np.nan, dtype=np.float64)
    for i, eps in enumerate(epsilons):
        set_params_from_flat(model, theta0 + eps * direction)
        losses[i] = eval_full_loss(model, data, device)
    over = losses > threshold
    crit = float(epsilons[over][0]) if over.any() else float("inf")
    # restore
    set_params_from_flat(model, theta0)
    print(f"  [{label:>10s}] critical_eps = {crit:.4g}  "
          f"(loss {losses[0]:.3f}..{losses[-1]:.3f})")
    return {
        "label": label,
        "critical_eps": crit,
        "losses": losses.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plateau-ckpt", required=True,
                    help="Checkpoint at the loss plateau (loss > 0.8 log K)")
    ap.add_argument("--converged-ckpt", required=True,
                    help="Converged checkpoint to probe the basin at")
    ap.add_argument("--D", type=int, default=10000)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--data-seed", type=int, default=None,
                    help="Dataset seed (default = D)")
    ap.add_argument("--ncv", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--lanczos-seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--n-epsilons", type=int, default=50)
    ap.add_argument("--eps-min", type=float, default=1e-3)
    ap.add_argument("--eps-max", type=float, default=100.0)
    ap.add_argument("--device", type=str, default="cpu",
                    help="CPU recommended for Lanczos (double-backward + MPS "
                         "can be flaky; sweep itself is fast on any device).")
    ap.add_argument("--sweep-device", type=str, default=None,
                    help="Device for the perturbation sweep (default: --device)")
    ap.add_argument("--out-dir", type=str, default="results/basin_width")
    args = ap.parse_args()

    data_seed = args.data_seed if args.data_seed is not None else args.D
    device = torch.device(args.device)
    sweep_device = torch.device(args.sweep_device) if args.sweep_device else device
    print(f"Lanczos device: {device}")
    print(f"Sweep device:   {sweep_device}")
    print(f"Plateau ckpt:   {args.plateau_ckpt}")
    print(f"Converged ckpt: {args.converged_ckpt}")

    dataset = SurjectiveMap(K=args.K, n_b=args.D // args.K, seed=data_seed)
    data = dataset.get_full()

    log_K = math.log(args.K)
    threshold = 0.5 * log_K
    plateau_threshold = 0.8 * log_K

    # ------------------------------------------------------------------
    # Step 1: load plateau model and verify it is actually on the plateau
    # ------------------------------------------------------------------
    plateau_model = Transformer.from_config(ModelConfig()).to(device)
    state = torch.load(args.plateau_ckpt, map_location=device, weights_only=True)
    plateau_model.load_state_dict(strip_compiled_prefix(state))
    for p in plateau_model.parameters():
        p.requires_grad_(True)
    plateau_model.train()
    plateau_loss = compute_loss(plateau_model, data, device,
                                batch_size=args.batch_size)
    print(f"\nPlateau loss: {plateau_loss:.4f}  "
          f"(threshold 0.8*log K = {plateau_threshold:.4f}, log K = {log_K:.4f})")
    if plateau_loss < plateau_threshold:
        print(f"  [WARN] plateau checkpoint loss {plateau_loss:.4f} is BELOW "
              f"0.8 log K — not strictly on plateau")

    # ------------------------------------------------------------------
    # Step 2: Lanczos for v_max and v_min at the plateau
    # ------------------------------------------------------------------
    params = get_param_list(plateau_model)
    n_params = sum(p.numel() for p in params)

    op, n_calls = make_linear_operator(plateau_model, data, device,
                                        args.batch_size, verbose=True)

    print(f"\nLanczos lambda_max (ncv={args.ncv}, tol={args.tol})...")
    n_calls[0] = 0
    t0 = time.time()
    lam_max, v_max = lanczos_one_side(op, n_params, "LA",
                                      args.ncv, args.tol, args.maxiter,
                                      args.lanczos_seed)
    print(f"  lambda_max = {lam_max:+.6e}  ({n_calls[0]} HVPs, "
          f"{time.time()-t0:.0f}s)")

    print(f"\nLanczos lambda_min (ncv={args.ncv}, tol={args.tol})...")
    n_calls[0] = 0
    t0 = time.time()
    lam_min, v_min = lanczos_one_side(op, n_params, "SA",
                                      args.ncv, args.tol, args.maxiter,
                                      args.lanczos_seed + 1)
    print(f"  lambda_min = {lam_min:+.6e}  ({n_calls[0]} HVPs, "
          f"{time.time()-t0:.0f}s)")

    # Ensure eigenvectors are unit-norm float32 tensors on the sweep device
    v_max_t = torch.from_numpy(v_max.astype(np.float32)).to(sweep_device)
    v_max_t /= v_max_t.norm()
    v_min_t = torch.from_numpy(v_min.astype(np.float32)).to(sweep_device)
    v_min_t /= v_min_t.norm()

    # Free the plateau model before loading the converged one
    del plateau_model, op, params
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3: sweep each direction on the converged checkpoint
    # ------------------------------------------------------------------
    converged_model = load_model(args.converged_ckpt, sweep_device)
    theta0 = flatten_params(converged_model).clone()
    base_loss = eval_full_loss(converged_model, data, sweep_device)
    print(f"\nConverged loss: {base_loss:.4f}  (threshold {threshold:.4f})")
    if base_loss > 0.05:
        print(f"  [WARN] converged checkpoint loss {base_loss:.4f} > 0.05")

    epsilons = np.logspace(math.log10(args.eps_min), math.log10(args.eps_max),
                           args.n_epsilons)

    results = []
    # v_min, v_max
    results.append(sweep_direction(converged_model, theta0, v_min_t, data,
                                    sweep_device, epsilons, threshold, "v_min"))
    results.append(sweep_direction(converged_model, theta0, v_max_t, data,
                                    sweep_device, epsilons, threshold, "v_max"))
    # Random directions
    for i in range(args.n_random):
        rng = np.random.RandomState(999 + i)
        r_np = rng.randn(n_params).astype(np.float32)
        r_np /= np.linalg.norm(r_np)
        r_t = torch.from_numpy(r_np).to(sweep_device)
        results.append(sweep_direction(converged_model, theta0, r_t, data,
                                        sweep_device, epsilons, threshold,
                                        f"random_{i+1}"))

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"{'direction':>12}  {'critical_eps':>14}")
    print("-" * 50)
    for r in results:
        print(f"{r['label']:>12}  {r['critical_eps']:>14.4g}")
    print()

    # Ratios vs random baseline
    rand_crits = [r["critical_eps"] for r in results
                  if r["label"].startswith("random") and math.isfinite(r["critical_eps"])]
    if rand_crits:
        rand_median = float(np.median(rand_crits))
        v_min_eps = next(r["critical_eps"] for r in results if r["label"] == "v_min")
        v_max_eps = next(r["critical_eps"] for r in results if r["label"] == "v_max")
        print(f"median(random)            = {rand_median:.4g}")
        print(f"v_min / median(random)    = {v_min_eps / rand_median:.3f}"
              f"  (< 1 ⇒ basin is TIGHTER along the escape direction)")
        print(f"v_max / median(random)    = {v_max_eps / rand_median:.3f}"
              f"  (> 1 ⇒ basin is LOOSER along the sharpest curvature)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"v_min": "#d62728", "v_max": "#1f77b4"}
    for r in results:
        label = r["label"]
        color = colors.get(label, "#808080")
        style = "-" if label in ("v_min", "v_max") else "--"
        lw = 2.2 if label in ("v_min", "v_max") else 1.2
        ax.plot(epsilons, r["losses"], style, color=color, lw=lw, label=label)
    ax.axhline(threshold, color="red", ls=":", lw=1,
               label=f"0.5 log K = {threshold:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"perturbation magnitude $\epsilon$")
    ax.set_ylabel("train loss after perturbation")
    ax.set_title(
        f"Basin anisotropy at D=10K converged checkpoint\n"
        f"λ_min={lam_min:+.3e}  λ_max={lam_max:+.3e}  (at plateau loss={plateau_loss:.3f})"
    )
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "basin_anisotropy.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")

    # ------------------------------------------------------------------
    # JSON dump
    # ------------------------------------------------------------------
    out = {
        "experiment": "basin_anisotropy_experiment_2",
        "plateau_ckpt": args.plateau_ckpt,
        "converged_ckpt": args.converged_ckpt,
        "plateau_loss": plateau_loss,
        "base_loss_converged": base_loss,
        "threshold": threshold,
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "n_params": int(n_params),
        "ncv": args.ncv,
        "tol": args.tol,
        "epsilons": epsilons.tolist(),
        "results": results,
    }
    json_path = os.path.join(args.out_dir, "basin_anisotropy.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
