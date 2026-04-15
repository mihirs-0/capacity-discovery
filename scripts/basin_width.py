#!/usr/bin/env python3
"""Experiment 1: basin width via random-direction parameter perturbation.

For each D in {1K, 3K, 5K, 10K, 20K}:
  - Load the converged backward-task checkpoint with lowest final loss.
  - Verify it is actually converged (loss < 0.05).
  - Sweep epsilon in logspace(-3, 2, 50) along n_directions unit-norm
    random directions in the concatenated-parameter space.
  - For each direction, report critical_epsilon = smallest epsilon whose
    loss exceeds 0.5 * log(K) (i.e., model has been kicked back onto the
    plateau).

Outputs:
  results/basin_width/basin_width.json   (raw curves + summary stats)
  results/basin_width/basin_width_vs_D.png
  results/basin_width/loss_vs_perturbation.png
"""

import argparse
import json
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap


# Lowest-final-loss seed per D (from metrics.jsonl inspection)
DEFAULT_CKPTS = {
    1000:  "results/phase1_d_sweep/runs/D1000_seed4/checkpoints/step_500.pt",
    3000:  "results/phase1_d_sweep/runs/D3000_seed1/checkpoints/step_1500.pt",
    5000:  "results/phase1_d_sweep/runs/D5000_seed0/checkpoints/step_2000.pt",
    10000: "results/phase1_d_sweep/runs/D10000_seed3/checkpoints/step_4500.pt",
    20000: "results/phase1_d_sweep/runs/D20000_seed2/checkpoints/step_17000.pt",
}


def strip_compiled_prefix(state_dict: dict) -> dict:
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()}


def load_model(ckpt_path: str, device: torch.device) -> Transformer:
    model = Transformer.from_config(ModelConfig()).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compiled_prefix(state)
    model.load_state_dict(state)
    model.eval()
    return model


def flatten_params(model: Transformer) -> torch.Tensor:
    """Flatten all trainable parameters into a single 1D tensor (detached)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()
                      if p.requires_grad])


def set_params_from_flat(model: Transformer, flat: torch.Tensor) -> None:
    """Write a flat parameter vector back into the model in-place."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(flat[offset:offset + n].view(p.shape))
            offset += n


@torch.no_grad()
def eval_full_loss(model: Transformer, data: torch.Tensor,
                   device: torch.device, batch_size: int = 2048) -> float:
    model.eval()
    total, n = 0.0, 0
    for start in range(0, data.shape[0], batch_size):
        batch = data[start:start + batch_size].to(device)
        loss, _ = model(batch, batch)
        total += loss.item() * batch.shape[0]
        n += batch.shape[0]
    return total / n


def run_one_D(D: int, ckpt_path: str, device: torch.device,
              n_directions: int, epsilons: np.ndarray,
              K: int = 20) -> dict:
    print(f"\n=== D={D}  {ckpt_path} ===", flush=True)

    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
    data = dataset.get_full()
    print(f"  dataset: D={dataset.D}  K={K}")

    model = load_model(ckpt_path, device)
    theta0 = flatten_params(model).clone()
    n_params = theta0.numel()
    base_loss = eval_full_loss(model, data, device)
    print(f"  n_params={n_params:,}  base_loss={base_loss:.4f}")

    if base_loss >= 0.05:
        print(f"  [WARN] base_loss {base_loss:.4f} >= 0.05 — not "
              f"'converged' by this experiment's criterion")

    threshold = 0.5 * math.log(K)
    print(f"  threshold (0.5 log K) = {threshold:.4f}")

    direction_results = []
    for d_idx in range(n_directions):
        # Deterministic per (D, direction) so the experiment is reproducible
        seed = D * 1000 + d_idx
        rng = np.random.RandomState(seed)
        r_np = rng.randn(n_params).astype(np.float32)
        r_np /= np.linalg.norm(r_np)
        r = torch.from_numpy(r_np).to(device)

        losses = np.full(len(epsilons), np.nan, dtype=np.float64)
        for i, eps in enumerate(epsilons):
            set_params_from_flat(model, theta0 + eps * r)
            losses[i] = eval_full_loss(model, data, device)

        # Smallest epsilon whose loss > threshold
        over = losses > threshold
        crit_eps = float(epsilons[over][0]) if over.any() else float("inf")
        print(f"  dir {d_idx}: critical_eps={crit_eps:.4g}  "
              f"(losses at eps={epsilons[0]:.3g}..{epsilons[-1]:.3g}: "
              f"{losses[0]:.3f}..{losses[-1]:.3f})")

        direction_results.append({
            "direction": d_idx,
            "seed": seed,
            "critical_eps": crit_eps,
            "losses": losses.tolist(),
        })

    # Restore
    set_params_from_flat(model, theta0)

    crit_vals = [d["critical_eps"] for d in direction_results
                 if np.isfinite(d["critical_eps"])]
    summary = {
        "D": D,
        "ckpt_path": ckpt_path,
        "n_params": int(n_params),
        "base_loss": base_loss,
        "threshold": threshold,
        "epsilons": epsilons.tolist(),
        "directions": direction_results,
        "critical_eps_mean": float(np.mean(crit_vals)) if crit_vals else None,
        "critical_eps_std":  float(np.std(crit_vals))  if crit_vals else None,
        "critical_eps_median": float(np.median(crit_vals)) if crit_vals else None,
    }
    return summary


def plot_results(results: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    Ds = [r["D"] for r in results]
    means = [r["critical_eps_mean"] for r in results]
    stds  = [r["critical_eps_std"]  for r in results]

    # --- Plot 1: critical_eps vs D, log-log, with reference slopes -----
    fig, ax = plt.subplots(figsize=(8, 6))
    # individual points
    for r in results:
        D = r["D"]
        for d in r["directions"]:
            ax.scatter([D], [d["critical_eps"]], color="steelblue",
                       alpha=0.3, s=30, zorder=3)
    ax.errorbar(Ds, means, yerr=stds, fmt="o-", color="navy", ms=8, lw=2,
                label="critical_eps (mean ± std over 5 directions)", zorder=5)

    # Reference slopes: c / sqrt(D) and c / D, fit to D=1000 mean
    if means[0] is not None and math.isfinite(means[0]):
        c_sqrt = means[0] * math.sqrt(Ds[0])
        c_lin  = means[0] * Ds[0]
        D_ref = np.array([Ds[0], Ds[-1]])
        ax.plot(D_ref, c_sqrt / np.sqrt(D_ref), "--", color="gray",
                label=r"$\propto 1/\sqrt{D}$", alpha=0.7)
        ax.plot(D_ref, c_lin / D_ref, ":", color="gray",
                label=r"$\propto 1/D$", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D (dataset size)")
    ax.set_ylabel(r"critical $\epsilon$  (smallest $\epsilon$ with loss $>$ $0.5\log K$)")
    ax.set_title("Basin width vs dataset size (random directions)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "basin_width_vs_D.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: loss vs epsilon, one curve per D -----------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    n = len(results)
    for i, r in enumerate(results):
        color = cmap(i / max(1, n - 1))
        eps = np.array(r["epsilons"])
        loss_grid = np.stack([np.array(d["losses"]) for d in r["directions"]])
        mean_loss = np.nanmean(loss_grid, axis=0)
        ax.plot(eps, mean_loss, "-", color=color, lw=2, label=f"D={r['D']}")
        # Individual direction traces as thin lines
        for row in loss_grid:
            ax.plot(eps, row, "-", color=color, lw=0.7, alpha=0.25)
    ax.axhline(0.5 * math.log(20), color="red", ls="--", lw=1,
               label=r"$0.5\,\log K$ (plateau)")
    ax.set_xscale("log")
    ax.set_xlabel(r"perturbation magnitude $\epsilon$")
    ax.set_ylabel("train loss after perturbation")
    ax.set_title("Loss vs perturbation radius")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_vs_perturbation.png"), dpi=150)
    plt.close(fig)


def print_table(results: list[dict]):
    print("\n" + "=" * 80)
    print(f"{'D':>8} {'base_loss':>10} {'crit_eps_mean':>14} "
          f"{'crit_eps_std':>14} {'* sqrt(D)':>14} {'* D':>14}")
    print("-" * 80)
    for r in results:
        D = r["D"]
        if r["critical_eps_mean"] is None:
            print(f"{D:>8} {r['base_loss']:>10.4f}  [all directions inf]")
            continue
        m = r["critical_eps_mean"]
        s = r["critical_eps_std"]
        print(f"{D:>8} {r['base_loss']:>10.4f} {m:>14.4g} {s:>14.4g} "
              f"{m * math.sqrt(D):>14.4g} {m * D:>14.4g}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-directions", type=int, default=5)
    ap.add_argument("--n-epsilons", type=int, default=50)
    ap.add_argument("--eps-min", type=float, default=1e-3)
    ap.add_argument("--eps-max", type=float, default=100.0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="results/basin_width")
    ap.add_argument("--K", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    epsilons = np.logspace(math.log10(args.eps_min), math.log10(args.eps_max),
                           args.n_epsilons)

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for D in sorted(DEFAULT_CKPTS.keys()):
        ckpt = DEFAULT_CKPTS[D]
        if not os.path.exists(ckpt):
            print(f"[skip] D={D}: checkpoint {ckpt} not found")
            continue
        r = run_one_D(D, ckpt, device,
                      n_directions=args.n_directions,
                      epsilons=epsilons, K=args.K)
        results.append(r)

    print_table(results)

    with open(os.path.join(args.out_dir, "basin_width.json"), "w") as f:
        json.dump({
            "experiment": "basin_width_experiment_1",
            "n_directions": args.n_directions,
            "n_epsilons": args.n_epsilons,
            "eps_range": [args.eps_min, args.eps_max],
            "K": args.K,
            "threshold": 0.5 * math.log(args.K),
            "results": results,
        }, f, indent=2)
    print(f"Saved: {args.out_dir}/basin_width.json")

    plot_results(results, args.out_dir)
    print(f"Saved: {args.out_dir}/basin_width_vs_D.png")
    print(f"Saved: {args.out_dir}/loss_vs_perturbation.png")


if __name__ == "__main__":
    main()
