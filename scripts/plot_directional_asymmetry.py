#!/usr/bin/env python3
"""Summary figure: directional asymmetry (forward vs backward).

Left panel : training loss vs step, forward (A->B) and backward (B,z->A),
             3 seeds each as thin lines plus seed-mean as thick line, with
             convergence step tau marked per direction.
Right panel: within-group layer-3 cosine similarity vs step for the forward
             model (from forward_convergence_over_time.json).

tau is defined as the first step at which the A-prediction loss drops below
`--tau-thresh` (default 0.1).  For backward, `train_loss` is already the
A+EOS loss.  For forward, we use `train_loss_b1` — the first B token is the
only one that actually requires solving A -> B (b2..b6 and EOS become trivial
almost immediately once the model can autoregress B's internal structure).

Output: results/forward_task/directional_asymmetry_summary.png

Usage:
    python scripts/plot_directional_asymmetry.py
"""

import argparse
import json
import os
import sys
from glob import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt


def load_metrics(path: str, loss_key: str) -> tuple[np.ndarray, np.ndarray]:
    steps, vals = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if loss_key not in r:
                continue
            steps.append(r["step"])
            vals.append(r[loss_key])
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(vals)[order]


def compute_tau(steps: np.ndarray, vals: np.ndarray,
                thresh: float) -> float | None:
    mask = vals < thresh
    if not mask.any():
        return None
    return float(steps[mask][0])


def union_grid(step_arrays: list[np.ndarray]) -> np.ndarray:
    return np.array(sorted({int(s) for arr in step_arrays for s in arr}))


def mean_on_grid(step_arrays: list[np.ndarray],
                 val_arrays: list[np.ndarray],
                 grid: np.ndarray) -> np.ndarray:
    ys = np.zeros_like(grid, dtype=np.float64)
    counts = np.zeros_like(grid, dtype=np.int64)
    for s, v in zip(step_arrays, val_arrays):
        interp = np.interp(grid, s, v, left=np.nan, right=np.nan)
        valid = ~np.isnan(interp)
        ys[valid] += interp[valid]
        counts[valid] += 1
    ys = np.where(counts > 0, ys / np.maximum(counts, 1), np.nan)
    return ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-experiment", default="forward_task")
    ap.add_argument("--backward-experiment", default="phase1_d_sweep")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n_b", type=int, default=500)
    ap.add_argument("--forward-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--backward-seeds", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--tau-thresh", type=float, default=0.1)
    ap.add_argument("--forward-loss-key", default="train_loss_b1",
                    help="Forward total loss is dominated by trivial positions; "
                         "b1 is the only one that actually tests A->B.")
    ap.add_argument("--backward-loss-key", default="train_loss")
    ap.add_argument("--convergence-json",
                    default="results/forward_task/forward_convergence_over_time.json")
    ap.add_argument("--output", default="results/forward_task/directional_asymmetry_summary.png")
    args = ap.parse_args()

    D_nominal = args.K * args.n_b

    # ---- Load forward curves ---------------------------------------------
    fwd_steps, fwd_vals, fwd_tau = [], [], []
    fwd_total_vals = []  # also keep total loss for context
    for s in args.forward_seeds:
        path = f"results/{args.forward_experiment}/runs/D{D_nominal}_seed{s}/metrics.jsonl"
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"[warn] forward seed {s}: missing or empty ({path})")
            continue
        st, val = load_metrics(path, args.forward_loss_key)
        fwd_steps.append(st)
        fwd_vals.append(val)
        _, total = load_metrics(path, "train_loss")
        fwd_total_vals.append(total)
        tau = compute_tau(st, val, args.tau_thresh)
        fwd_tau.append(tau)
        print(f"  forward seed {s}: {len(st)} points, "
              f"tau(b1<{args.tau_thresh})="
              f"{tau if tau is not None else 'not reached'}")

    # ---- Load backward curves --------------------------------------------
    bwd_steps, bwd_vals, bwd_tau = [], [], []
    for s in args.backward_seeds:
        path = f"results/{args.backward_experiment}/runs/D{D_nominal}_seed{s}/metrics.jsonl"
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"[warn] backward seed {s}: missing or empty ({path})")
            continue
        st, val = load_metrics(path, args.backward_loss_key)
        bwd_steps.append(st)
        bwd_vals.append(val)
        tau = compute_tau(st, val, args.tau_thresh)
        bwd_tau.append(tau)
        print(f"  backward seed {s}: {len(st)} points, "
              f"tau(loss<{args.tau_thresh})="
              f"{tau if tau is not None else 'not reached'}")

    fwd_mean_tau = float(np.mean([t for t in fwd_tau if t is not None])) \
        if any(t is not None for t in fwd_tau) else None
    bwd_mean_tau = float(np.mean([t for t in bwd_tau if t is not None])) \
        if any(t is not None for t in bwd_tau) else None
    ratio = (fwd_mean_tau / bwd_mean_tau) if (
        fwd_mean_tau is not None and bwd_mean_tau is not None and bwd_mean_tau > 0
    ) else None

    # ---- Convergence JSON ------------------------------------------------
    conv = None
    if os.path.exists(args.convergence_json):
        with open(args.convergence_json) as f:
            conv = json.load(f)

    # ---- Plot ------------------------------------------------------------
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: loss curves
    c_fwd = "#d62728"
    c_bwd = "#1f77b4"
    for st, v in zip(fwd_steps, fwd_vals):
        ax_l.plot(st, v, color=c_fwd, alpha=0.35, lw=1)
    for st, v in zip(bwd_steps, bwd_vals):
        ax_l.plot(st, v, color=c_bwd, alpha=0.35, lw=1)

    if fwd_steps:
        grid = union_grid(fwd_steps)
        mean = mean_on_grid(fwd_steps, fwd_vals, grid)
        ax_l.plot(grid, mean, color=c_fwd, lw=2.5,
                  label=f"forward A→B (b1 loss, n={len(fwd_steps)})")
    if bwd_steps:
        grid = union_grid(bwd_steps)
        mean = mean_on_grid(bwd_steps, bwd_vals, grid)
        ax_l.plot(grid, mean, color=c_bwd, lw=2.5,
                  label=f"backward B,z→A (n={len(bwd_steps)})")

    if fwd_mean_tau is not None:
        ax_l.axvline(fwd_mean_tau, color=c_fwd, ls="--", lw=1.1, alpha=0.8)
        ax_l.text(fwd_mean_tau, 0.95, f"  τ_fwd={fwd_mean_tau:.0f}",
                  transform=ax_l.get_xaxis_transform(),
                  color=c_fwd, fontsize=9, va="top")
    if bwd_mean_tau is not None:
        ax_l.axvline(bwd_mean_tau, color=c_bwd, ls="--", lw=1.1, alpha=0.8)
        ax_l.text(bwd_mean_tau, 0.88, f"  τ_bwd={bwd_mean_tau:.0f}",
                  transform=ax_l.get_xaxis_transform(),
                  color=c_bwd, fontsize=9, va="top")
    ax_l.axhline(args.tau_thresh, color="gray", ls=":", lw=0.8, alpha=0.6)

    title = "Training loss vs step  (D=10K, K=20)"
    if ratio is not None:
        title += f"\nτ_fwd / τ_bwd = {ratio:.2f}×"
    ax_l.set_title(title)
    ax_l.set_xlabel("Training step")
    ax_l.set_ylabel("Loss on A-prediction positions")
    ax_l.legend(fontsize=9, loc="upper right")
    ax_l.grid(True, alpha=0.3)
    ax_l.set_yscale("log")

    # Right: representation convergence
    if conv is not None:
        ax_r.plot(conv["checkpoint_steps"], conv["within_group_sim_L3"],
                  "o-", color="#2ca02c", lw=2, ms=7,
                  label="within-group cosine sim (layer 3)")
        ax_r.set_xlabel("Training step")
        ax_r.set_ylabel("Within-group cosine similarity (layer 3)")
        ax_r.set_title(f"Forward model: representation convergence\n"
                       f"(seed={conv['forward_seed']}, "
                       f"{conv['n_groups']} groups)")
        # Overlay b1 loss on secondary axis
        if "loss_steps" in conv and "train_loss_b1" in conv:
            ax_rr = ax_r.twinx()
            ax_rr.plot(conv["loss_steps"], conv["train_loss_b1"],
                       "-", color="#ff7f0e", alpha=0.6, lw=1.2,
                       label="b1 loss")
            ax_rr.set_ylabel("b1 loss", color="#ff7f0e")
            ax_rr.tick_params(axis="y", labelcolor="#ff7f0e")
            h1, l1 = ax_r.get_legend_handles_labels()
            h2, l2 = ax_rr.get_legend_handles_labels()
            ax_r.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
        else:
            ax_r.legend(loc="best", fontsize=9)
        ax_r.grid(True, alpha=0.3)
    else:
        ax_r.text(0.5, 0.5,
                  "Run analyze_convergence_over_time.py first",
                  ha="center", va="center", transform=ax_r.transAxes)
        ax_r.set_axis_off()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")

    print("\nSummary:")
    print(f"  tau_forward  (mean) = {fwd_mean_tau}")
    print(f"  tau_backward (mean) = {bwd_mean_tau}")
    print(f"  ratio (fwd / bwd)   = {ratio}")


if __name__ == "__main__":
    main()
