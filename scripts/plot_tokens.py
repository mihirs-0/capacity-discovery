#!/usr/bin/env python3
"""Token-normalized scaling analysis: tau in tokens-seen vs D.

Converts step counts to actual data exposure (tau_tokens = tau_steps * batch_size).
Reveals coverage: how many times each example was seen before convergence.

Usage:
  python scripts/plot_tokens.py [results/phase1_d_sweep] [--K 20] [--batch-size 128]
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from scripts.analyze import find_runs, compute_tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str,
                        default="results/phase1_d_sweep", nargs="?")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold-frac", type=float, default=0.5,
                        help="Fraction of log(K) for tau threshold (default 0.5)")
    parser.add_argument("--exclude-d", type=int, nargs="+", default=[100000],
                        help="D values to exclude from power-law fit")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

    runs = find_runs(args.experiment_dir)
    if not runs:
        print(f"No runs found in {args.experiment_dir}")
        return

    runs_by_D = defaultdict(list)
    for r in runs:
        runs_by_D[r["D"]].append(r)

    # Compute tau in steps and tokens for each run
    rows = []
    for D in sorted(runs_by_D.keys()):
        taus = [compute_tau(r["metrics"], args.K, args.threshold_frac)
                for r in runs_by_D[D]]
        valid = [t for t in taus if t is not None]
        if not valid:
            tau_mean = None
            tau_std = None
            tokens_mean = None
            coverage = None
        else:
            tau_mean = float(np.mean(valid))
            tau_std = float(np.std(valid)) if len(valid) > 1 else 0.0
            tokens_mean = tau_mean * args.batch_size
            coverage = tokens_mean / D
        rows.append({
            "D": D,
            "n_total": len(taus),
            "n_converged": len(valid),
            "tau_steps": tau_mean,
            "tau_steps_std": tau_std,
            "tau_tokens": tokens_mean,
            "coverage": coverage,
            "individual_taus": valid,
        })

    # Print table
    print(f"\n{'D':>8} {'n':>4} {'tau_steps':>12} {'tau_tokens':>14} "
          f"{'coverage':>10}  {'tokens/D'}")
    print("-" * 65)
    for row in rows:
        D = row["D"]
        n = f"{row['n_converged']}/{row['n_total']}"
        if row["tau_steps"] is None:
            print(f"{D:>8} {n:>4} {'NaN':>12} {'NaN':>14} {'NaN':>10}  (did not converge)")
        else:
            print(f"{D:>8} {n:>4} {row['tau_steps']:>12.0f} "
                  f"{row['tau_tokens']:>14.0f} {row['coverage']:>10.2f}x")

    # Build arrays for plotting
    all_D = []
    all_tau_tokens = []
    mean_D = []
    mean_tau = []
    mean_std = []
    for row in rows:
        if row["tau_steps"] is None:
            continue
        for t in row["individual_taus"]:
            all_D.append(row["D"])
            all_tau_tokens.append(t * args.batch_size)
        mean_D.append(row["D"])
        mean_tau.append(row["tau_tokens"])
        mean_std.append(row["tau_steps_std"] * args.batch_size)

    if not mean_D:
        print("\nNo converged runs to plot.")
        return

    mean_D = np.array(mean_D)
    mean_tau = np.array(mean_tau)
    mean_std = np.array(mean_std)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))

    # Individual seeds (light)
    ax.scatter(all_D, all_tau_tokens, alpha=0.35, s=40, color="steelblue",
               label="Individual seeds")

    # Means with error bars (dark)
    ax.errorbar(mean_D, mean_tau, yerr=mean_std, fmt="o-", color="navy",
                capsize=5, linewidth=2, markersize=8, label="Mean ± std")

    # Power-law fit (excluding specified D values)
    fit_mask = ~np.isin(mean_D, args.exclude_d)
    fit_D = mean_D[fit_mask]
    fit_tau = mean_tau[fit_mask]
    if len(fit_D) >= 2:
        log_D = np.log(fit_D)
        log_tau = np.log(fit_tau)
        coeffs = np.polyfit(log_D, log_tau, 1)
        exponent = coeffs[0]
        intercept = coeffs[1]
        # Plot fit over the full range
        D_plot = np.array([mean_D.min(), mean_D.max()])
        tau_fit = np.exp(intercept) * D_plot ** exponent
        ax.plot(D_plot, tau_fit, "--", color="red", alpha=0.8, linewidth=2,
                label=f"Power law fit: τ_tokens ∝ D^{exponent:.2f}")
    else:
        exponent = None

    # Linear reference: tau_tokens ∝ D (slope 1 in log-log)
    # Anchor at the smallest D point
    if len(mean_D) > 0:
        ref_D = np.array([mean_D.min(), mean_D.max()])
        ref_anchor = mean_tau[0]
        ref_tau = ref_anchor * (ref_D / mean_D.min())  # slope = 1
        ax.plot(ref_D, ref_tau, ":", color="gray", alpha=0.6, linewidth=1.5,
                label="Linear reference: τ_tokens ∝ D")

    # Mark excluded points
    excl_mask = np.isin(mean_D, args.exclude_d)
    if excl_mask.any():
        ax.scatter(mean_D[excl_mask], mean_tau[excl_mask],
                   s=140, facecolors="none", edgecolors="red", linewidth=2,
                   label=f"Excluded from fit: D={list(mean_D[excl_mask])}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D (dataset size)")
    ax.set_ylabel("τ_tokens = τ_steps × batch_size  (tokens seen)")
    title = f"Token-Normalized Waiting Time vs Dataset Size (K={args.K})"
    if exponent is not None:
        title += f"\nFit exponent: {exponent:.3f}"
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "tau_tokens_vs_D.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved tau_tokens_vs_D.png to {args.output_dir}")

    if exponent is not None:
        print(f"\nPower-law fit (excluding D={args.exclude_d}):")
        print(f"  tau_tokens ∝ D^{exponent:.3f}")
        print(f"  Linear (slope 1) would mean: each example needs constant exposure")
        if exponent > 1.05:
            print(f"  Exponent > 1: examples need MORE exposure as D grows (super-linear)")
        elif exponent < 0.95:
            print(f"  Exponent < 1: examples need LESS exposure as D grows (sub-linear)")
        else:
            print(f"  Exponent ≈ 1: linear scaling")


if __name__ == "__main__":
    main()
