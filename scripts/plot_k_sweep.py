#!/usr/bin/env python3
"""Phase 1.5 plots: K-sweep at fixed D=10000.

Verifies that tau depends on D (not K). If tau is flat across K,
the prior paper's D-not-K finding replicates.
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


def load_metrics(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def find_runs_for_K(k_dir):
    """Walk a phase1_5_k_sweep/K{K}/runs/ directory."""
    runs_dir = os.path.join(k_dir, "runs")
    if not os.path.isdir(runs_dir):
        return []
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.startswith("D"):
            continue
        parts = name.split("_")
        seed = int(parts[1][4:])
        metrics_path = os.path.join(runs_dir, name, "metrics.jsonl")
        if not os.path.exists(metrics_path):
            continue
        metrics = load_metrics(metrics_path)
        if not metrics:
            continue
        runs.append({"seed": seed, "metrics": metrics})
    return runs


def compute_tau(metrics, K, threshold_frac=0.5):
    threshold = threshold_frac * math.log(K)
    for m in metrics:
        if m["train_loss"] < threshold:
            return m["step"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str,
                        default="results/phase1_5_k_sweep", nargs="?")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

    # Find K subdirectories
    runs_by_K = {}
    for entry in sorted(os.listdir(args.experiment_dir)):
        if not entry.startswith("K"):
            continue
        try:
            K = int(entry[1:])
        except ValueError:
            continue
        k_dir = os.path.join(args.experiment_dir, entry)
        runs = find_runs_for_K(k_dir)
        if runs:
            runs_by_K[K] = runs

    if not runs_by_K:
        print(f"No K subdirectories found in {args.experiment_dir}")
        print("Expected layout: <experiment_dir>/K{5,10,20}/runs/D10000_seed{0..4}/metrics.jsonl")
        return

    print(f"Found K values: {sorted(runs_by_K.keys())}")
    for K, runs in sorted(runs_by_K.items()):
        print(f"  K={K}: {len(runs)} runs")

    # Plot 1: Loss curves grouped by K
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {5: "steelblue", 10: "seagreen", 20: "coral"}

    for K in sorted(runs_by_K.keys()):
        color = colors.get(K, "gray")
        runs = runs_by_K[K]
        for r in runs:
            steps = [m["step"] for m in r["metrics"]]
            losses = [m["train_loss"] for m in r["metrics"]]
            ax.plot(steps, losses, color=color, alpha=0.4, linewidth=0.8)
        # Mean
        step_to_losses = defaultdict(list)
        for r in runs:
            for m in r["metrics"]:
                step_to_losses[m["step"]].append(m["train_loss"])
        mean_steps = sorted(step_to_losses.keys())
        mean_losses = [np.mean(step_to_losses[s]) for s in mean_steps]
        ax.plot(mean_steps, mean_losses, color=color, linewidth=2.5,
                label=f"K={K} (log K = {math.log(K):.2f})")
        ax.axhline(math.log(K), color=color, linestyle="--", alpha=0.4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss (nats)")
    ax.set_title("Phase 1.5: Loss vs Step at D=10000 across K values")
    ax.legend()
    ax.set_ylim(bottom=-0.1)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "k_sweep_loss.png"), dpi=150)
    plt.close(fig)
    print("Saved k_sweep_loss.png")

    # Plot 2: tau vs K (should be flat if D-not-K replicates)
    fig, ax = plt.subplots(figsize=(8, 6))
    K_vals = []
    tau_means = []
    tau_stds = []
    for K in sorted(runs_by_K.keys()):
        taus = [compute_tau(r["metrics"], K, 0.5) for r in runs_by_K[K]]
        valid = [t for t in taus if t is not None]
        if valid:
            K_vals.append(K)
            tau_means.append(np.mean(valid))
            tau_stds.append(np.std(valid) if len(valid) > 1 else 0)
            for t in valid:
                ax.scatter(K, t, alpha=0.5, s=40, color="steelblue")

    K_arr = np.array(K_vals)
    means = np.array(tau_means)
    stds = np.array(tau_stds)
    ax.errorbar(K_arr, means, yerr=stds, fmt="o-", color="navy",
                capsize=5, linewidth=2, markersize=8)

    ax.set_xlabel("K (targets per group)")
    ax.set_ylabel("τ (waiting time, steps)")
    ax.set_title("Phase 1.5: τ vs K at fixed D=10000\n"
                 "(should be flat if τ depends on D, not K)")
    ax.set_xticks(sorted(runs_by_K.keys()))
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "tau_vs_K.png"), dpi=150)
    plt.close(fig)
    print("Saved tau_vs_K.png")

    # Print summary table
    print(f"\n{'K':>4}  {'n':>3}  {'tau_mean':>10}  {'tau_std':>10}  {'final_loss':>12}")
    print("-" * 50)
    for K in sorted(runs_by_K.keys()):
        runs = runs_by_K[K]
        taus = [compute_tau(r["metrics"], K, 0.5) for r in runs]
        finals = [r["metrics"][-1]["train_loss"] for r in runs]
        valid = [t for t in taus if t is not None]
        tau_mean = np.mean(valid) if valid else float("nan")
        tau_std = np.std(valid) if len(valid) > 1 else float("nan")
        final = np.mean(finals)
        print(f"{K:>4}  {len(runs):>3}  {tau_mean:>10.0f}  {tau_std:>10.0f}  {final:>12.4f}")


if __name__ == "__main__":
    main()
