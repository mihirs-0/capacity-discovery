#!/usr/bin/env python3
"""Plot Experiment A: Small model test results."""

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


def find_tau(metrics, K, threshold=0.5):
    thresh = threshold * math.log(K)
    for m in metrics:
        if m["train_loss"] < thresh:
            return m["step"]
    return None


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results", "small_model_test")
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    K = 20
    log_K = math.log(K)

    # Collect runs by lr
    runs_by_lr = defaultdict(list)
    for lr_dir in sorted(os.listdir(results_dir)):
        if not lr_dir.startswith("lr_"):
            continue
        runs_path = os.path.join(results_dir, lr_dir, "runs")
        if not os.path.isdir(runs_path):
            continue
        for run_name in sorted(os.listdir(runs_path)):
            metrics_path = os.path.join(runs_path, run_name, "metrics.jsonl")
            if not os.path.exists(metrics_path):
                continue
            metrics = load_metrics(metrics_path)
            if not metrics:
                continue
            lr_label = lr_dir.replace("lr_", "lr=")
            runs_by_lr[lr_label].append({"name": run_name, "metrics": metrics})

    if not runs_by_lr:
        print("No results found.")
        return

    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"lr=1e-03": "steelblue", "lr=3e-03": "coral"}

    for lr_label, runs in sorted(runs_by_lr.items()):
        color = colors.get(lr_label, "gray")
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
                label=f"{lr_label} (n={len(runs)})")

    ax.axhline(log_K, color="red", linestyle="--", alpha=0.7,
               label=f"log({K}) = {log_K:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss (nats)")
    ax.set_title("Small Model (~40K params) Loss Curves")
    ax.legend()
    ax.set_ylim(bottom=-0.1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "small_model_loss.png"), dpi=150)
    plt.close(fig)
    print(f"Saved small_model_loss.png")

    # Print tau summary
    print(f"\n{'lr':>12}  {'seed':>4}  {'tau_50':>8}  {'final_loss':>10}")
    print("-" * 45)
    for lr_label in sorted(runs_by_lr.keys()):
        taus = []
        for r in runs_by_lr[lr_label]:
            tau = find_tau(r["metrics"], K, 0.5)
            final = r["metrics"][-1]["train_loss"]
            seed = r["name"]
            taus.append(tau)
            print(f"{lr_label:>12}  {seed:>4}  "
                  f"{str(tau) if tau else 'N/A':>8}  {final:>10.4f}")
        valid = [t for t in taus if t is not None]
        if valid:
            print(f"{'mean':>12}  {'':>4}  {np.mean(valid):>8.0f}")
        else:
            print(f"{'mean':>12}  {'':>4}  {'N/A':>8}")
        print()


if __name__ == "__main__":
    main()
