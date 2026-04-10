"""Phase 1 specific plots: loss curves, tau vs D, per-position, z-gap, etc."""

import argparse
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.analyze import find_runs, compute_tau


def plot_loss_curves(runs_by_D: dict, K: int, output_dir: str):
    """Loss vs step for all D values. Handles early-stopped runs by
    extending the last loss to max step (model would stay there)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs_by_D)))

    # Find max step across all runs (for extension)
    max_step = max(m["step"] for runs in runs_by_D.values()
                   for r in runs for m in r["metrics"])

    for (D, runs), color in zip(sorted(runs_by_D.items()), colors):
        for r in runs:
            steps = [m["step"] for m in r["metrics"]]
            losses = [m["train_loss"] for m in r["metrics"]]
            # Plot raw curve (solid)
            ax.plot(steps, losses, color=color, alpha=0.3, linewidth=0.8)
            # Extend with dotted line if early stopped
            if steps[-1] < max_step:
                ax.plot([steps[-1], max_step], [losses[-1], losses[-1]],
                        color=color, alpha=0.2, linewidth=0.6, linestyle=":")

        # Mean curve: at each step, extend early-stopped runs at their final loss
        all_step_losses = defaultdict(list)
        all_steps = sorted(set(m["step"] for r in runs for m in r["metrics"]))
        for r in runs:
            run_steps = [m["step"] for m in r["metrics"]]
            run_losses = [m["train_loss"] for m in r["metrics"]]
            final_loss = run_losses[-1]
            final_step = run_steps[-1]
            step_idx = 0
            for s in all_steps:
                if s <= final_step:
                    while step_idx < len(run_steps) - 1 and run_steps[step_idx + 1] <= s:
                        step_idx += 1
                    all_step_losses[s].append(run_losses[step_idx])
                else:
                    all_step_losses[s].append(final_loss)
        mean_steps = sorted(all_step_losses.keys())
        mean_losses = [np.mean(all_step_losses[s]) for s in mean_steps]
        ax.plot(mean_steps, mean_losses, color=color, linewidth=2.5, label=f"D={D}")

    ax.axhline(math.log(K), color="red", linestyle="--", alpha=0.7,
               label=f"log({K}) = {math.log(K):.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss (nats)")
    ax.set_title("Training Loss vs Step (D-sweep)\n"
                 "Solid: actual data. Dotted: extension (early-stopped runs).")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=-0.1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)


def plot_tau_vs_D(runs_by_D: dict, K: int, output_dir: str):
    """Log-log plot of tau vs D with power law fit."""
    fig, ax = plt.subplots(figsize=(8, 6))

    D_vals = []
    tau_means = []
    tau_stds = []
    all_points_D = []
    all_points_tau = []

    for D in sorted(runs_by_D.keys()):
        taus = [compute_tau(r["metrics"], K, 0.5) for r in runs_by_D[D]]
        valid_taus = [t for t in taus if t is not None]
        if valid_taus:
            D_vals.append(D)
            tau_means.append(np.mean(valid_taus))
            tau_stds.append(np.std(valid_taus))
            for t in valid_taus:
                all_points_D.append(D)
                all_points_tau.append(t)

    # Individual points
    ax.scatter(all_points_D, all_points_tau, alpha=0.4, s=30, color="steelblue")

    # Mean ± std
    D_vals = np.array(D_vals)
    tau_means = np.array(tau_means)
    tau_stds = np.array(tau_stds)
    ax.errorbar(D_vals, tau_means, yerr=tau_stds, fmt="o-", color="navy",
                capsize=4, linewidth=2, markersize=6)

    # Power law fit on log scale
    if len(D_vals) >= 2:
        log_D = np.log(D_vals)
        log_tau = np.log(tau_means)
        mask = np.isfinite(log_tau)
        if mask.sum() >= 2:
            coeffs = np.polyfit(log_D[mask], log_tau[mask], 1)
            exponent = coeffs[0]
            fit_tau = np.exp(np.polyval(coeffs, log_D))
            ax.plot(D_vals, fit_tau, "--", color="red", alpha=0.7,
                    label=f"Power law: τ ∝ D^{exponent:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D (dataset size)")
    ax.set_ylabel("τ (waiting time, steps)")
    ax.set_title("Waiting Time vs Dataset Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tau_vs_D.png"), dpi=150)
    plt.close(fig)


def plot_per_position_loss(runs_by_D: dict, target_D: int, output_dir: str):
    """Per-position loss for one representative seed at a given D."""
    if target_D not in runs_by_D:
        print(f"No runs at D={target_D}")
        return

    r = runs_by_D[target_D][0]  # first seed
    steps = [m["step"] for m in r["metrics"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    for pos in range(1, 5):
        key = f"train_loss_pos{pos}"
        vals = [m.get(key, 0) for m in r["metrics"]]
        ax.plot(steps, vals, label=f"Position {pos} of A", linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (nats)")
    ax.set_title(f"Per-Position Loss (D={target_D}, seed={r['seed']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_position_loss.png"), dpi=150)
    plt.close(fig)


def plot_z_shuffle_gap(runs_by_D: dict, output_dir: str):
    """z-shuffle gap vs step for each D value."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs_by_D)))

    for (D, runs), color in zip(sorted(runs_by_D.items()), colors):
        step_to_gaps = {}
        for r in runs:
            for m in r["metrics"]:
                step_to_gaps.setdefault(m["step"], []).append(
                    m.get("z_shuffle_gap", 0))
        steps = sorted(step_to_gaps.keys())
        means = [np.mean(step_to_gaps[s]) for s in steps]
        ax.plot(steps, means, color=color, linewidth=1.5, label=f"D={D}")

    ax.set_xlabel("Step")
    ax.set_ylabel("z-shuffle gap (nats)")
    ax.set_title("z-Shuffle Gap vs Step")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "z_shuffle_gap.png"), dpi=150)
    plt.close(fig)


def plot_group_accuracy(runs_by_D: dict, output_dir: str):
    """Group accuracy vs step for each D value."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs_by_D)))

    for (D, runs), color in zip(sorted(runs_by_D.items()), colors):
        step_to_acc = {}
        for r in runs:
            for m in r["metrics"]:
                step_to_acc.setdefault(m["step"], []).append(
                    m.get("group_accuracy_frac_80", 0))
        steps = sorted(step_to_acc.keys())
        means = [np.mean(step_to_acc[s]) for s in steps]
        ax.plot(steps, means, color=color, linewidth=1.5, label=f"D={D}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of groups ≥ 80% accuracy")
    ax.set_title("Per-Group Accuracy vs Step")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "group_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_stable_ranks(runs_by_D: dict, target_D: int, output_dir: str):
    """Stable rank vs step for key weight matrices at a given D."""
    if target_D not in runs_by_D:
        print(f"No runs at D={target_D}")
        return

    r = runs_by_D[target_D][0]
    steps = [m["step"] for m in r["metrics"]]

    # Select key matrices
    keys = ["stable_rank_embed", "stable_rank_unembed",
            "stable_rank_attn_L0_Q", "stable_rank_attn_L0_V",
            "stable_rank_mlp_L0_in", "stable_rank_mlp_L0_out"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in keys:
        vals = [m.get(key, 0) for m in r["metrics"]]
        if any(v > 0 for v in vals):
            label = key.replace("stable_rank_", "")
            ax.plot(steps, vals, label=label, linewidth=1.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Stable Rank")
    ax.set_title(f"Stable Ranks Over Training (D={target_D}, seed={r['seed']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "stable_ranks.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

    runs = find_runs(args.experiment_dir)
    if not runs:
        print("No runs found.")
        return

    from collections import defaultdict
    runs_by_D = defaultdict(list)
    for r in runs:
        runs_by_D[r["D"]].append(r)

    print(f"Found {len(runs)} runs across {len(runs_by_D)} D values")

    plot_loss_curves(runs_by_D, args.K, args.output_dir)
    print("  Saved loss_curves.png")

    plot_tau_vs_D(runs_by_D, args.K, args.output_dir)
    print("  Saved tau_vs_D.png")

    plot_per_position_loss(runs_by_D, 10000, args.output_dir)
    print("  Saved per_position_loss.png")

    plot_z_shuffle_gap(runs_by_D, args.output_dir)
    print("  Saved z_shuffle_gap.png")

    plot_group_accuracy(runs_by_D, args.output_dir)
    print("  Saved group_accuracy.png")

    plot_stable_ranks(runs_by_D, 10000, args.output_dir)
    print("  Saved stable_ranks.png")


if __name__ == "__main__":
    main()
