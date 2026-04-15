#!/usr/bin/env python3
"""Plot attention trajectories from checkpoints, comparing D regimes.

The key comparison: D=100K (never escapes plateau) vs D=50K (partial
transition) vs D=20K (escapes cleanly). Does z-attention rise during
the plateau at D=100K (hidden progress) or stay flat (genuine stagnation)?
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def find_runs(experiment_dir):
    runs_dir = os.path.join(experiment_dir, "runs")
    pattern = re.compile(r"D(\d+)_seed(\d+)")
    by_D = defaultdict(list)
    for name in sorted(os.listdir(runs_dir)):
        m = pattern.match(name)
        if not m:
            continue
        D = int(m.group(1))
        seed = int(m.group(2))
        run_dir = os.path.join(runs_dir, name)
        attn_path = os.path.join(run_dir, "attention_from_ckpt.jsonl")
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        if not os.path.exists(attn_path):
            continue
        attn = load_jsonl(attn_path)
        metrics = load_jsonl(metrics_path) if os.path.exists(metrics_path) else []
        if attn:
            by_D[D].append({"seed": seed, "attn": attn, "metrics": metrics})
    return by_D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str,
                        default="results/phase1_d_sweep")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

    runs_by_D = find_runs(args.experiment_dir)
    if not runs_by_D:
        print(f"No attention data found in {args.experiment_dir}")
        return

    print(f"Found attention data for D values: {sorted(runs_by_D.keys())}")

    n_layers, n_heads = 4, 4

    # Plot 1: Per-head attention for D=100K (stuck) vs D=50K vs D=20K
    interesting_D = [d for d in [20000, 50000, 100000] if d in runs_by_D]
    if len(interesting_D) >= 2:
        fig, axes = plt.subplots(n_layers, n_heads, figsize=(16, 12),
                                 sharex=False, sharey=True)
        colors = {20000: "#2ca02c", 50000: "#ff7f0e", 100000: "#d62728"}

        for li in range(n_layers):
            for hi in range(n_heads):
                ax = axes[li][hi]
                key = f"z_attn_L{li}H{hi}"
                for D in interesting_D:
                    color = colors.get(D, "gray")
                    for r in runs_by_D[D]:
                        steps = [d["step"] for d in r["attn"]]
                        vals = [d[key] for d in r["attn"]]
                        ax.plot(steps, vals, color=color, alpha=0.6,
                                linewidth=1.0,
                                label=f"D={D}" if li == 0 and hi == 0 and r is runs_by_D[D][0] else None)
                ax.set_title(f"L{li}H{hi}", fontsize=9)
                if li == n_layers - 1:
                    ax.set_xlabel("Step", fontsize=8)
                if hi == 0:
                    ax.set_ylabel("z-attention", fontsize=8)
                ax.tick_params(labelsize=7)
                # Reference line: random attention to 2 z-positions out of how many
                # are visible to the A-pred position. With causal mask at pos 10,
                # 11 positions are visible. Random uniform: 2/11 ≈ 0.18
                ax.axhline(2/11, color="gray", linestyle=":", alpha=0.4, linewidth=0.6)

        # Single legend
        handles, labels = [], []
        for D in interesting_D:
            handles.append(plt.Line2D([], [], color=colors[D], linewidth=2))
            labels.append(f"D={D}")
        handles.append(plt.Line2D([], [], color="gray", linestyle=":"))
        labels.append("uniform (2/11)")
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10,
                   bbox_to_anchor=(0.5, 1.0))
        fig.suptitle("z-attention per head: D=20K (escapes) vs D=50K (partial) vs D=100K (stuck)",
                     fontsize=11, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(args.output_dir, "attention_per_head_by_D.png"), dpi=150)
        plt.close(fig)
        print("Saved attention_per_head_by_D.png")

    # Plot 2: Layer-0 max attention vs step (the most likely routing layer)
    # for each D, with loss overlay
    fig, axes = plt.subplots(1, len(interesting_D), figsize=(18, 6), sharey=True)
    if len(interesting_D) == 1:
        axes = [axes]

    for i, D in enumerate(interesting_D):
        ax = axes[i]
        color = colors[D]
        for r in runs_by_D[D]:
            steps = [d["step"] for d in r["attn"]]
            # Take max over the 4 heads of layer 0
            max_l0 = []
            for d in r["attn"]:
                vals = [d[f"z_attn_L0H{hi}"] for hi in range(n_heads)]
                max_l0.append(max(vals))
            ax.plot(steps, max_l0, color=color, alpha=0.7, linewidth=1.5,
                    label=f"seed={r['seed']}")

        ax.axhline(2/11, color="gray", linestyle=":", alpha=0.5,
                   label="uniform")
        ax.set_xlabel("Step")
        if i == 0:
            ax.set_ylabel("max(L0 head z-attention)")
        ax.set_title(f"D={D}")
        ax.legend(fontsize=8)

        # Loss overlay
        if runs_by_D[D][0]["metrics"]:
            ax2 = ax.twinx()
            for r in runs_by_D[D]:
                if not r["metrics"]:
                    continue
                ms = [m["step"] for m in r["metrics"]]
                ls = [m["train_loss"] for m in r["metrics"]]
                ax2.plot(ms, ls, color="black", alpha=0.25, linewidth=0.8)
            ax2.set_ylabel("loss", fontsize=8, color="gray")
            ax2.tick_params(labelsize=7)

    fig.suptitle("Layer 0 max z-attention across D regimes (loss overlaid in gray)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "layer0_max_attention_by_D.png"), dpi=150)
    plt.close(fig)
    print("Saved layer0_max_attention_by_D.png")

    # Plot 3: For D=100K, check if ANY head shows monotonic rise
    # (the "hidden progress" hypothesis)
    if 100000 in runs_by_D:
        fig, ax = plt.subplots(figsize=(10, 6))
        # For each head, plot mean across seeds
        for li in range(n_layers):
            for hi in range(n_heads):
                key = f"z_attn_L{li}H{hi}"
                step_to_vals = defaultdict(list)
                for r in runs_by_D[100000]:
                    for d in r["attn"]:
                        step_to_vals[d["step"]].append(d[key])
                steps = sorted(step_to_vals.keys())
                means = [np.mean(step_to_vals[s]) for s in steps]
                # Check monotonicity: how much does it rise from start to end?
                rise = means[-1] - means[0]
                label = f"L{li}H{hi}" + (f" (Δ={rise:+.3f})" if abs(rise) > 0.01 else "")
                ax.plot(steps, means, linewidth=1.2, label=label, alpha=0.8)

        ax.axhline(2/11, color="gray", linestyle=":", alpha=0.5, label="uniform")
        ax.set_xlabel("Step")
        ax.set_ylabel("z-attention (mean across seeds)")
        ax.set_title("D=100K (never escapes plateau): per-head z-attention\n"
                     "Δ shown for heads with notable rise/fall")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "d100k_all_heads.png"), dpi=150)
        plt.close(fig)
        print("Saved d100k_all_heads.png")

    # Print numerical summary: what's the rise in max-L0 attention from
    # start to end of training, per D?
    print("\nMax(L0) z-attention rise from start to end (mean across seeds):")
    print(f"{'D':>8} {'n_seeds':>8} {'start':>10} {'end':>10} {'delta':>10}")
    print("-" * 50)
    for D in sorted(runs_by_D.keys()):
        starts = []
        ends = []
        for r in runs_by_D[D]:
            if not r["attn"]:
                continue
            first = r["attn"][0]
            last = r["attn"][-1]
            starts.append(max(first[f"z_attn_L0H{h}"] for h in range(n_heads)))
            ends.append(max(last[f"z_attn_L0H{h}"] for h in range(n_heads)))
        if starts:
            print(f"{D:>8} {len(starts):>8} {np.mean(starts):>10.4f} "
                  f"{np.mean(ends):>10.4f} {np.mean(ends) - np.mean(starts):>+10.4f}")


if __name__ == "__main__":
    main()
