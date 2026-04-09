#!/usr/bin/env python3
"""Plot Experiment B: Z-attention tracking results."""

import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    run_dir = os.path.join(PROJECT_ROOT, "results", "attention_tracking", "seed_0")
    output_dir = os.path.join(PROJECT_ROOT, "results", "attention_tracking", "plots")
    os.makedirs(output_dir, exist_ok=True)

    attn_path = os.path.join(run_dir, "attention.jsonl")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    if not os.path.exists(attn_path):
        print(f"No attention data at {attn_path}")
        return

    attn_data = load_jsonl(attn_path)
    metrics_data = load_jsonl(metrics_path) if os.path.exists(metrics_path) else []

    attn_steps = [d["step"] for d in attn_data]
    metric_steps = [d["step"] for d in metrics_data]
    losses = [d["train_loss"] for d in metrics_data]

    n_layers = 4
    n_heads = 4

    # Plot 1: All 16 heads, z-attention over training
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(16, 12), sharex=True, sharey=True)
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi]
            key = f"z_attn_L{li}H{hi}"
            vals = [d[key] for d in attn_data]
            ax.plot(attn_steps, vals, linewidth=0.8, color="steelblue")
            ax.set_title(f"L{li}H{hi}", fontsize=9)
            if li == n_layers - 1:
                ax.set_xlabel("Step", fontsize=8)
            if hi == 0:
                ax.set_ylabel("z-attention", fontsize=8)
            ax.tick_params(labelsize=7)

            # Overlay loss on secondary axis
            if losses:
                ax2 = ax.twinx()
                ax2.plot(metric_steps, losses, linewidth=0.5, color="red", alpha=0.3)
                ax2.tick_params(labelsize=6)
                if hi == n_heads - 1:
                    ax2.set_ylabel("loss", fontsize=7, color="red")

    fig.suptitle("Per-Head Attention to z-positions (from A-prediction positions)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_heads_z_attention.png"), dpi=150)
    plt.close(fig)
    print("Saved all_heads_z_attention.png")

    # Plot 2: Layer 0 heads only (most likely routing heads)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for hi in range(n_heads):
        key = f"z_attn_L0H{hi}"
        vals = [d[key] for d in attn_data]
        ax.plot(attn_steps, vals, linewidth=1.5, color=colors[hi],
                label=f"L0H{hi}", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("z-attention score")
    ax.set_title("Layer 0 Heads: Attention to z-positions")
    ax.legend()

    if losses:
        ax2 = ax.twinx()
        ax2.plot(metric_steps, losses, linewidth=1.5, color="gray",
                 alpha=0.4, linestyle="--", label="loss")
        ax2.set_ylabel("Training loss (nats)", color="gray")
        ax2.axhline(math.log(20), color="red", linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "layer0_z_attention.png"), dpi=150)
    plt.close(fig)
    print("Saved layer0_z_attention.png")

    # Plot 3: z-reads-B (reverse direction)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(16, 12), sharex=True, sharey=True)
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi]
            key = f"z_reads_b_L{li}H{hi}"
            vals = [d.get(key, 0) for d in attn_data]
            ax.plot(attn_steps, vals, linewidth=0.8, color="darkorange")
            ax.set_title(f"L{li}H{hi}", fontsize=9)
            if li == n_layers - 1:
                ax.set_xlabel("Step", fontsize=8)
            if hi == 0:
                ax.set_ylabel("z-reads-B", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("Per-Head: How much z-positions attend to B-positions", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_heads_z_reads_b.png"), dpi=150)
    plt.close(fig)
    print("Saved all_heads_z_reads_b.png")


if __name__ == "__main__":
    main()
