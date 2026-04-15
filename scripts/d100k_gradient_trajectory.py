#!/usr/bin/env python3
"""‖∇L‖ trajectory for D=100K across all phase1_d_sweep checkpoints.

Question: at D=100K the model never escapes. Is the gradient:
  (a) flat at ~3e-3 throughout training  → "capacity wall": buildup fails entirely
  (b) growing slowly (e.g., 1e-4 → 3e-3)  → same mechanism as D=10K, just ~100× slower

This decides whether the writeup frames D=50K→100K as a sharp capacity
threshold or as a quantitative extension of the same gradient-buildup
mechanism.

Uses phase1_d_sweep/runs/D100000_seed0/checkpoints/{step_*}.pt, which
has 21 evenly-spaced checkpoints (step 0, 2500, 5000, ..., 50000).
Each gradient measurement is one forward-backward on the full D=100K
dataset on CPU (~50s per checkpoint). Total ~18 minutes.
"""

import json
import math
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap

# reuse helpers
from scripts.gradient_scaling_and_trajectory import (
    load_model, gradient_norm, eval_loss,
)


D = 100000
K = 20
LOG_K = math.log(K)
THRESHOLD = 0.5 * LOG_K

CKPT_DIR = "results/phase1_d_sweep/runs/D100000_seed0/checkpoints"
METRICS_PATH = "results/phase1_d_sweep/runs/D100000_seed0/metrics.jsonl"
OUT_DIR = "results/gradient_trajectory"


def list_checkpoints(ckpt_dir):
    out = []
    for f in os.listdir(ckpt_dir):
        m = re.search(r"step_(\d+)\.pt$", f)
        if m:
            out.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
    out.sort()
    return out


def load_metrics_by_step(path):
    d = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[r["step"]] = r
    return d


def main():
    device = torch.device("cpu")
    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
    data = dataset.data
    print(f"dataset D={dataset.D}, K={K}")

    ckpts = list_checkpoints(CKPT_DIR)
    print(f"found {len(ckpts)} checkpoints")

    metrics = load_metrics_by_step(METRICS_PATH)

    results = []
    print(f"\n{'step':>6}  {'loss':>8}  {'||∇L||':>12}  {'wall':>6}")
    print("-" * 44)
    t_total = time.time()
    for step, ckpt in ckpts:
        t0 = time.time()
        model = load_model(ckpt, device)
        gn = gradient_norm(model, data, device, batch_size=2000)
        loss = metrics.get(step, {}).get("train_loss")
        if loss is None:
            loss = eval_loss(model, data, device, batch_size=2000)
        dt = time.time() - t0
        print(f"{step:>6}  {loss:>8.4f}  {gn:>12.4e}  {dt:>5.0f}s",
              flush=True)
        results.append({"step": step, "loss": loss, "grad_norm": gn})
        del model

    total_wall = time.time() - t_total
    print(f"\nTotal: {total_wall:.0f}s ({total_wall/60:.1f} min)")

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "gradient_trajectory_D100K.json")
    with open(json_path, "w") as f:
        json.dump({"D": D, "K": K, "results": results}, f, indent=2)
    print(f"Saved: {json_path}")

    # --- Plot: mirror the D=10K trajectory plot style for direct comparison --
    steps = np.array([r["step"] for r in results])
    losses = np.array([r["loss"] for r in results])
    gns = np.array([r["grad_norm"] for r in results])

    fig, ax = plt.subplots(figsize=(10, 6))
    c_grad = "#1f77b4"
    c_loss = "#d62728"

    ax.plot(steps, gns, "o-", color=c_grad, ms=6, lw=1.8, label="‖∇L‖ (left)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("‖∇L‖", color=c_grad)
    ax.tick_params(axis="y", labelcolor=c_grad)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(steps, losses, "s-", color=c_loss, ms=5, lw=1.3, alpha=0.85,
             label="train_loss (right)")
    ax2.set_ylabel("train_loss", color=c_loss)
    ax2.tick_params(axis="y", labelcolor=c_loss)
    ax2.axhline(THRESHOLD, color=c_loss, ls=":", lw=0.8, alpha=0.5)

    # Determine growth rate: fit log(gn) vs step in the "stable" regime
    if len(gns) >= 2:
        log_gns = np.log(gns)
        slope, intercept = np.polyfit(steps, log_gns, 1)
        title_line2 = f"log(‖∇L‖) slope = {slope:+.2e} /step "
        title_line2 += f"(doubling time ≈ {math.log(2)/slope:.0f} steps)" if slope > 0 \
                       else "(not growing)"
    else:
        title_line2 = ""

    ax.set_title(f"D=100K gradient trajectory (seed=0)\n{title_line2}")

    # Legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "gradient_trajectory_D100K.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Saved: {png}")

    # Numerical summary
    print("\n--- summary ---")
    print(f"initial ‖∇L‖ (step 0):   {gns[0]:.4e}")
    print(f"final ‖∇L‖ (step {steps[-1]}): {gns[-1]:.4e}")
    print(f"ratio final/initial:     {gns[-1]/gns[0]:.4f}")
    ratio_to_d10k = gns[-1] / 0.257   # D=10K step 100 plateau floor
    print(f"final ‖∇L‖ vs D=10K plateau floor (0.257): {ratio_to_d10k:.4f}")
    print(f"final ‖∇L‖ vs D=10K just-before-escape (0.498): {gns[-1]/0.498:.4f}")


if __name__ == "__main__":
    main()
