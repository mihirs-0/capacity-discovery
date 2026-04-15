#!/usr/bin/env python3
"""Per-example gradient coherence through the D=10K plateau → escape
transition. Uses dense checkpoints from the retrained seed=3 run.

At each checkpoint, compute per-example gradients for n=500 fixed samples,
then report:
  batch_grad_norm     = ‖ḡ‖              (what SGD sees)
  rms_per_example     = √E[‖g_i‖²]        (typical individual gradient size)
  coherence           = ‖ḡ‖² / E[‖g_i‖²]  (1 if aligned, 1/n if orthogonal)
  coherence × n       = coherence * 500   (≈1.0 means random baseline)

The question: does coherence stay at 1.0 (random) all the way through
the snap, or does it spike when the loss starts dropping?
"""

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

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap


TARGET_STEPS = [100, 300, 500, 700, 900, 1000, 1100, 1200, 1300, 1400,
                1500, 1600, 1800, 2000, 2500, 3000]
CKPT_DIR = "results/basin_plateau_retrain/runs/D10000_seed3/checkpoints"
D = 10000
K = 20
LOG_K = math.log(K)
HALF_LOG_K = 0.5 * LOG_K
N_SAMPLES = 500
SAMPLE_SEED = 42
OUT_DIR = "results/gradient_trajectory"


def strip_prefix(state):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state.items()}


def load_model(ckpt, device):
    m = Transformer.from_config(ModelConfig()).to(device)
    s = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(strip_prefix(s))
    for p in m.parameters():
        p.requires_grad_(True)
    m.train()
    return m


@torch.no_grad()
def eval_full_loss(model, data, device, batch_size=2000):
    model.eval()
    total, n = 0.0, 0
    for s in range(0, data.shape[0], batch_size):
        b = data[s:s + batch_size].to(device)
        loss, _ = model(b, b)
        total += loss.item() * b.shape[0]
        n += b.shape[0]
    model.train()
    return total / n


def per_example_stats(model, samples: torch.Tensor, device: torch.device) -> dict:
    """Single forward-backward per example; return coherence statistics."""
    n = samples.shape[0]
    sum_g = None
    per_example_sq_norms = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for p in model.parameters():
            p.grad = None
        x = samples[i:i + 1].to(device)
        loss, _ = model(x, x)
        loss.backward()
        g = torch.cat([p.grad.detach().flatten() for p in model.parameters()]).cpu()
        per_example_sq_norms[i] = float((g * g).sum().item())
        if sum_g is None:
            sum_g = g.clone()
        else:
            sum_g += g

    mean_g = sum_g / n
    mean_sq_norm = float(per_example_sq_norms.mean())
    batch_sq = float((mean_g * mean_g).sum().item())
    coherence = batch_sq / mean_sq_norm if mean_sq_norm > 0 else float("nan")
    return {
        "n_samples": n,
        "rms_per_example_norm": math.sqrt(mean_sq_norm),
        "batch_grad_norm": math.sqrt(batch_sq),
        "coherence": coherence,
        "coherence_x_n": coherence * n,
    }


def find_checkpoint(step: int) -> str | None:
    p = os.path.join(CKPT_DIR, f"step_{step}.pt")
    return p if os.path.exists(p) else None


def phase_label(loss: float) -> str:
    if loss > 2.3:
        return "plateau"
    if loss > 0.5:
        return "escape"
    return "converged"


def main():
    device = torch.device("cpu")

    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
    full_data = dataset.data

    # Fixed 500-sample subset for coherence measurement
    sample_rng = np.random.RandomState(SAMPLE_SEED)
    sample_idx = sample_rng.choice(dataset.D, size=N_SAMPLES, replace=False)
    samples = full_data[sample_idx]
    print(f"Samples: {N_SAMPLES} fixed indices (seed {SAMPLE_SEED})")
    print(f"Random baseline for coherence = 1/n = {1.0/N_SAMPLES:.4e}")
    print()

    results = []
    print(f"{'step':>5}  {'loss':>7}  {'‖∇L‖':>10}  {'rms_per_ex':>10}  "
          f"{'coherence':>10}  {'coh×n':>7}  {'phase':>9}  {'wall':>6}")
    print("-" * 78)

    for step in TARGET_STEPS:
        ckpt = find_checkpoint(step)
        if ckpt is None:
            print(f"{step:>5}  [checkpoint not found]")
            continue

        t0 = time.time()
        model = load_model(ckpt, device)

        # Full-dataset loss (quick: forward only)
        loss = eval_full_loss(model, full_data, device)

        # Per-example coherence stats (n=500 single-example backprops)
        stats = per_example_stats(model, samples, device)
        dt = time.time() - t0

        row = {
            "step": step,
            "ckpt": ckpt,
            "loss": loss,
            "batch_grad_norm": stats["batch_grad_norm"],
            "rms_per_example_norm": stats["rms_per_example_norm"],
            "coherence": stats["coherence"],
            "coherence_x_n": stats["coherence_x_n"],
            "n_samples": N_SAMPLES,
            "phase": phase_label(loss),
            "wall_seconds": dt,
        }
        results.append(row)

        print(f"{step:>5}  {loss:>7.4f}  {stats['batch_grad_norm']:>10.4e}  "
              f"{stats['rms_per_example_norm']:>10.4e}  "
              f"{stats['coherence']:>10.4e}  {stats['coherence_x_n']:>7.3f}  "
              f"{row['phase']:>9}  {dt:>5.1f}s", flush=True)
        del model

    # Save JSON
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "coherence_trajectory_D10K.json")
    with open(json_path, "w") as f:
        json.dump({
            "D": D, "K": K, "n_samples": N_SAMPLES,
            "sample_seed": SAMPLE_SEED,
            "random_baseline_coherence": 1.0 / N_SAMPLES,
            "half_log_K": HALF_LOG_K,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ---- Figure 1: coherence × n vs step, with loss on right axis ----
    steps = np.array([r["step"] for r in results])
    losses = np.array([r["loss"] for r in results])
    coh_x_n = np.array([r["coherence_x_n"] for r in results])
    grad_norms = np.array([r["batch_grad_norm"] for r in results])
    rms = np.array([r["rms_per_example_norm"] for r in results])

    fig, ax = plt.subplots(figsize=(10, 6))
    c_coh = "#2ca02c"
    c_loss = "#d62728"

    ax.plot(steps, coh_x_n, "o-", color=c_coh, lw=2, ms=7,
            label="coherence × n  (random baseline = 1)")
    ax.axhline(1.0, color=c_coh, ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("coherence × n", color=c_coh)
    ax.tick_params(axis="y", labelcolor=c_coh)
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(steps, losses, "s-", color=c_loss, lw=1.5, ms=5, alpha=0.8,
             label="train loss")
    ax2.axhline(HALF_LOG_K, color=c_loss, ls=":", lw=0.8, alpha=0.6,
                label=f"0.5 log K = {HALF_LOG_K:.3f}")
    ax2.set_ylabel("train loss", color=c_loss)
    ax2.tick_params(axis="y", labelcolor=c_loss)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    ax.set_title("D=10K: per-example gradient coherence vs training step")
    fig.tight_layout()
    png1 = os.path.join(OUT_DIR, "coherence_trajectory_D10K.png")
    fig.savefig(png1, dpi=150)
    plt.close(fig)
    print(f"Saved: {png1}")

    # ---- Figure 2: ‖∇L‖ vs coherence×n ----
    fig, ax = plt.subplots(figsize=(10, 6))
    c_mag = "#1f77b4"
    c_coh = "#2ca02c"
    ax.plot(steps, grad_norms, "o-", color=c_mag, lw=2, ms=7, label="‖∇L‖ batch")
    ax.plot(steps, rms, "d--", color="#9467bd", lw=1.2, ms=5, alpha=0.7,
            label="RMS per-example ‖g_i‖")
    ax.set_xlabel("Training step")
    ax.set_ylabel("gradient magnitude", color=c_mag)
    ax.tick_params(axis="y", labelcolor=c_mag)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(steps, coh_x_n, "s-", color=c_coh, lw=1.5, ms=6, alpha=0.8,
             label="coherence × n")
    ax2.axhline(1.0, color=c_coh, ls="--", lw=0.8, alpha=0.5)
    ax2.set_ylabel("coherence × n", color=c_coh)
    ax2.tick_params(axis="y", labelcolor=c_coh)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    ax.set_title("D=10K: ‖∇L‖ vs coherence — does magnitude lead or lag alignment?")
    fig.tight_layout()
    png2 = os.path.join(OUT_DIR, "magnitude_vs_coherence_D10K.png")
    fig.savefig(png2, dpi=150)
    plt.close(fig)
    print(f"Saved: {png2}")


if __name__ == "__main__":
    main()
