#!/usr/bin/env python3
"""Per-example loss variance at the D=10K plateau vs D=100K stuck.

If D=10K has higher per-example loss variance than D=100K at matched mean,
then the "D=10K is starting to differentiate, D=100K hasn't" story is
supported — some examples are getting near-correct predictions while
others are getting confidently-wrong predictions, creating a spread of
per-example losses. If D=100K has lower variance, the model's output
distribution is more uniform across examples, explaining the smaller
per-example gradient magnitudes.

This is a forward-only measurement (no gradients needed) so it's much
faster than the per-example gradient coherence measurement.
"""

import json
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap


def strip_prefix(state):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state.items()}


def load_model(ckpt, device):
    m = Transformer.from_config(ModelConfig()).to(device)
    s = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(strip_prefix(s))
    m.eval()
    return m


@torch.no_grad()
def per_example_losses(model, data, device, batch_size=2000):
    """Return a numpy array of shape (N,) — the per-example loss, averaged
    over the 5 loss positions."""
    all_losses = []
    for start in range(0, data.shape[0], batch_size):
        batch = data[start:start + batch_size].to(device)
        _, logits = model(batch)  # (B, 16, V)
        # Loss positions: 10..14 predicting targets at 11..15
        loss_logits = logits[:, Transformer.LOSS_START:Transformer.LOSS_END]
        loss_targets = batch[:, Transformer.LOSS_START + 1:Transformer.LOSS_END + 1]
        # CE with reduction='none' → shape (B, 5)
        B, T, V = loss_logits.shape
        per_token = F.cross_entropy(
            loss_logits.reshape(-1, V),
            loss_targets.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        per_example = per_token.mean(dim=1)  # average the 5 loss positions
        all_losses.append(per_example.cpu().numpy())
    return np.concatenate(all_losses)


def summarize(losses: np.ndarray, label: str) -> dict:
    s = {
        "label": label,
        "N": int(len(losses)),
        "mean": float(losses.mean()),
        "std": float(losses.std()),
        "variance": float(losses.var()),
        "min": float(losses.min()),
        "max": float(losses.max()),
        "p05": float(np.percentile(losses, 5)),
        "p25": float(np.percentile(losses, 25)),
        "p50": float(np.percentile(losses, 50)),
        "p75": float(np.percentile(losses, 75)),
        "p95": float(np.percentile(losses, 95)),
    }
    return s


def main():
    device = torch.device("cpu")
    K = 20

    targets = [
        ("D=10K plateau (step 1200)",
         "results/basin_plateau_retrain/runs/D10000_seed3/checkpoints/step_1200.pt",
         10000),
        ("D=100K stuck (step 50000)",
         "results/phase1_d_sweep/runs/D100000_seed0/checkpoints/step_50000.pt",
         100000),
    ]

    results = []
    all_losses = {}
    for label, ckpt, D in targets:
        print(f"\n=== {label} ===")
        model = load_model(ckpt, device)
        dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
        data = dataset.data
        print(f"  dataset D={dataset.D}, K={K}")
        import time
        t0 = time.time()
        losses = per_example_losses(model, data, device, batch_size=2000)
        print(f"  forward on full dataset: {time.time()-t0:.0f}s")
        s = summarize(losses, label)
        s["D"] = D
        s["ckpt"] = ckpt
        results.append(s)
        all_losses[label] = losses
        print(f"  N                 = {s['N']}")
        print(f"  mean              = {s['mean']:.4f}")
        print(f"  std               = {s['std']:.4f}")
        print(f"  variance          = {s['variance']:.4f}")
        print(f"  min               = {s['min']:.4f}")
        print(f"  p05..p95          = {s['p05']:.4f}, {s['p25']:.4f}, "
              f"{s['p50']:.4f}, {s['p75']:.4f}, {s['p95']:.4f}")
        print(f"  max               = {s['max']:.4f}")
        del model

    # Comparison
    r10k, r100k = results
    print("\n" + "=" * 78)
    print("CROSS-D COMPARISON: per-example loss distribution")
    print("=" * 78)
    print(f"{'quantity':>20}  {'D=10K':>14}  {'D=100K':>14}  {'ratio':>10}")
    print("-" * 78)
    for key in ["mean", "std", "variance", "p05", "p25", "p50", "p75", "p95",
                "min", "max"]:
        a = r10k[key]; b = r100k[key]
        ratio = a / b if b != 0 else float("inf")
        print(f"{key:>20}  {a:>14.6f}  {b:>14.6f}  {ratio:>10.3f}")
    print()

    # Quick sanity: coefficient of variation (std / mean)
    cv_10k = r10k["std"] / r10k["mean"]
    cv_100k = r100k["std"] / r100k["mean"]
    print(f"  CV (std/mean) D=10K  = {cv_10k:.6f}")
    print(f"  CV (std/mean) D=100K = {cv_100k:.6f}")
    print(f"  CV ratio             = {cv_10k / cv_100k:.3f}")
    print()

    # Interpretation
    if r10k["std"] > 2 * r100k["std"]:
        print("  → D=10K per-example losses are MORE spread out than D=100K.")
        print("    Some examples near 0, others near uniform. This creates")
        print("    larger per-example gradients for the mispredicted examples.")
        print("    Supports: 'D=10K is starting to differentiate, D=100K hasn't'.")
    elif r100k["std"] > 2 * r10k["std"]:
        print("  → D=100K has MORE spread than D=10K. Unexpected.")
    else:
        print("  → Comparable spread. Per-example magnitude gap not explained")
        print("    by loss variance alone. Try: logit magnitude, output entropy,")
        print("    or a deeper mechanistic probe.")

    os.makedirs("results/gradient_trajectory", exist_ok=True)
    # Save summary and a coarse histogram
    hist_bins = np.linspace(0, 6, 61)
    for s, losses in zip(results, [all_losses[targets[0][0]],
                                    all_losses[targets[1][0]]]):
        hist, _ = np.histogram(losses, bins=hist_bins)
        s["hist_counts"] = hist.tolist()
    with open("results/gradient_trajectory/per_example_loss_variance.json", "w") as f:
        json.dump({
            "hist_bin_edges": hist_bins.tolist(),
            "results": results,
        }, f, indent=2)
    print("Saved: results/gradient_trajectory/per_example_loss_variance.json")

    # Plot the histograms side-by-side
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for label, losses in all_losses.items():
            ax.hist(losses, bins=hist_bins, histtype="step", lw=1.8, label=label,
                    density=True)
        ax.set_xlabel("per-example loss (averaged over 5 A+EOS positions)")
        ax.set_ylabel("density")
        ax.set_title("Per-example loss distribution at the plateau")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png = "results/gradient_trajectory/per_example_loss_hist.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"Saved: {png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
