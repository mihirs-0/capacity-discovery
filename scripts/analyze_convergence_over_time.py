#!/usr/bin/env python3
"""Track within-group representation similarity over training.

Runs the convergence analysis at multiple checkpoints of the forward model
and plots layer-3 within-group similarity vs training step, overlaid with
training loss.

Expected checkpoints: step_0, step_1000, step_2000, step_5000, step_10000,
step_20000, step_50000 (from scripts/run_forward.py with
checkpoint_every=500). If an exact step is missing we fall back to the
nearest available checkpoint and flag it.

Output: results/forward_task/forward_convergence_over_time.png
        results/forward_task/forward_convergence_over_time.json

Usage:
    python scripts/analyze_convergence_over_time.py --forward-seed 0
"""

import argparse
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import ModelConfig
from src.model import Transformer
from src.task_forward import ForwardSurjectiveMap
from scripts.analyze_convergence import (
    load_model,
    residual_per_layer,
    mean_pairwise_cos,
    mean_cross_cos,
    LAYER_NAMES,
    FORWARD_PROBE_POS,
)


REQUESTED_STEPS = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
                   10000, 20000, 50000]


def list_checkpoints(run_dir: str) -> list[tuple[int, str]]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out = []
    for f in os.listdir(ckpt_dir):
        m = re.search(r"step_(\d+)\.pt$", f)
        if m:
            out.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
    out.sort()
    return out


def pick_checkpoints(available: list[tuple[int, str]],
                     requested: list[int]) -> list[tuple[int, str, int]]:
    """Return (requested_step, ckpt_path, actual_step) per requested step,
    skipping requests that have no nearby checkpoint (tolerance 250 steps)."""
    picked = []
    avail_steps = np.array([s for s, _ in available])
    for r in requested:
        if len(avail_steps) == 0:
            continue
        nearest_idx = int(np.argmin(np.abs(avail_steps - r)))
        actual = int(avail_steps[nearest_idx])
        # Allow exact matches or a small slop (one checkpoint interval)
        if abs(actual - r) <= 500:
            picked.append((r, available[nearest_idx][1], actual))
    # Deduplicate by actual checkpoint
    seen = set()
    unique = []
    for req, path, actual in picked:
        if actual not in seen:
            seen.add(actual)
            unique.append((req, path, actual))
    return unique


@torch.no_grad()
def within_and_between_L3(model: Transformer,
                           group_inputs: list[torch.Tensor],
                           n_between: int,
                           rng: np.random.RandomState) -> tuple[float, float, float]:
    """Return (within_mean, between_mean, ratio) at layer 3 over all groups."""
    all_reps = []
    for inputs in group_inputs:
        reps = residual_per_layer(model, inputs, FORWARD_PROBE_POS)
        all_reps.append(reps["after_L3"])

    n_groups = len(all_reps)
    within_vals, between_vals = [], []
    for i in range(n_groups):
        within_vals.append(mean_pairwise_cos(all_reps[i]))
        others = [j for j in range(n_groups) if j != i]
        pick = rng.choice(len(others),
                          size=min(n_between, len(others)), replace=False)
        cross = [mean_cross_cos(all_reps[i], all_reps[others[j]]) for j in pick]
        between_vals.append(float(np.mean(cross)))
    w = float(np.nanmean(within_vals))
    b = float(np.nanmean(between_vals))
    ratio = w / b if b != 0 else float("inf")
    return w, b, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-seed", type=int, default=0)
    ap.add_argument("--experiment", default="forward_task")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n_b", type=int, default=500)
    ap.add_argument("--data-seed", type=int, default=10000)
    ap.add_argument("--n-groups", type=int, default=100)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    D_nominal = args.K * args.n_b
    run_dir = os.path.join("results", args.experiment, "runs",
                           f"D{D_nominal}_seed{args.forward_seed}")
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints in {run_dir}")
    selected = pick_checkpoints(checkpoints, REQUESTED_STEPS)
    print(f"Selected {len(selected)} checkpoints:")
    for req, _, actual in selected:
        tag = "" if req == actual else f"  (requested {req})"
        print(f"  step {actual}{tag}")

    # Dataset + sampled groups (same for every checkpoint)
    dataset = ForwardSurjectiveMap(K=args.K, n_b=args.n_b, seed=args.data_seed)
    rng = np.random.RandomState(args.sampling_seed)
    valid = [gi for gi in range(dataset.n_b) if dataset.group_size(gi) >= 2]
    pick = rng.choice(len(valid),
                      size=min(args.n_groups, len(valid)), replace=False)
    group_inputs = [dataset.get_group_members(valid[i]) for i in pick]

    # Load training-loss curve from metrics.jsonl
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    loss_steps, loss_vals, loss_b1 = [], [], []
    with open(metrics_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            loss_steps.append(r["step"])
            loss_vals.append(r["train_loss"])
            loss_b1.append(r.get("train_loss_b1", float("nan")))

    # Run analysis at each checkpoint
    within_vals, between_vals, ratios = [], [], []
    actual_steps = []
    for _req, ckpt_path, actual in selected:
        model = load_model(ckpt_path, device)
        rng_bw = np.random.RandomState(args.sampling_seed + actual)
        w, b, r = within_and_between_L3(model, group_inputs,
                                        n_between=5, rng=rng_bw)
        within_vals.append(w)
        between_vals.append(b)
        ratios.append(r)
        actual_steps.append(actual)
        print(f"  step {actual}: within={w:+.4f}  between={b:+.4f}  "
              f"ratio={r:.2f}×")

    # ---- Plot ---------------------------------------------------------
    out_dir = os.path.join("results", args.experiment)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "forward_convergence_over_time.png")
    json_path = os.path.join(out_dir, "forward_convergence_over_time.json")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    c_within = "#1f77b4"
    c_between = "#9ecae1"
    c_ratio = "#2ca02c"
    ax1.plot(actual_steps, within_vals, "o-", color=c_within, lw=2, ms=7,
             label="within-group sim (layer 3)")
    ax1.plot(actual_steps, between_vals, "s-", color=c_between, lw=1.5, ms=6,
             label="between-group sim (layer 3)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Cosine similarity (layer 3)", color=c_within)
    ax1.tick_params(axis="y", labelcolor=c_within)
    ax1.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    c_loss = "#d62728"
    ax2.plot(loss_steps, loss_b1, "-", color=c_loss, lw=1.5, alpha=0.7,
             label="b1 loss")
    ax2.axhline(0.5 * np.log(20), color=c_loss, ls=":", lw=0.8, alpha=0.5)
    ax2.set_ylabel("b1 loss", color=c_loss)
    ax2.tick_params(axis="y", labelcolor=c_loss)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=9)

    ax1.set_title(
        f"Forward model (seed={args.forward_seed}): "
        f"representation convergence vs training loss\n"
        f"group structure = within/between ratio "
        f"(final: {ratios[-1]:.1f}×)"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"\nSaved: {png_path}")

    with open(json_path, "w") as f:
        json.dump({
            "forward_seed": args.forward_seed,
            "probe_position": FORWARD_PROBE_POS,
            "n_groups": len(group_inputs),
            "checkpoint_steps": actual_steps,
            "within_group_sim_L3": within_vals,
            "between_group_sim_L3": between_vals,
            "ratio_L3": ratios,
            "loss_steps": loss_steps,
            "train_loss": loss_vals,
            "train_loss_b1": loss_b1,
        }, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
