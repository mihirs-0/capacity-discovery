#!/usr/bin/env python3
"""Compute per-head attention to z-positions from saved checkpoints.

Uses the existing checkpoints from results/phase1_d_sweep/ to compute
attention trajectories retroactively. No retraining needed.

For each D value, walks all checkpoints in step order and computes:
  - z_attn_LiHj: how much A-prediction positions attend to z-positions
  - z_reads_b_LiHj: how much z-positions attend to B-positions

Output: results/phase1_d_sweep/runs/D{D}_seed{S}/attention_from_ckpt.jsonl
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import ExperimentConfig, TaskConfig, ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap


# Sequence layout constants (verified)
Z_POSITIONS = [8, 9]
A_PRED_POSITIONS = [10, 11, 12, 13]
B_POSITIONS = [1, 2, 3, 4, 5, 6]


def strip_compiled_prefix(state_dict):
    """Strip _orig_mod. prefix added by torch.compile."""
    new = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new[k[len("_orig_mod."):]] = v
        else:
            new[k] = v
    return new


def compute_z_attention(model, batch, n_layers, n_heads):
    """Run forward_with_attention and compute per-head scores."""
    model.eval()
    with torch.no_grad():
        _, all_attn = model.forward_with_attention(batch)

    result = {}
    for li, attn in enumerate(all_attn):
        # attn: (B, n_heads, T, T)
        for hi in range(n_heads):
            # A-prediction positions attending to z-positions
            z_attn = attn[:, hi][:, A_PRED_POSITIONS][:, :, Z_POSITIONS]
            # (B, 4, 2): sum over z, mean over a, mean over batch
            score = z_attn.sum(dim=-1).mean(dim=-1).mean(dim=0).item()
            result[f"z_attn_L{li}H{hi}"] = score

            # z-positions attending to B-positions
            z_reads_b = attn[:, hi][:, Z_POSITIONS][:, :, B_POSITIONS]
            score_b = z_reads_b.sum(dim=-1).mean(dim=-1).mean(dim=0).item()
            result[f"z_reads_b_L{li}H{hi}"] = score_b

    return result


def find_d_seed_runs(experiment_dir):
    """Yield (D, seed, run_dir) for each run with checkpoints."""
    runs_dir = os.path.join(experiment_dir, "runs")
    pattern = re.compile(r"D(\d+)_seed(\d+)")
    for name in sorted(os.listdir(runs_dir)):
        m = pattern.match(name)
        if not m:
            continue
        D = int(m.group(1))
        seed = int(m.group(2))
        run_dir = os.path.join(runs_dir, name)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda f: int(re.search(r"step_(\d+)", f).group(1)),
        )
        if ckpts:
            yield D, seed, run_dir, ckpts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str,
                        default="results/phase1_d_sweep")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Examples used to compute attention")
    parser.add_argument("--filter-d", type=int, nargs="+", default=None,
                        help="Only process these D values")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Device: {device}")

    # Default model config (matches what was trained)
    model_cfg = ModelConfig()
    n_layers = model_cfg.n_layers
    n_heads = model_cfg.n_heads

    # Cache datasets per D value (we need a batch from the same data the
    # model was trained on, with the same data_seed)
    dataset_cache = {}

    runs = list(find_d_seed_runs(args.experiment_dir))
    if args.filter_d:
        runs = [r for r in runs if r[0] in args.filter_d]

    print(f"Processing {len(runs)} runs\n")

    for D, seed, run_dir, ckpts in runs:
        # data_seed in our sweeps is set to D (run_fast.py phase1_configs)
        if D not in dataset_cache:
            dataset_cache[D] = SurjectiveMap(
                K=args.K, n_b=D // args.K, seed=D
            )
        dataset = dataset_cache[D]

        # Build model fresh
        model = Transformer.from_config(model_cfg).to(device)

        # Fixed batch for attention extraction
        rng = np.random.RandomState(999)
        batch = dataset.get_batch(min(args.batch_size, dataset.D), rng).to(device)

        out_path = os.path.join(run_dir, "attention_from_ckpt.jsonl")
        with open(out_path, "w") as f:
            for ckpt_name in ckpts:
                step = int(re.search(r"step_(\d+)", ckpt_name).group(1))
                ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                state = strip_compiled_prefix(state)
                model.load_state_dict(state)

                metrics = {"step": step}
                metrics.update(compute_z_attention(model, batch, n_layers, n_heads))
                f.write(json.dumps(metrics) + "\n")

        print(f"  D={D:>6} seed={seed}: {len(ckpts)} checkpoints -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
