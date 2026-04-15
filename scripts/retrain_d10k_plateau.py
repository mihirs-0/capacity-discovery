#!/usr/bin/env python3
"""Retrain D=10K backward briefly with dense checkpointing to capture a
plateau checkpoint (needed for Experiment 2's Hessian Lanczos).

phase1_d_sweep D=10K runs used checkpoint_every=2500 so the earliest
non-trivial checkpoint is step 2500 which is already past the escape.
This script runs the same model + data seed for 2500 steps and
checkpoints every 100 so we can pick one with loss in the plateau range
(0.8 * log K = 2.39, earlier is still on plateau).

Writes into results/basin_plateau_retrain/runs/D10000_seed{seed}/ so it
does not touch the existing phase1_d_sweep files.
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap
from src.diagnostics import compute_train_loss


def get_lr(step, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    return base_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-seed", type=int, default=3)
    ap.add_argument("--D", type=int, default=10000)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--experiment-name", type=str, default="basin_plateau_retrain")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Dataset (data_seed = D, matches phase1_d_sweep convention)
    dataset = SurjectiveMap(K=args.K, n_b=args.D // args.K, seed=args.D)
    data = dataset.get_full()
    print(f"Dataset: D={dataset.D}  K={args.K}  data_seed={args.D}")

    # Model
    torch.manual_seed(args.model_seed)
    model = Transformer.from_config(ModelConfig()).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999),
        weight_decay=0.01, eps=1e-8,
    )

    # Output
    run_dir = os.path.join(
        "results", args.experiment_name, "runs",
        f"D{args.D}_seed{args.model_seed}"
    )
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    batch_rng = np.random.RandomState(args.model_seed + 10000)
    t_start = time.time()

    def save_ckpt(step):
        path = os.path.join(ckpt_dir, f"step_{step}.pt")
        torch.save(model.state_dict(), path)

    def log(step):
        model.eval()
        m = {"step": step}
        m.update(compute_train_loss(model, data, device))
        m["wall_time_seconds"] = time.time() - t_start
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        model.train()
        return m

    save_ckpt(0)
    init = log(0)
    print(f"[step 0] train_loss={init['train_loss']:.4f}")

    for step in range(1, args.max_steps + 1):
        lr = get_lr(step, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = dataset.get_batch(args.batch_size, batch_rng).to(device)
        loss, _ = model(batch, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0:
            m = log(step)
            print(f"[step {step:>5}] train_loss={m['train_loss']:.4f}  "
                  f"pos1={m.get('train_loss_pos1', -1):.4f}")

        if step % args.checkpoint_every == 0:
            save_ckpt(step)

    wall = time.time() - t_start
    print(f"\nDone. {wall:.0f}s ({wall/60:.1f} min).")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
