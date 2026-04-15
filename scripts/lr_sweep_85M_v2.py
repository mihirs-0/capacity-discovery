#!/usr/bin/env python3
"""LR sweep for 85M model with early stopping.

Stops early if:
  - loss < 0.1 (escaped and converged)
  - z_gap < 0.01 for 2000 consecutive steps (subcritical, not going to escape)
  - loss > 10 (diverged)
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

from src.config import ModelConfig, TaskConfig, TrainingConfig, EvalConfig, ExperimentConfig
from src.model import Transformer
from src.task import SurjectiveMap
from src.diagnostics import compute_train_loss, compute_z_shuffle_gap

K = 2
N_B = 5000
DATA_SEED = 42
MODEL_SEED = 0
MAX_STEPS = 5000
EVAL_EVERY = 100

LRS = [1e-2, 3e-2, 1e-1]  # skip 1e-3, 3e-3 (already have data)

# Early stopping
CONVERGED_THRESH = 0.1
DIVERGED_THRESH = 10.0
SUBCRITICAL_ZGAP = 0.01
SUBCRITICAL_PATIENCE = 2000  # steps of z_gap < threshold before stopping


def get_lr(step, warmup, base):
    return base * step / max(1, warmup) if step < warmup else base


def run_one(lr: float, device: torch.device):
    label = f"lr_{lr:g}"
    run_dir = os.path.join("results", "lr_sweep_85M", label, "runs",
                           f"D{K*N_B}_seed{MODEL_SEED}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    print(f"\n{'='*60}")
    print(f"LR = {lr:g}  (85M model, K={K}, D={K*N_B})")
    print(f"{'='*60}")

    dataset = SurjectiveMap(K=K, n_b=N_B, seed=DATA_SEED)
    full_data = dataset.get_full()

    torch.manual_seed(MODEL_SEED)
    model = Transformer(
        n_layers=12, n_heads=12, d_model=768, d_mlp=3072,
        vocab_size=40, max_seq_len=16,
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999),
        weight_decay=0.01, eps=1e-8,
    )
    batch_rng = np.random.RandomState(MODEL_SEED + 10000)
    t0 = time.time()

    subcritical_steps = 0
    stopped_reason = None

    for step in range(MAX_STEPS + 1):
        # Eval
        if step % EVAL_EVERY == 0:
            model.eval()
            loss_m = compute_train_loss(model, full_data, device)
            z_m = compute_z_shuffle_gap(model, full_data, device, batch_size=1024)
            m = {"step": step, **loss_m, **z_m,
                 "wall_time_seconds": time.time() - t0}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(m) + "\n")

            loss = m["train_loss"]
            zgap = m["z_shuffle_gap"]

            if step % 500 == 0:
                print(f"  [lr={lr:g}] step {step:>4}  loss={loss:.4f}  "
                      f"z_gap={zgap:.4f}  wall={m['wall_time_seconds']:.0f}s",
                      flush=True)

            # Early stopping checks
            if loss < CONVERGED_THRESH:
                stopped_reason = f"converged (loss={loss:.4f} < {CONVERGED_THRESH})"
                print(f"  ** EARLY STOP: {stopped_reason} at step {step}")
                break
            if loss > DIVERGED_THRESH:
                stopped_reason = f"diverged (loss={loss:.4f} > {DIVERGED_THRESH})"
                print(f"  ** EARLY STOP: {stopped_reason} at step {step}")
                break
            if zgap < SUBCRITICAL_ZGAP:
                subcritical_steps += EVAL_EVERY
            else:
                subcritical_steps = 0
            if subcritical_steps >= SUBCRITICAL_PATIENCE and step >= SUBCRITICAL_PATIENCE:
                stopped_reason = f"subcritical (z_gap < {SUBCRITICAL_ZGAP} for {subcritical_steps} steps)"
                print(f"  ** EARLY STOP: {stopped_reason} at step {step}")
                break

            model.train()

        if step == MAX_STEPS:
            break

        # LR schedule
        lr_now = get_lr(step, 500, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Train step
        batch = dataset.get_batch(128, batch_rng).to(device)
        loss_val, _ = model(batch, batch)

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            stopped_reason = f"NaN/Inf loss at step {step}"
            print(f"  ** EARLY STOP: {stopped_reason}")
            m = {"step": step, "train_loss": float("nan"), "z_shuffle_gap": 0,
                 "wall_time_seconds": time.time() - t0}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(m) + "\n")
            break

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    wall = time.time() - t0
    with open(metrics_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    final = rows[-1] if rows else {}
    result = {
        "lr": lr,
        "final_step": final.get("step", 0),
        "final_loss": final.get("train_loss"),
        "final_z_gap": final.get("z_shuffle_gap"),
        "stopped_reason": stopped_reason or "completed",
        "wall_time_seconds": wall,
        "n_evals": len(rows),
    }
    print(f"  Done: {wall:.0f}s ({wall/60:.1f} min). "
          f"Final: loss={result['final_loss']:.4f}, z_gap={result['final_z_gap']:.4f}. "
          f"Reason: {result['stopped_reason']}")

    # Save summary
    with open(os.path.join(run_dir, "..", "..", "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    results = []
    for lr in LRS:
        r = run_one(lr, device)
        results.append(r)

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'lr':>8}  {'final_step':>10}  {'final_loss':>10}  "
          f"{'final_z_gap':>10}  {'reason':>20}")
    print("-" * 70)
    for r in results:
        print(f"{r['lr']:>8g}  {r['final_step']:>10}  "
              f"{r['final_loss']:>10.4f}  {r['final_z_gap']:>10.4f}  "
              f"{r['stopped_reason']:>20}")

    with open("results/lr_sweep_85M/sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/lr_sweep_85M/sweep_summary.json")


if __name__ == "__main__":
    main()
