#!/usr/bin/env python3
"""K=20 test: does SP+Adam failure generalize beyond K=2?

Runs:
  1. 803K at K=20 D=10K (reference — should nucleate)
  2. 85M at K=20 D=10K SP+Adam eps=1e-8 (THE TEST)
  3. (conditional) 85M at K=20 D=10K eps=1e-4 (if run 2 fails)

Fully autonomous.
"""

import csv
import json
import math
import os
import sys
import time
import types
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import Transformer
from src.task import SurjectiveMap
from src.diagnostics import compute_train_loss, compute_z_shuffle_gap

K = 20
N_B = 500
DATA_SEED = 10000  # matches phase1_d_sweep convention (seed=D)
MODEL_SEED = 0
BATCH_SIZE = 128
WARMUP = 500
MAX_STEPS = 5000
EVAL_EVERY = 100

OUT_DIR = "results/overparameterization/k20_test"

CONVERGE_THRESH = 0.1
DIVERGE_THRESH = 100.0
SUBCRITICAL_ZGAP = 0.01
SUBCRITICAL_PATIENCE = 2000


def run_one(label, n_layers, n_heads, d_model, d_mlp, lr, eps, device):
    run_dir = os.path.join(OUT_DIR, label)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.csv")

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {n_layers}L/{n_heads}H/{d_model}D/{d_mlp}MLP, lr={lr}, eps={eps}")
    print(f"  K={K}, D={K*N_B}, data_seed={DATA_SEED}")
    print(f"{'='*60}", flush=True)

    dataset = SurjectiveMap(K=K, n_b=N_B, seed=DATA_SEED)
    full_data = dataset.get_full()

    torch.manual_seed(MODEL_SEED)
    model = Transformer(
        n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_mlp=d_mlp,
        vocab_size=40, max_seq_len=16,
    ).to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999),
        weight_decay=0.01, eps=eps,
    )

    batch_rng = np.random.RandomState(MODEL_SEED + 10000)
    t0 = time.time()
    subcritical_steps = 0
    stopped_reason = None
    rows = []

    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "z_shuffle_gap", "wall_seconds"])

    for step in range(MAX_STEPS + 1):
        if step % EVAL_EVERY == 0:
            model.eval()
            lm = compute_train_loss(model, full_data, device)
            zm = compute_z_shuffle_gap(model, full_data, device, batch_size=1024)
            wall = time.time() - t0
            row = {"step": step, "train_loss": lm["train_loss"],
                   "z_shuffle_gap": zm["z_shuffle_gap"], "wall_seconds": round(wall, 1)}
            rows.append(row)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([step, row["train_loss"],
                                        row["z_shuffle_gap"], row["wall_seconds"]])

            loss = row["train_loss"]
            zgap = row["z_shuffle_gap"]

            if step % 500 == 0:
                print(f"  [{label}] step {step:>4}  loss={loss:.4f}  "
                      f"z_gap={zgap:.4f}  wall={wall:.0f}s", flush=True)

            if loss < CONVERGE_THRESH:
                stopped_reason = f"converged (loss={loss:.4f})"
                print(f"  ** EARLY STOP: {stopped_reason} at step {step}")
                break
            if loss > DIVERGE_THRESH or math.isnan(loss):
                stopped_reason = f"diverged (loss={loss:.1f})"
                print(f"  ** EARLY STOP: {stopped_reason} at step {step}")
                break
            if zgap < SUBCRITICAL_ZGAP:
                subcritical_steps += EVAL_EVERY
            else:
                subcritical_steps = 0
            if subcritical_steps >= SUBCRITICAL_PATIENCE and step >= SUBCRITICAL_PATIENCE:
                stopped_reason = f"subcritical ({subcritical_steps} steps)"
                print(f"  ** EARLY STOP: {stopped_reason} at step {step}")
                break
            model.train()

        if step == MAX_STEPS:
            break

        lr_now = lr * min(1.0, step / max(1, WARMUP))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        batch = dataset.get_batch(BATCH_SIZE, batch_rng).to(device)
        loss_val, _ = model(batch, batch)
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            stopped_reason = f"NaN at step {step}"
            print(f"  ** EARLY STOP: {stopped_reason}")
            break
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    wall = time.time() - t0
    final = rows[-1] if rows else {}
    nucleated = any(r["z_shuffle_gap"] > 0.05 for r in rows[5:])

    result = {
        "label": label, "K": K, "D": K * N_B,
        "n_layers": n_layers, "n_heads": n_heads, "d_model": d_model,
        "n_params": model.count_parameters(),
        "lr": lr, "eps": eps,
        "final_step": final.get("step", 0),
        "final_loss": final.get("train_loss"),
        "final_z_gap": final.get("z_shuffle_gap"),
        "nucleated": nucleated,
        "stopped_reason": stopped_reason or "completed",
        "wall_seconds": round(wall, 1),
    }
    print(f"  Done: {wall:.0f}s. loss={result['final_loss']:.4f} "
          f"z_gap={result['final_z_gap']:.4f} nucleated={'YES' if nucleated else 'no'}")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def plot_results(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"803K_K20": "#1f77b4", "85M_K20_SP": "#d62728",
              "85M_K20_eps1e-4": "#2ca02c"}
    for r in results:
        label = r["label"]
        csv_path = os.path.join(OUT_DIR, label, "metrics.csv")
        if not os.path.exists(csv_path):
            continue
        steps, zgaps = [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                zgaps.append(max(float(row["z_shuffle_gap"]), 1e-4))
        ax.plot(steps, zgaps, "o-", color=colors.get(label, "gray"),
                lw=2, ms=4, label=f"{label} ({r['n_params']//1000}K params)")
    ax.axhline(0.05, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("z_shuffle_gap")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 10)
    ax.set_title(f"K=20 D=10K: does the 85M SP+Adam failure generalize?")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "k20_test.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/k20_test.png")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"K={K}, D={K*N_B}, data_seed={DATA_SEED}")
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []

    # Run 1: 803K reference
    r = run_one("803K_K20", n_layers=4, n_heads=4, d_model=128, d_mlp=512,
                lr=1e-3, eps=1e-8, device=device)
    results.append(r)

    # Run 2: 85M SP+Adam (THE TEST)
    r = run_one("85M_K20_SP", n_layers=12, n_heads=12, d_model=768, d_mlp=3072,
                lr=1e-3, eps=1e-8, device=device)
    results.append(r)

    # Run 3 (conditional): if run 2 failed, try eps=1e-4
    if not results[-1]["nucleated"]:
        print("\n  85M K=20 SP failed — running eps=1e-4 follow-up")
        r = run_one("85M_K20_eps1e-4", n_layers=12, n_heads=12,
                    d_model=768, d_mlp=3072, lr=1e-3, eps=1e-4, device=device)
        results.append(r)
    else:
        print("\n  85M K=20 SP SUCCEEDED — no follow-up needed")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'label':>18}  {'K':>3}  {'params':>10}  {'eps':>8}  "
          f"{'loss':>8}  {'z_gap':>8}  {'nucleated':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:>18}  {r['K']:>3}  {r['n_params']:>10,}  "
              f"{r['eps']:>8g}  {r['final_loss']:>8.4f}  "
              f"{r['final_z_gap']:>8.4f}  "
              f"{'YES' if r['nucleated'] else 'no':>10}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_results(results)

    # DONE
    with open(os.path.join(OUT_DIR, "DONE.txt"), "w") as f:
        f.write(f"Completed: {datetime.now()}\n")
        nucleated_names = [r["label"] for r in results if r["nucleated"]]
        f.write(f"Nucleated: {', '.join(nucleated_names) or 'none'}\n")
    print(f"\nDONE. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
