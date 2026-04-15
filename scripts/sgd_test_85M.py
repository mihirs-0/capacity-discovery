#!/usr/bin/env python3
"""SGD vs Adam test for 85M model: is nucleation failure due to Adam's
per-parameter normalization or the architecture itself?

5 runs:
  1. SGD lr=0.1, no momentum     — raw gradient, high lr
  2. SGD lr=0.3, no momentum     — even higher
  3. SGD lr=1.0, no momentum     — extreme (may diverge)
  4. SGD lr=0.1, momentum=0.9    — SGD with momentum
  5. Adam lr=1e-3, 4 heads       — same 85M params, fewer heads (bystander test)

All: K=2, D=10K, data_seed=42, model_seed=0, batch=128, 5000 steps,
eval_every=100. Early stop on divergence (loss>100) or subcritical
(z_gap<0.01 for 2000 steps).
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

from src.model import Transformer
from src.task import SurjectiveMap
from src.diagnostics import compute_train_loss, compute_z_shuffle_gap

K = 2
N_B = 5000
DATA_SEED = 42
MODEL_SEED = 0
BATCH_SIZE = 128
MAX_STEPS = 5000
EVAL_EVERY = 100
WARMUP = 500

DIVERGE_THRESH = 100.0
SUBCRITICAL_ZGAP = 0.01
SUBCRITICAL_PATIENCE = 2000

RUNS = [
    {"label": "SGD_lr0.1",       "opt": "sgd",  "lr": 0.1, "mom": 0.0, "wd": 0.0, "n_heads": 12},
    {"label": "SGD_lr0.3",       "opt": "sgd",  "lr": 0.3, "mom": 0.0, "wd": 0.0, "n_heads": 12},
    {"label": "SGD_lr1.0",       "opt": "sgd",  "lr": 1.0, "mom": 0.0, "wd": 0.0, "n_heads": 12},
    {"label": "SGD_mom_lr0.1",   "opt": "sgd",  "lr": 0.1, "mom": 0.9, "wd": 0.0, "n_heads": 12},
    {"label": "Adam_4head",      "opt": "adam", "lr": 1e-3, "mom": None,"wd": 0.01,"n_heads": 4},
]

OUT_DIR = "results/overparameterization/sgd_test"


def get_lr(step, warmup, base):
    return base * step / max(1, warmup) if step < warmup else base


def run_one(cfg, device):
    label = cfg["label"]
    run_dir = os.path.join(OUT_DIR, label)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    print(f"\n{'='*60}")
    print(f"{label}  (opt={cfg['opt']}, lr={cfg['lr']}, "
          f"n_heads={cfg['n_heads']}, mom={cfg.get('mom')})")
    print(f"{'='*60}")

    dataset = SurjectiveMap(K=K, n_b=N_B, seed=DATA_SEED)
    full_data = dataset.get_full()

    torch.manual_seed(MODEL_SEED)
    model = Transformer(
        n_layers=12, n_heads=cfg["n_heads"], d_model=768, d_mlp=3072,
        vocab_size=40, max_seq_len=16,
    ).to(device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,}  (n_heads={cfg['n_heads']}, "
          f"d_head={768//cfg['n_heads']})")

    if cfg["opt"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg["lr"],
            momentum=cfg.get("mom", 0.0),
            weight_decay=cfg.get("wd", 0.0),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"],
            betas=(0.9, 0.999), weight_decay=cfg.get("wd", 0.01), eps=1e-8,
        )

    batch_rng = np.random.RandomState(MODEL_SEED + 10000)
    t0 = time.time()
    subcritical_steps = 0
    stopped_reason = None

    for step in range(MAX_STEPS + 1):
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
                print(f"  [{label}] step {step:>4}  loss={loss:.4f}  "
                      f"z_gap={zgap:.4f}  wall={m['wall_time_seconds']:.0f}s",
                      flush=True)

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

        lr_now = get_lr(step, WARMUP, cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        batch = dataset.get_batch(BATCH_SIZE, batch_rng).to(device)
        loss_val, _ = model(batch, batch)

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            stopped_reason = f"NaN at step {step}"
            m = {"step": step, "train_loss": float("nan"), "z_shuffle_gap": 0,
                 "wall_time_seconds": time.time() - t0}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"  ** EARLY STOP: {stopped_reason}")
            break

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    wall = time.time() - t0
    with open(metrics_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    final = rows[-1] if rows else {}

    # Detect nucleation: z_gap > 0.05 sustained for 500+ steps
    nucleated = False
    nucleation_step = None
    above_count = 0
    for r in rows:
        if r["z_shuffle_gap"] > 0.05:
            above_count += EVAL_EVERY
            if above_count >= 500 and nucleation_step is None:
                nucleation_step = r["step"] - 500
                nucleated = True
        else:
            above_count = 0

    result = {
        "label": label,
        "optimizer": cfg["opt"],
        "lr": cfg["lr"],
        "momentum": cfg.get("mom"),
        "n_heads": cfg["n_heads"],
        "n_params": n_params,
        "final_step": final.get("step", 0),
        "final_loss": final.get("train_loss"),
        "final_z_gap": final.get("z_shuffle_gap"),
        "stopped_reason": stopped_reason or "completed",
        "nucleated": nucleated,
        "nucleation_step": nucleation_step,
        "wall_time_seconds": wall,
    }
    print(f"  Done: {wall:.0f}s. Final: loss={result['final_loss']:.4f}, "
          f"z_gap={result['final_z_gap']:.4f}. "
          f"Nucleated={nucleated}. Reason: {result['stopped_reason']}")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def plot_results(results, ref_803k_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = {
        "SGD_lr0.1": "#1f77b4",
        "SGD_lr0.3": "#ff7f0e",
        "SGD_lr1.0": "#2ca02c",
        "SGD_mom_lr0.1": "#9467bd",
        "Adam_4head": "#d62728",
    }

    # Load 803K reference
    ref_rows = []
    if os.path.exists(ref_803k_path):
        with open(ref_803k_path) as f:
            ref_rows = [json.loads(l) for l in f if l.strip()]

    for r in results:
        label = r["label"]
        mp = os.path.join(OUT_DIR, label, "metrics.jsonl")
        with open(mp) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        steps = [x["step"] for x in rows]
        zgaps = [x["z_shuffle_gap"] for x in rows]
        losses = [x["train_loss"] for x in rows]
        c = colors.get(label, "gray")

        # Clip for display (diverged runs have huge loss values)
        losses_clipped = [min(l, 20) for l in losses]
        zgaps_clipped = [max(z, 1e-5) for z in zgaps]

        ax1.plot(steps, zgaps_clipped, "-", color=c, lw=1.8, label=label)
        ax2.plot(steps, losses_clipped, "-", color=c, lw=1.8, label=label)

    # 803K reference
    if ref_rows:
        steps_ref = [x["step"] for x in ref_rows]
        zgaps_ref = [max(x["z_shuffle_gap"], 1e-5) for x in ref_rows]
        losses_ref = [x["train_loss"] for x in ref_rows]
        ax1.plot(steps_ref, zgaps_ref, "k--", lw=2, alpha=0.6, label="803K ref")
        ax2.plot(steps_ref, losses_ref, "k--", lw=2, alpha=0.6, label="803K ref")

    # Also add the Adam lr=1e-3 12-head baseline from the lr sweep
    adam_path = "results/lr_sweep_85M/lr_1e-3/runs/D10000_seed0/metrics.jsonl"
    if os.path.exists(adam_path):
        with open(adam_path) as f:
            adam_rows = [json.loads(l) for l in f if l.strip()]
        ax1.plot([x["step"] for x in adam_rows],
                 [max(x["z_shuffle_gap"], 1e-5) for x in adam_rows],
                 "gray", ls="-", lw=1.2, alpha=0.5, label="Adam-12h lr=1e-3")
        ax2.plot([x["step"] for x in adam_rows],
                 [x["train_loss"] for x in adam_rows],
                 "gray", ls="-", lw=1.2, alpha=0.5, label="Adam-12h lr=1e-3")

    ax1.axhline(0.02, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("z_shuffle_gap")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-5, 10)
    ax1.set_title("z_shuffle_gap: does z-routing nucleate?")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("train_loss")
    ax2.set_title("Loss trajectory")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("SGD vs Adam at 85M: optimizer or architecture?", fontsize=13)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "sgd_vs_adam_85M.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Saved: {png}")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    for cfg in RUNS:
        r = run_one(cfg, device)
        results.append(r)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'label':>18}  {'opt':>5}  {'lr':>6}  {'heads':>5}  "
          f"{'steps':>6}  {'loss':>8}  {'z_gap':>8}  {'nucleated':>10}  {'reason':>20}")
    print("-" * 90)
    for r in results:
        loss_str = f"{r['final_loss']:.4f}" if r['final_loss'] and not math.isnan(r['final_loss']) else "NaN"
        zgap_str = f"{r['final_z_gap']:.4f}" if r['final_z_gap'] else "—"
        print(f"{r['label']:>18}  {r['optimizer']:>5}  {r['lr']:>6g}  "
              f"{r['n_heads']:>5}  {r['final_step']:>6}  {loss_str:>8}  "
              f"{zgap_str:>8}  {'YES' if r['nucleated'] else 'no':>10}  "
              f"{r['stopped_reason']:>20}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/summary.json")

    ref_path = "results/synthetic_k2_original/runs/D10000_seed0/metrics.jsonl"
    plot_results(results, ref_path)


if __name__ == "__main__":
    main()
