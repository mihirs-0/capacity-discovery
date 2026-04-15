#!/usr/bin/env python3
"""Experiment 3: warm-start fine-tuning from a D=10K converged checkpoint
onto the D=20K dataset.

Reference baselines (from phase1_d_sweep, used in the comparison):
  D=20K from random init: tau ≈ 4300 (mean; seeds 0,2,4)

Each fine-tune run:
  - Loads a converged D=10K backward checkpoint.
  - Builds the D=20K dataset using the phase1_d_sweep seed convention (seed=D).
  - Runs 20000 training steps with the same optimizer as backward training.
  - Logs diagnostics every 100 steps, checkpoints every 500.

Runs multiple model seeds (each controls the optimizer batch RNG; the model
weights themselves start identical per run since they come from the same
checkpoint).

Outputs:
  results/finetune_transfer/runs/warmstart_D10K_to_D20K_seed{seed}/metrics.jsonl
  results/finetune_transfer/runs/warmstart_D10K_to_D20K_seed{seed}/checkpoints/
  results/finetune_transfer/summary.json                  (tau per run)
  results/finetune_transfer/finetune_vs_scratch.png       (curve comparison)
"""

import argparse
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
from src.diagnostics import compute_train_loss


def get_lr(step, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    return base_lr


def strip_compiled_prefix(state_dict):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()}


def first_cross_tau(metrics: list[dict], threshold: float,
                    key: str = "train_loss") -> float | None:
    for m in metrics:
        v = m.get(key)
        if v is not None and v < threshold:
            return float(m["step"])
    return None


def finetune_one_seed(src_ckpt: str, target_D: int, K: int, data_seed: int,
                      model_seed: int, max_steps: int, eval_every: int,
                      checkpoint_every: int, device: torch.device,
                      out_dir: str, warmup_steps: int, lr: float,
                      batch_size: int, no_warmup: bool) -> dict:
    run_dir = os.path.join(out_dir, f"warmstart_D10K_to_D{target_D}_seed{model_seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    print(f"\n=== fine-tuning: {src_ckpt} -> D={target_D}  seed={model_seed} ===")

    # Build target dataset (follows phase1_d_sweep convention: seed=D)
    dataset = SurjectiveMap(K=K, n_b=target_D // K, seed=data_seed)
    data = dataset.get_full()
    print(f"target dataset: D={dataset.D}  K={K}  seed={data_seed}")

    # Load model from checkpoint (weights identical across runs)
    model = Transformer.from_config(ModelConfig()).to(device)
    state = torch.load(src_ckpt, map_location=device, weights_only=True)
    state = strip_compiled_prefix(state)
    model.load_state_dict(state)
    print(f"model params: {model.count_parameters():,}")

    # Fresh optimizer (same hyperparameters as phase1_d_sweep backward)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999),
        weight_decay=0.01, eps=1e-8,
    )

    batch_rng = np.random.RandomState(model_seed + 10000)
    t_start = time.time()

    def log(step):
        model.eval()
        m = {"step": step}
        m.update(compute_train_loss(model, data, device))
        m["wall_time_seconds"] = time.time() - t_start
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        model.train()
        return m

    def save_ckpt(step):
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"step_{step}.pt"))

    # Initial eval + checkpoint (step 0 = state after loading, before any update)
    save_ckpt(0)
    init = log(0)
    print(f"[step 0]  train_loss={init['train_loss']:.4f}  "
          f"pos1={init.get('train_loss_pos1', -1):.4f}")

    for step in range(1, max_steps + 1):
        # If no_warmup, start at full LR (since the model is already trained).
        # Otherwise use the phase1_d_sweep warmup schedule.
        lr_now = lr if no_warmup else get_lr(step, warmup_steps, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        batch = dataset.get_batch(batch_size, batch_rng).to(device)
        loss, _ = model(batch, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            m = log(step)
            if step % (eval_every * 10) == 0:
                print(f"[step {step:>5}] loss={m['train_loss']:.4f}  "
                      f"pos1={m.get('train_loss_pos1', -1):.4f}  "
                      f"wall={m['wall_time_seconds']:.0f}s")

        if step % checkpoint_every == 0:
            save_ckpt(step)

    wall = time.time() - t_start
    print(f"done in {wall:.0f}s ({wall/60:.1f} min)")

    # Load full metric trace for tau computation
    with open(metrics_path) as f:
        metrics = [json.loads(l) for l in f if l.strip()]

    threshold = 0.5 * math.log(K)
    tau_total = first_cross_tau(metrics, threshold, "train_loss")
    tau_pos1 = first_cross_tau(metrics, threshold, "train_loss_pos1")

    return {
        "run_dir": run_dir,
        "src_ckpt": src_ckpt,
        "target_D": target_D,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "initial_loss": init["train_loss"],
        "initial_loss_pos1": init.get("train_loss_pos1"),
        "final_loss": metrics[-1]["train_loss"],
        "final_loss_pos1": metrics[-1].get("train_loss_pos1"),
        "tau_total": tau_total,
        "tau_pos1": tau_pos1,
        "threshold": threshold,
        "wall_time_seconds": wall,
        "metrics_path": metrics_path,
    }


def load_metrics(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def plot_comparison(finetune_runs: list[dict], scratch_paths: list[str],
                    out_path: str, threshold: float, K: int):
    fig, ax = plt.subplots(figsize=(10, 6))
    c_warm = "#2ca02c"
    c_scratch = "#d62728"

    # Warm-start traces
    for r in finetune_runs:
        m = load_metrics(r["metrics_path"])
        steps = [x["step"] for x in m]
        losses = [x["train_loss"] for x in m]
        ax.plot(steps, losses, "-", color=c_warm, alpha=0.8, lw=1.5)
    ax.plot([], [], "-", color=c_warm, lw=2,
            label=f"warm-start from D=10K (n={len(finetune_runs)})")

    # From-scratch traces (phase1_d_sweep D=20K runs)
    for p in scratch_paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        m = load_metrics(p)
        steps = [x["step"] for x in m]
        losses = [x["train_loss"] for x in m]
        ax.plot(steps, losses, "--", color=c_scratch, alpha=0.7, lw=1.2)
    ax.plot([], [], "--", color=c_scratch, lw=2,
            label=f"from-scratch D=20K (phase1_d_sweep)")

    ax.axhline(threshold, color="black", ls=":", lw=1,
               label=f"0.5 log K = {threshold:.3f}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("train loss")
    ax.set_title("Warm-start fine-tune vs from-scratch training at D=20K")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    # Cap x-axis at 20000 to see early dynamics
    max_step = max(max((x["step"] for x in load_metrics(r["metrics_path"])),
                       default=0) for r in finetune_runs)
    ax.set_xlim(0, max_step)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-ckpt", type=str,
                    default="results/phase1_d_sweep/runs/D10000_seed3/checkpoints/step_4500.pt",
                    help="D=10K converged checkpoint to warm-start from")
    ap.add_argument("--target-D", type=int, default=20000)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--data-seed", type=int, default=None,
                    help="Dataset seed for the target (default = target_D)")
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-warmup", action="store_true",
                    help="Skip the LR warmup (since weights are pre-trained)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="results/finetune_transfer/runs")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if not os.path.exists(args.src_ckpt):
        raise SystemExit(f"Source checkpoint not found: {args.src_ckpt}")

    data_seed = args.data_seed if args.data_seed is not None else args.target_D

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    for seed in args.model_seeds:
        r = finetune_one_seed(
            src_ckpt=args.src_ckpt, target_D=args.target_D, K=args.K,
            data_seed=data_seed, model_seed=seed, max_steps=args.max_steps,
            eval_every=args.eval_every, checkpoint_every=args.checkpoint_every,
            device=device, out_dir=out_dir, warmup_steps=args.warmup_steps,
            lr=args.lr, batch_size=args.batch_size, no_warmup=args.no_warmup,
        )
        runs.append(r)

    # ---- summary ---------------------------------------------------------
    threshold = 0.5 * math.log(args.K)

    # Baseline: phase1_d_sweep D=20K from-scratch runs — compute tau from them
    scratch_paths = [
        f"results/phase1_d_sweep/runs/D{args.target_D}_seed{s}/metrics.jsonl"
        for s in range(5)
    ]
    scratch_taus = []
    for p in scratch_paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        m = load_metrics(p)
        t = first_cross_tau(m, threshold, "train_loss")
        if t is not None:
            scratch_taus.append(t)

    scratch_tau_mean = float(np.mean(scratch_taus)) if scratch_taus else None

    warm_taus = [r["tau_total"] for r in runs if r["tau_total"] is not None]
    warm_tau_mean = float(np.mean(warm_taus)) if warm_taus else None

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"threshold = 0.5 log K = {threshold:.4f}")
    print()
    print(f"from-scratch D={args.target_D}:")
    print(f"  tau per seed: {scratch_taus}")
    print(f"  tau mean    : {scratch_tau_mean}")
    print()
    print(f"warm-start from D=10K -> D={args.target_D}:")
    for r in runs:
        print(f"  seed {r['model_seed']}: initial_loss={r['initial_loss']:.4f}  "
              f"final_loss={r['final_loss']:.4f}  "
              f"tau(total)={r['tau_total']}  tau(pos1)={r['tau_pos1']}")
    print(f"  tau mean (total): {warm_tau_mean}")

    if scratch_tau_mean and warm_tau_mean:
        ratio = warm_tau_mean / scratch_tau_mean
        print(f"\nratio warm / scratch = {ratio:.3f}")
        if ratio < 0.5:
            verdict = "WARM-START HELPS — D=10K basin is inside the D=20K basin"
        elif ratio < 1.5:
            verdict = "NO EFFECT — warm-start neither helps nor hurts"
        else:
            verdict = "WARM-START HURTS — D=10K solution is in a bad region"
        print(verdict)
    else:
        verdict = "inconclusive"

    summary = {
        "experiment": "finetune_transfer_experiment_3",
        "src_ckpt": args.src_ckpt,
        "target_D": args.target_D,
        "threshold": threshold,
        "scratch_taus": scratch_taus,
        "scratch_tau_mean": scratch_tau_mean,
        "warm_runs": runs,
        "warm_tau_mean": warm_tau_mean,
        "ratio_warm_over_scratch": (warm_tau_mean / scratch_tau_mean)
            if (warm_tau_mean and scratch_tau_mean) else None,
        "verdict": verdict,
    }
    summary_path = os.path.join(os.path.dirname(out_dir.rstrip("/")), "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    png_path = os.path.join(os.path.dirname(out_dir.rstrip("/")),
                            "finetune_vs_scratch.png")
    plot_comparison(runs, scratch_paths, png_path, threshold, args.K)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
