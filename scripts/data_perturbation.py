#!/usr/bin/env python3
"""Data-perturbation experiments on the D=100K plateau.

Exp 1 (subset_seed): load D=100K stuck ckpt, train on a 500-group random subset
                     for 2000 steps, then switch to full D=100K for 10000 steps.
                     Does it escape on the subset? Does it stay escaped?

Exp 2 (curriculum):  train from scratch with an expanding dataset
                     D=5K → 10K → 20K → 50K → 100K, each stage until
                     loss < 0.05 or a step cap. Total steps to reach
                     loss<0.05 at D=100K vs never-converging from scratch.

Exp 3 (gradual):     load D=100K stuck ckpt, train 20000 steps with the
                     active group set linearly expanding from 500 to 5000.

All three use the same model architecture (ModelConfig defaults),
same optimizer (AdamW, lr=1e-3, batch=128), and the SurjectiveMap
with seed=100000 so that n_b-N is a strict prefix of n_b=5000.

Usage:
  python scripts/data_perturbation.py --experiment subset_seed
  python scripts/data_perturbation.py --experiment curriculum
  python scripts/data_perturbation.py --experiment gradual
  python scripts/data_perturbation.py --experiment all
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
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap
from src.diagnostics import compute_train_loss, compute_z_shuffle_gap


# ----- Constants -------------------------------------------------------------
FULL_D = 100000
K = 20
DATA_SEED = 100000            # matches phase1_d_sweep D=100K convention
LOG_K = math.log(K)
HALF_LOG_K = 0.5 * LOG_K
D100K_CKPT = "results/phase1_d_sweep/runs/D100000_seed0/checkpoints/step_50000.pt"


# ----- Utilities -------------------------------------------------------------

def strip_prefix(state):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state.items()}


def load_model_from_ckpt(ckpt: str, device: torch.device) -> Transformer:
    m = Transformer.from_config(ModelConfig()).to(device)
    s = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(strip_prefix(s))
    return m


def fresh_model(model_seed: int, device: torch.device) -> Transformer:
    torch.manual_seed(model_seed)
    return Transformer.from_config(ModelConfig()).to(device)


def make_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        weight_decay=0.01, eps=1e-8,
    )


def get_lr(step, warmup, base_lr):
    return base_lr * step / max(1, warmup) if step < warmup else base_lr


def sample_batch_from_groups(data: torch.Tensor, n_b_active: int,
                              batch_size: int, rng: np.random.RandomState) -> torch.Tensor:
    """Sample a batch from the first n_b_active groups of the dataset.
    Each group contributes K contiguous examples.
    """
    max_idx = n_b_active * K
    indices = rng.randint(0, max_idx, size=batch_size)
    return data[indices]


def sample_batch_from_group_set(data: torch.Tensor, group_set: np.ndarray,
                                 batch_size: int, rng: np.random.RandomState
                                 ) -> torch.Tensor:
    """Sample a batch from an arbitrary set of groups (non-contiguous)."""
    # Each row of data at index [g*K .. g*K+K-1] belongs to group g.
    # Pick a batch of (group, in-group-index) pairs.
    group_choices = group_set[rng.randint(0, len(group_set), size=batch_size)]
    within = rng.randint(0, K, size=batch_size)
    flat_idx = group_choices * K + within
    return data[flat_idx]


@torch.no_grad()
def eval_loss_on_data(model, data, device, batch_size=2048) -> float:
    model.eval()
    total, n = 0.0, 0
    for s in range(0, data.shape[0], batch_size):
        b = data[s:s + batch_size].to(device)
        loss, _ = model(b, b)
        total += loss.item() * b.shape[0]
        n += b.shape[0]
    model.train()
    return total / n


@torch.no_grad()
def compute_metrics(model, full_data, device,
                    active_data: torch.Tensor | None = None,
                    compute_z_gap: bool = True) -> dict:
    """Loss on full D=100K, loss on active subset (if any), and optional
    z_shuffle_gap on the full dataset."""
    m = {}
    m["full_loss"] = eval_loss_on_data(model, full_data, device)
    if active_data is not None and active_data.shape[0] > 0:
        m["active_loss"] = eval_loss_on_data(model, active_data, device)
    else:
        m["active_loss"] = m["full_loss"]
    if compute_z_gap:
        zs = compute_z_shuffle_gap(model, full_data, device, batch_size=1024)
        m["z_shuffle_gap"] = zs["z_shuffle_gap"]
    return m


def log_row(metrics_path: str, row: dict):
    with open(metrics_path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ==============================================================================
# Experiment 1: Subset seeding
# ==============================================================================

def run_exp1(seed: int, out_dir: str, device: torch.device,
             phase_a_steps: int = 2000, phase_b_steps: int = 10000,
             eval_every: int = 200, n_subset_groups: int = 500) -> dict:
    print(f"\n==== EXP1 subset_seed  seed={seed} ====")
    run_dir = os.path.join(out_dir, "subset_seeding", f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    dataset = SurjectiveMap(K=K, n_b=FULL_D // K, seed=DATA_SEED)
    full_data = dataset.data
    # Pick a random subset of 500 groups (controlled by seed)
    rng_subset = np.random.RandomState(seed + 7000)
    chosen_groups = np.sort(
        rng_subset.choice(dataset.n_b, size=n_subset_groups, replace=False)
    )
    # Build a subset data tensor for convenient loss eval
    subset_idx = np.concatenate([
        np.arange(g * K, (g + 1) * K) for g in chosen_groups
    ])
    subset_data = full_data[subset_idx]
    print(f"  chose {len(chosen_groups)} groups out of {dataset.n_b}  "
          f"(subset D={subset_data.shape[0]})")

    model = load_model_from_ckpt(D100K_CKPT, device)
    optimizer = make_optimizer(model)
    batch_rng = np.random.RandomState(seed + 10000)
    t0 = time.time()

    def log(step, phase):
        m = compute_metrics(model, full_data, device, active_data=subset_data)
        m.update({"step": step, "phase": phase,
                  "wall_time_seconds": time.time() - t0})
        log_row(metrics_path, m)
        return m

    # Step 0 baseline
    m0 = log(0, "A")
    print(f"  [phase A] step 0:    full_loss={m0['full_loss']:.4f}  "
          f"subset_loss={m0['active_loss']:.4f}  z_gap={m0['z_shuffle_gap']:.4f}")

    # --- Phase A: train on subset only -----------------------------------
    model.train()
    for step in range(1, phase_a_steps + 1):
        lr = get_lr(step, 500, 1e-3)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        batch = sample_batch_from_group_set(
            full_data, chosen_groups, 128, batch_rng
        ).to(device)
        loss, _ = model(batch, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_every == 0:
            m = log(step, "A")
            if step % (eval_every * 5) == 0:
                print(f"  [phase A] step {step:>5}: "
                      f"full_loss={m['full_loss']:.4f}  "
                      f"subset_loss={m['active_loss']:.4f}  "
                      f"z_gap={m['z_shuffle_gap']:.4f}")

    # Final Phase A log
    if phase_a_steps % eval_every != 0:
        log(phase_a_steps, "A")

    # --- Phase B: switch to full dataset ----------------------------------
    # Reset batch RNG offset to differentiate
    batch_rng_b = np.random.RandomState(seed + 20000)
    for step in range(phase_a_steps + 1, phase_a_steps + phase_b_steps + 1):
        lr = 1e-3
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        # Sample from ALL 5000 groups
        batch = sample_batch_from_groups(
            full_data, dataset.n_b, 128, batch_rng_b
        ).to(device)
        loss, _ = model(batch, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_every == 0:
            m = log(step, "B")
            if step % (eval_every * 5) == 0:
                print(f"  [phase B] step {step:>5}: "
                      f"full_loss={m['full_loss']:.4f}  "
                      f"subset_loss={m['active_loss']:.4f}  "
                      f"z_gap={m['z_shuffle_gap']:.4f}")

    wall = time.time() - t0
    print(f"  done in {wall:.0f}s ({wall/60:.1f} min)")

    # Summary
    with open(metrics_path) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]
    a_rows = [r for r in all_rows if r["phase"] == "A"]
    b_rows = [r for r in all_rows if r["phase"] == "B"]

    def first_cross(rows, key, thresh):
        for r in rows:
            if r.get(key) is not None and r[key] < thresh:
                return r["step"]
        return None

    summary = {
        "seed": seed,
        "run_dir": run_dir,
        "phase_a_steps": phase_a_steps,
        "phase_b_steps": phase_b_steps,
        "n_subset_groups": n_subset_groups,
        "phase_a_last_full_loss": a_rows[-1]["full_loss"] if a_rows else None,
        "phase_a_last_subset_loss": a_rows[-1]["active_loss"] if a_rows else None,
        "phase_b_last_full_loss": b_rows[-1]["full_loss"] if b_rows else None,
        "phase_b_last_subset_loss": b_rows[-1]["active_loss"] if b_rows else None,
        # Escape-time metrics
        "a_tau_subset":
            first_cross(a_rows, "active_loss", HALF_LOG_K),
        "a_tau_full":
            first_cross(a_rows, "full_loss", HALF_LOG_K),
        "b_tau_full":
            first_cross(b_rows, "full_loss", HALF_LOG_K),
        "wall_time_seconds": wall,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ==============================================================================
# Experiment 2: Curriculum training
# ==============================================================================

CURRICULUM_STAGES = [
    # (n_b, max_steps)
    ( 250, 10000),   # stage 1: D=5K
    ( 500, 10000),   # stage 2: D=10K
    (1000, 20000),   # stage 3: D=20K
    (2500, 30000),   # stage 4: D=50K
    (5000, 50000),   # stage 5: D=100K
]
CONV_THRESHOLD = 0.05
CONV_PATIENCE = 3       # consecutive evals below threshold


def run_exp2(seed: int, out_dir: str, device: torch.device,
             eval_every: int = 200) -> dict:
    print(f"\n==== EXP2 curriculum  seed={seed} ====")
    run_dir = os.path.join(out_dir, "curriculum", f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    # One dataset at full n_b — each stage uses a prefix via sample_batch_from_groups
    dataset = SurjectiveMap(K=K, n_b=FULL_D // K, seed=DATA_SEED)
    full_data = dataset.data

    model = fresh_model(model_seed=seed, device=device)
    optimizer = make_optimizer(model)
    batch_rng = np.random.RandomState(seed + 10000)

    t0 = time.time()
    global_step = 0
    stage_metrics = []

    for stage_idx, (n_b_stage, max_steps) in enumerate(CURRICULUM_STAGES):
        stage_D = n_b_stage * K
        active_data = full_data[:stage_D]
        print(f"  -- stage {stage_idx+1}: n_b={n_b_stage}  D={stage_D}  "
              f"max_steps={max_steps} --")

        # Log stage start
        m = compute_metrics(model, full_data, device, active_data=active_data)
        m.update({"step": global_step, "stage": stage_idx + 1,
                  "stage_n_b": n_b_stage, "stage_D": stage_D,
                  "stage_step": 0,
                  "wall_time_seconds": time.time() - t0})
        log_row(metrics_path, m)
        print(f"     start: full_loss={m['full_loss']:.4f}  "
              f"stage_loss={m['active_loss']:.4f}  z_gap={m['z_shuffle_gap']:.4f}")

        stage_start_global = global_step
        below_streak = 0
        stage_tau = None
        stage_converged = False
        model.train()
        for stage_step in range(1, max_steps + 1):
            global_step += 1
            # Keep a single global warmup schedule across all stages
            lr = get_lr(global_step, 500, 1e-3)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch = sample_batch_from_groups(
                full_data, n_b_stage, 128, batch_rng
            ).to(device)
            loss, _ = model(batch, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if stage_step % eval_every == 0 or stage_step == max_steps:
                m = compute_metrics(model, full_data, device, active_data=active_data)
                m.update({"step": global_step, "stage": stage_idx + 1,
                          "stage_n_b": n_b_stage, "stage_D": stage_D,
                          "stage_step": stage_step,
                          "wall_time_seconds": time.time() - t0})
                log_row(metrics_path, m)

                if stage_tau is None and m["active_loss"] < HALF_LOG_K:
                    stage_tau = stage_step

                if m["active_loss"] < CONV_THRESHOLD:
                    below_streak += 1
                else:
                    below_streak = 0

                if below_streak >= CONV_PATIENCE:
                    stage_converged = True
                    print(f"     [early stop] stage_step {stage_step}  "
                          f"stage_loss={m['active_loss']:.4f}")
                    break

                if stage_step % (eval_every * 5) == 0:
                    print(f"     stage_step {stage_step:>5}: "
                          f"full_loss={m['full_loss']:.4f}  "
                          f"stage_loss={m['active_loss']:.4f}")

        stage_wall = time.time() - t0
        stage_metrics.append({
            "stage": stage_idx + 1,
            "n_b": n_b_stage,
            "stage_D": stage_D,
            "stage_steps": global_step - stage_start_global,
            "global_step_end": global_step,
            "final_full_loss": m["full_loss"],
            "final_stage_loss": m["active_loss"],
            "stage_tau": stage_tau,
            "stage_converged": stage_converged,
            "wall_time_seconds": stage_wall,
        })
        print(f"     stage end: full_loss={m['full_loss']:.4f}  "
              f"stage_loss={m['active_loss']:.4f}  "
              f"steps={global_step - stage_start_global}  converged={stage_converged}")

        if not stage_converged:
            print(f"     [note] stage {stage_idx+1} didn't hit loss<{CONV_THRESHOLD} "
                  f"in {max_steps} steps — continuing to next stage anyway")

    wall = time.time() - t0
    print(f"  done in {wall:.0f}s ({wall/60:.1f} min)")

    summary = {
        "seed": seed,
        "run_dir": run_dir,
        "stages": stage_metrics,
        "total_steps": global_step,
        "final_full_loss": stage_metrics[-1]["final_full_loss"] if stage_metrics else None,
        "all_stages_converged": all(s["stage_converged"] for s in stage_metrics),
        "wall_time_seconds": wall,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ==============================================================================
# Experiment 3: Gradual group expansion
# ==============================================================================

def run_exp3(seed: int, out_dir: str, device: torch.device,
             total_steps: int = 20000, eval_every: int = 200,
             start_groups: int = 500, end_groups: int = 5000) -> dict:
    print(f"\n==== EXP3 gradual  seed={seed} ====")
    run_dir = os.path.join(out_dir, "gradual_expansion", f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    dataset = SurjectiveMap(K=K, n_b=FULL_D // K, seed=DATA_SEED)
    full_data = dataset.data

    model = load_model_from_ckpt(D100K_CKPT, device)
    optimizer = make_optimizer(model)
    batch_rng = np.random.RandomState(seed + 10000)

    t0 = time.time()
    slope = (end_groups - start_groups) / max(1, total_steps)

    def n_active_at(step: int) -> int:
        n = start_groups + int(round(step * slope))
        return max(start_groups, min(end_groups, n))

    def log(step: int):
        n_active = n_active_at(step)
        active_data = full_data[:n_active * K]
        m = compute_metrics(model, full_data, device, active_data=active_data)
        m.update({"step": step, "n_active": n_active,
                  "wall_time_seconds": time.time() - t0})
        log_row(metrics_path, m)
        return m

    m0 = log(0)
    print(f"  step 0: full_loss={m0['full_loss']:.4f}  "
          f"active_loss={m0['active_loss']:.4f}  n_active={m0['n_active']}")

    model.train()
    for step in range(1, total_steps + 1):
        lr = get_lr(step, 500, 1e-3)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        n_active = n_active_at(step)
        batch = sample_batch_from_groups(
            full_data, n_active, 128, batch_rng
        ).to(device)
        loss, _ = model(batch, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            m = log(step)
            if step % (eval_every * 5) == 0:
                print(f"  step {step:>5}: full_loss={m['full_loss']:.4f}  "
                      f"active_loss={m['active_loss']:.4f}  "
                      f"z_gap={m['z_shuffle_gap']:.4f}  "
                      f"n_active={m['n_active']}")

    wall = time.time() - t0
    print(f"  done in {wall:.0f}s ({wall/60:.1f} min)")

    with open(metrics_path) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]

    def first_cross(rows, key, thresh):
        for r in rows:
            if r.get(key) is not None and r[key] < thresh:
                return r["step"]
        return None

    summary = {
        "seed": seed,
        "run_dir": run_dir,
        "total_steps": total_steps,
        "final_full_loss": all_rows[-1]["full_loss"],
        "final_active_loss": all_rows[-1]["active_loss"],
        "tau_full": first_cross(all_rows, "full_loss", HALF_LOG_K),
        "tau_active": first_cross(all_rows, "active_loss", HALF_LOG_K),
        "wall_time_seconds": wall,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ==============================================================================
# Main / plotting
# ==============================================================================

def plot_all(out_dir: str, exp1_summaries, exp2_summaries, exp3_summaries):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhline(HALF_LOG_K, color="black", ls="--", lw=0.8,
               label=f"0.5 log K = {HALF_LOG_K:.3f}")
    ax.axhline(2.866, color="gray", ls=":", lw=0.8,
               label="D=100K stuck baseline (2.866)")

    def _load_trace(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    for s in exp1_summaries:
        rows = _load_trace(os.path.join(s["run_dir"], "metrics.jsonl"))
        steps = [r["step"] for r in rows]
        losses = [r["full_loss"] for r in rows]
        ax.plot(steps, losses, color="#1f77b4", alpha=0.85, lw=1.3,
                label="Exp1 subset_seed (full loss)"
                if s is exp1_summaries[0] else None)
        # Mark the phase boundary
        ax.axvline(s["phase_a_steps"], color="#1f77b4", ls=":",
                   lw=0.6, alpha=0.4)

    for s in exp2_summaries:
        rows = _load_trace(os.path.join(s["run_dir"], "metrics.jsonl"))
        steps = [r["step"] for r in rows]
        losses = [r["full_loss"] for r in rows]
        stage_losses = [r["active_loss"] for r in rows]
        ax.plot(steps, losses, color="#2ca02c", alpha=0.7, lw=1.3,
                label="Exp2 curriculum (full loss)"
                if s is exp2_summaries[0] else None)
        ax.plot(steps, stage_losses, color="#2ca02c", alpha=0.35, lw=1.0,
                ls="--", label="Exp2 curriculum (stage loss)"
                if s is exp2_summaries[0] else None)
        # Stage boundaries
        prev_cum = 0
        for st in s["stages"]:
            prev_cum = st["global_step_end"]
            ax.axvline(prev_cum, color="#2ca02c", ls=":", lw=0.5, alpha=0.3)

    for s in exp3_summaries:
        rows = _load_trace(os.path.join(s["run_dir"], "metrics.jsonl"))
        steps = [r["step"] for r in rows]
        losses = [r["full_loss"] for r in rows]
        ax.plot(steps, losses, color="#d62728", alpha=0.85, lw=1.3,
                label="Exp3 gradual (full loss)"
                if s is exp3_summaries[0] else None)

    ax.set_xlabel("Training step (within experiment)")
    ax.set_ylabel("train_loss")
    ax.set_yscale("log")
    ax.set_title("Data-perturbation experiments: does D=100K escape with bootstrapping?")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "data_perturbation_all.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["subset_seed", "curriculum", "gradual", "all"],
                    default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--out-dir", default="results/data_perturbation")
    ap.add_argument("--device", default=None)
    ap.add_argument("--phase-a-steps", type=int, default=2000)
    ap.add_argument("--phase-b-steps", type=int, default=10000)
    ap.add_argument("--exp3-steps", type=int, default=20000)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    want1 = args.experiment in ("subset_seed", "all")
    want2 = args.experiment in ("curriculum", "all")
    want3 = args.experiment in ("gradual", "all")

    exp1_results = []
    exp2_results = []
    exp3_results = []

    if want1:
        for s in args.seeds:
            r = run_exp1(s, args.out_dir, device,
                         phase_a_steps=args.phase_a_steps,
                         phase_b_steps=args.phase_b_steps,
                         eval_every=args.eval_every)
            exp1_results.append(r)

    if want2:
        for s in args.seeds:
            r = run_exp2(s, args.out_dir, device, eval_every=args.eval_every)
            exp2_results.append(r)

    if want3:
        for s in args.seeds:
            r = run_exp3(s, args.out_dir, device,
                         total_steps=args.exp3_steps,
                         eval_every=args.eval_every)
            exp3_results.append(r)

    # Summary
    print("\n" + "=" * 98)
    print("SUMMARY TABLE")
    print("=" * 98)
    print(f"{'experiment':>20}  {'seed':>5}  {'converged?':>12}  "
          f"{'total_steps':>12}  {'final_loss':>12}  {'notes':>28}")
    print("-" * 98)
    for s in exp1_results:
        conv_a = "YES" if (s["phase_a_last_subset_loss"] is not None
                           and s["phase_a_last_subset_loss"] < CONV_THRESHOLD) else "no"
        conv_b = "YES" if (s["phase_b_last_full_loss"] is not None
                           and s["phase_b_last_full_loss"] < CONV_THRESHOLD) else "no"
        print(f"{'exp1_subset_phaseA':>20}  {s['seed']:>5}  {conv_a:>12}  "
              f"{s['phase_a_steps']:>12}  {s['phase_a_last_subset_loss']:>12.4f}  "
              f"{'subset loss':>28}")
        print(f"{'exp1_subset_phaseB':>20}  {s['seed']:>5}  {conv_b:>12}  "
              f"{s['phase_b_steps']:>12}  {s['phase_b_last_full_loss']:>12.4f}  "
              f"{'full loss post-switch':>28}")
    for s in exp2_results:
        conv = "YES" if s["all_stages_converged"] else "no"
        print(f"{'exp2_curriculum':>20}  {s['seed']:>5}  {conv:>12}  "
              f"{s['total_steps']:>12}  "
              f"{s['final_full_loss']:>12.4f}  "
              f"{len(s['stages']):>3}/{len(CURRICULUM_STAGES)} stages done")
    for s in exp3_results:
        conv = "YES" if s["final_full_loss"] < CONV_THRESHOLD else "no"
        print(f"{'exp3_gradual':>20}  {s['seed']:>5}  {conv:>12}  "
              f"{s['total_steps']:>12}  "
              f"{s['final_full_loss']:>12.4f}  "
              f"{'final active tau='+str(s['tau_active']):>28}")
    print()

    # Save combined JSON
    combined = {
        "exp1": exp1_results,
        "exp2": exp2_results,
        "exp3": exp3_results,
    }
    with open(os.path.join(args.out_dir, "combined_summary.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved: {args.out_dir}/combined_summary.json")

    if exp1_results or exp2_results or exp3_results:
        plot_all(args.out_dir, exp1_results, exp2_results, exp3_results)


if __name__ == "__main__":
    main()
