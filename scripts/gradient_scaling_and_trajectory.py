#!/usr/bin/env python3
"""Part 1: ||∇L|| at plateau checkpoints for D ∈ {3K, 5K, 10K, 20K, 50K, 100K}.
Part 2: ||∇L|| trajectory through the D=10K retrained-seed-3 training
        (every checkpoint from step 0 to step 5500, every 100 steps).

Fast — no Lanczos, just one forward-backward per checkpoint on the full
training set. Runs on CPU so it doesn't fight any concurrent MPS job.
"""

import argparse
import json
import math
import os
import re
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
def eval_loss(model, data, device, batch_size=2000):
    total, n = 0.0, 0
    for s in range(0, data.shape[0], batch_size):
        b = data[s:s + batch_size].to(device)
        loss, _ = model(b, b)
        total += loss.item() * b.shape[0]
        n += b.shape[0]
    return total / n


def gradient_norm(model, data, device, batch_size=2000):
    for p in model.parameters():
        p.grad = None
    n_batches = 0
    for s in range(0, data.shape[0], batch_size):
        b = data[s:s + batch_size].to(device)
        loss, _ = model(b, b)
        loss.backward()
        n_batches += 1
    # average across minibatches to match reduction='mean' semantics
    for p in model.parameters():
        p.grad.div_(n_batches)
    g = torch.cat([p.grad.detach().flatten() for p in model.parameters()])
    return float(g.norm().item())


# ---- Part 1: plateau gradient across D ---------------------------------------

PART1_TARGETS = [
    # (label, ckpt, D, K, note)
    (  3000, "results/basin_plateau_retrain/runs/D3000_seed1/checkpoints/step_360.pt",
     "latest plateau (seed=1 retrain)"),
    (  5000, "results/basin_plateau_retrain/runs/D5000_seed0/checkpoints/step_620.pt",
     "latest plateau (seed=0 retrain)"),
    ( 10000, "results/basin_plateau_retrain/runs/D10000_seed3/checkpoints/step_1200.pt",
     "latest plateau (seed=3 retrain)"),
    ( 20000, "results/phase1_d_sweep/runs/D20000_seed0/checkpoints/step_2500.pt",
     "mid-plateau (phase1_d_sweep seed=0)"),
    ( 50000, "results/phase1_d_sweep/runs/D50000_seed1/checkpoints/step_32500.pt",
     "late plateau (phase1_d_sweep seed=1)"),
    (100000, "results/phase1_d_sweep/runs/D100000_seed0/checkpoints/step_50000.pt",
     "stuck plateau (phase1_d_sweep seed=0)"),
]


def run_part1(device, batch_size, K):
    print("\n" + "=" * 90)
    print("PART 1: ||∇L|| at plateau checkpoints across D")
    print("=" * 90)
    print(f"{'D':>7}  {'loss':>8}  {'||∇L||':>12}  {'||θ||':>10}  note")
    print("-" * 90)
    results = []
    for D, ckpt, note in PART1_TARGETS:
        t0 = time.time()
        model = load_model(ckpt, device)
        dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
        data = dataset.data
        loss = eval_loss(model, data, device, batch_size=batch_size)
        gn = gradient_norm(model, data, device, batch_size=batch_size)
        param_norm = float(
            torch.cat([p.detach().flatten() for p in model.parameters()]).norm().item()
        )
        dt = time.time() - t0
        print(f"{D:>7}  {loss:>8.4f}  {gn:>12.4e}  {param_norm:>10.4e}  {note} "
              f"({dt:.0f}s)", flush=True)
        results.append({
            "D": D, "ckpt": ckpt, "note": note,
            "loss": loss, "grad_norm": gn, "param_norm": param_norm,
        })
        del model

    # Fit a simple power law to the plateau ||∇L|| ~ D^α
    Ds = np.array([r["D"] for r in results], dtype=float)
    gns = np.array([r["grad_norm"] for r in results], dtype=float)
    logD = np.log(Ds)
    logG = np.log(gns)
    slope, intercept = np.polyfit(logD, logG, 1)
    print()
    print(f"log-log fit: ||∇L|| ≈ exp({intercept:.2f}) · D^{slope:.3f}")
    print(f"             = {np.exp(intercept):.4g} · D^{slope:.3f}")
    return results, {"slope": float(slope), "intercept": float(intercept)}


# ---- Part 2: D=10K trajectory -----------------------------------------------

def part2_checkpoints():
    d = "results/basin_plateau_retrain/runs/D10000_seed3/checkpoints"
    out = []
    for f in sorted(os.listdir(d),
                    key=lambda f: int(re.search(r"step_(\d+)", f).group(1))):
        m = re.search(r"step_(\d+)\.pt$", f)
        if m:
            out.append((int(m.group(1)), os.path.join(d, f)))
    return out


def load_trajectory_losses():
    """Map step -> train_loss from the retrain metrics.jsonl."""
    p = "results/basin_plateau_retrain/runs/D10000_seed3/metrics.jsonl"
    out = {}
    with open(p) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["step"]] = r
    return out


def run_part2(device, batch_size, K):
    print("\n" + "=" * 90)
    print("PART 2: ||∇L|| trajectory through D=10K retrained-seed-3 training")
    print("=" * 90)
    D = 10000
    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
    data = dataset.data

    metrics_by_step = load_trajectory_losses()
    checkpoints = part2_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints in "
          f"results/basin_plateau_retrain/runs/D10000_seed3/checkpoints")

    print(f"\n{'step':>6}  {'loss':>8}  {'||∇L||':>12}")
    print("-" * 36)
    traj = []
    for step, ckpt in checkpoints:
        t0 = time.time()
        model = load_model(ckpt, device)
        gn = gradient_norm(model, data, device, batch_size=batch_size)
        loss = metrics_by_step.get(step, {}).get("train_loss")
        if loss is None:
            loss = eval_loss(model, data, device, batch_size=batch_size)
        dt = time.time() - t0
        print(f"{step:>6}  {loss:>8.4f}  {gn:>12.4e}  ({dt:.0f}s)", flush=True)
        traj.append({"step": step, "loss": loss, "grad_norm": gn})
        del model
    return traj


# ---- Plot ---------------------------------------------------------------------

def plot_part1(part1_results, fit, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    Ds = np.array([r["D"] for r in part1_results], dtype=float)
    gns = np.array([r["grad_norm"] for r in part1_results], dtype=float)
    ax.plot(Ds, gns, "o-", color="#1f77b4", ms=8, lw=2,
            label="‖∇L‖ at plateau")
    # fit line
    xs = np.array([Ds.min(), Ds.max()])
    ys = np.exp(fit["intercept"]) * xs ** fit["slope"]
    ax.plot(xs, ys, "--", color="gray",
            label=f"fit: D^{fit['slope']:.2f}")
    # 1/√D reference anchored at D=3K
    anchor = gns[0] * np.sqrt(Ds[0])
    ax.plot(xs, anchor / np.sqrt(xs), ":", color="#d62728",
            label="CLT prediction 1/√D")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D")
    ax.set_ylabel("‖∇L‖ at plateau")
    ax.set_title("Gradient magnitude at the plateau vs dataset size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_part2(traj, out_path, threshold):
    steps = np.array([r["step"] for r in traj])
    losses = np.array([r["loss"] for r in traj])
    gns = np.array([r["grad_norm"] for r in traj])

    fig, ax = plt.subplots(figsize=(10, 6))
    c_grad = "#1f77b4"
    c_loss = "#d62728"

    l1 = ax.plot(steps, gns, "o-", color=c_grad, ms=5, lw=1.8,
                 label="‖∇L‖ (left axis)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("‖∇L‖", color=c_grad)
    ax.tick_params(axis="y", labelcolor=c_grad)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    l2 = ax2.plot(steps, losses, "s-", color=c_loss, ms=4, lw=1.3, alpha=0.85,
                  label="train_loss (right axis)")
    ax2.set_ylabel("train_loss", color=c_loss)
    ax2.tick_params(axis="y", labelcolor=c_loss)
    ax2.axhline(threshold, color=c_loss, ls=":", lw=0.8, alpha=0.7)

    # Escape step
    escape_step = None
    for r in traj:
        if r["loss"] < threshold:
            escape_step = r["step"]
            break
    if escape_step is not None:
        ax.axvline(escape_step, color="black", ls="--", lw=1, alpha=0.6,
                   label=f"escape @ step {escape_step}")

    lines = l1 + l2
    if escape_step is not None:
        lines += [ax.get_lines()[-1]]
    ax.legend(loc="best", fontsize=9)
    ax.set_title("D=10K: ‖∇L‖ trajectory vs train_loss  (seed=3 retrain)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--out-dir", default="results/gradient_trajectory")
    ap.add_argument("--skip-part1", action="store_true")
    ap.add_argument("--skip-part2", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    K = args.K
    threshold = 0.5 * math.log(K)

    out = {}

    if not args.skip_part1:
        results1, fit = run_part1(device, args.batch_size, K)
        out["part1"] = {"results": results1, "fit": fit}
        plot_part1(results1, fit, os.path.join(args.out_dir, "gradient_vs_D.png"))
        print(f"Saved: {args.out_dir}/gradient_vs_D.png")

    if not args.skip_part2:
        traj = run_part2(device, args.batch_size, K)
        out["part2"] = {"trajectory": traj}
        plot_part2(traj, os.path.join(args.out_dir, "gradient_trajectory_D10K.png"),
                    threshold)
        print(f"Saved: {args.out_dir}/gradient_trajectory_D10K.png")

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
