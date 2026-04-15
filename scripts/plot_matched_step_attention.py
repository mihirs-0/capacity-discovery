"""Matched-step z-attention comparison across D values (Phase 1 D-sweep).

Averages across whichever seeds have attention data (per-D available seeds),
since no single seed covers all Ds.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/Users/mihir/capacity/staged-discovery/results/phase1_d_sweep/runs")
OUT = Path("/Users/mihir/capacity/staged-discovery/results/phase1_d_sweep/plots/matched_step_attention.png")

D_VALUES = [1000, 3000, 5000, 10000, 20000, 50000, 100000]
TABLE_STEPS = [500, 1000, 2000, 3000, 4000]
X_CAP = 10000
ESCAPE_STEP = 4300
LAYER0_HEADS = ["z_attn_L0H0", "z_attn_L0H1", "z_attn_L0H2", "z_attn_L0H3"]


def load_seed(path: Path):
    steps, vals = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            steps.append(row["step"])
            vals.append(max(row[h] for h in LAYER0_HEADS))
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(vals)[order]


def collect(d: int):
    """Return list of (steps, vals) per available seed and the seeds used."""
    seeds_used = []
    trajs = []
    for seed in range(5):
        p = BASE / f"D{d}_seed{seed}" / "attention_from_ckpt.jsonl"
        if p.exists() and p.stat().st_size > 0:
            s, v = load_seed(p)
            trajs.append((s, v))
            seeds_used.append(seed)
    return trajs, seeds_used


def mean_curve(trajs):
    """Average z-attn across seeds on the union of their step grids.
    For each sample step in any seed, linearly interpolate the other seeds
    (only within their own [0, last_step] range) and take the mean."""
    all_steps = sorted({int(s) for traj, _ in [(t, None) for t in trajs] for s in traj[0]})
    ys = []
    for step in all_steps:
        samples = []
        for s, v in trajs:
            if step <= s[-1]:
                samples.append(np.interp(step, s, v))
        ys.append(np.mean(samples) if samples else np.nan)
    return np.array(all_steps), np.array(ys)


def interp_at(trajs, step):
    samples = []
    for s, v in trajs:
        if step <= s[-1]:
            samples.append(float(np.interp(step, s, v)))
    if not samples:
        return float("nan")
    return float(np.mean(samples))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("viridis")
    colors = {d: cmap(i / (len(D_VALUES) - 1)) for i, d in enumerate(D_VALUES)}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    curves = {}
    seed_info = {}
    for d in D_VALUES:
        trajs, seeds = collect(d)
        if not trajs:
            print(f"[warn] no attention data for D={d}")
            continue
        steps, vals = mean_curve(trajs)
        curves[d] = (trajs, steps, vals)
        seed_info[d] = seeds
        mask = steps <= X_CAP
        label = f"D={d:>6d}  (seeds {','.join(map(str, seeds))}, n={len(seeds)})"
        ax.plot(steps[mask], vals[mask], marker="o", color=colors[d], label=label, lw=1.8, ms=5)

    ax.axvline(ESCAPE_STEP, color="red", linestyle="--", lw=1.2, alpha=0.7,
               label=f"D=20K escape (step {ESCAPE_STEP})")
    ax.set_xlim(0, X_CAP)
    ax.set_xlabel("Training step")
    ax.set_ylabel("z-attention (max over layer-0 heads)")
    ax.set_title("Matched-step z-attention comparison across D  (seed-averaged)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"\nSaved: {OUT}")

    # Table
    print("\nz-attention (max over layer-0 heads), seed-averaged, linearly interpolated:")
    print("NaN = requested step is beyond last available checkpoint for every available seed.\n")
    header = "  D     | seeds used      | " + " | ".join(f"step {s:>5d}" for s in TABLE_STEPS)
    print(header)
    print("-" * len(header))
    for d in D_VALUES:
        if d not in curves:
            continue
        trajs, _, _ = curves[d]
        seeds = seed_info[d]
        last = max(int(traj[0][-1]) for traj in trajs)
        row_cells = []
        for s in TABLE_STEPS:
            v = interp_at(trajs, s)
            row_cells.append(f"{'  nan   ' if np.isnan(v) else f'  {v:.4f}'}")
        seed_str = ",".join(map(str, seeds))
        print(f" {d:>6d} | {seed_str:<15s} | " + " | ".join(row_cells)
              + f"    (last step {last})")


if __name__ == "__main__":
    main()
