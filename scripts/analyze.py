"""Load results, compute tau, and produce summary statistics."""

import json
import os
import math
import numpy as np
import argparse


def load_metrics(run_dir: str) -> list[dict]:
    """Load metrics.jsonl from a run directory."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_tau(metrics: list[dict], K: int, threshold_frac: float = 0.5) -> float | None:
    """Compute waiting time tau: first step where loss < threshold_frac * log(K).

    Returns None if the threshold is never crossed.
    """
    threshold = threshold_frac * math.log(K)
    for m in metrics:
        if m["train_loss"] < threshold:
            return m["step"]
    return None


def compute_z_onset(metrics: list[dict], gap_threshold: float = 0.1,
                     n_consec: int = 3) -> float | None:
    """First step where z_shuffle_gap > gap_threshold for n_consec evaluations."""
    count = 0
    for m in metrics:
        if m.get("z_shuffle_gap", 0) > gap_threshold:
            count += 1
            if count >= n_consec:
                # Return the step where the streak started
                idx = metrics.index(m)
                return metrics[idx - n_consec + 1]["step"]
        else:
            count = 0
    return None


def find_runs(experiment_dir: str) -> list[dict]:
    """Find all run directories and parse D/seed from names."""
    runs_dir = os.path.join(experiment_dir, "runs")
    if not os.path.exists(runs_dir):
        return []

    results = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.startswith("D"):
            continue
        parts = name.split("_")
        D = int(parts[0][1:])
        seed = int(parts[1][4:])
        run_dir = os.path.join(runs_dir, name)
        metrics = load_metrics(run_dir)
        if not metrics:
            continue
        results.append({
            "D": D,
            "seed": seed,
            "run_dir": run_dir,
            "metrics": metrics,
        })
    return results


def summarize_phase1(experiment_dir: str, K: int = 20):
    """Summarize Phase 1 results."""
    runs = find_runs(experiment_dir)
    if not runs:
        print("No runs found.")
        return

    # Group by D
    from collections import defaultdict
    by_D = defaultdict(list)
    for r in runs:
        by_D[r["D"]].append(r)

    print(f"{'D':>8} {'n_runs':>6} {'tau_mean':>10} {'tau_std':>10} "
          f"{'final_loss':>12} {'z_onset':>10}")
    print("-" * 70)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    for D in sorted(by_D.keys()):
        group = by_D[D]
        taus = []
        finals = []
        z_onsets = []

        for r in group:
            tau = compute_tau(r["metrics"], K, 0.5)
            taus.append(tau)
            final = r["metrics"][-1]["train_loss"] if r["metrics"] else None
            finals.append(final)
            z_on = compute_z_onset(r["metrics"])
            z_onsets.append(z_on)

        valid_taus = [t for t in taus if t is not None]
        valid_finals = [f for f in finals if f is not None]
        valid_z = [z for z in z_onsets if z is not None]

        tau_mean = np.mean(valid_taus) if valid_taus else float("nan")
        tau_std = np.std(valid_taus) if len(valid_taus) > 1 else float("nan")
        final_mean = np.mean(valid_finals) if valid_finals else float("nan")
        z_mean = np.mean(valid_z) if valid_z else float("nan")

        print(f"{D:>8} {len(group):>6} {tau_mean:>10.1f} {tau_std:>10.1f} "
              f"{final_mean:>12.4f} {z_mean:>10.1f}")

        # Also compute tau at multiple thresholds
        for thresh in thresholds:
            thresh_taus = [compute_tau(r["metrics"], K, thresh) for r in group]
            valid = [t for t in thresh_taus if t is not None]
            mean = np.mean(valid) if valid else float("nan")
            print(f"         tau@{thresh}: {mean:.1f} ({len(valid)}/{len(group)} converged)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str,
                        help="Path to experiment directory (e.g., results/phase1_d_sweep)")
    parser.add_argument("--K", type=int, default=20)
    args = parser.parse_args()

    summarize_phase1(args.experiment_dir, args.K)


if __name__ == "__main__":
    main()
