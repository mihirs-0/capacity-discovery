"""Launch Phase 1 and Phase 1.5 sweeps."""

import argparse
import subprocess
import sys
import os


def phase1_configs():
    """Phase 1: D-sweep at fixed model. K=20, 7 D values, 5 seeds each."""
    K = 20
    D_values = [1000, 3000, 5000, 10000, 20000, 50000, 100000]
    seeds = [0, 1, 2, 3, 4]

    configs = []
    for D in D_values:
        n_b = D // K
        for seed in seeds:
            configs.append({
                "K": K,
                "n_b": n_b,
                "data_seed": D,  # same data for all seeds at given D
                "model_seed": seed,
                "experiment_name": "phase1_d_sweep",
            })
    return configs


def phase1_5_configs():
    """Phase 1.5: K-sweep at fixed D=10000. K in {5,10,20}, 5 seeds each."""
    D = 10000
    K_values = [5, 10, 20]
    seeds = [0, 1, 2, 3, 4]

    configs = []
    for K in K_values:
        n_b = D // K
        for seed in seeds:
            configs.append({
                "K": K,
                "n_b": n_b,
                "data_seed": 10000,
                "model_seed": seed,
                "experiment_name": "phase1_5_k_sweep",
            })
    return configs


def run_config(cfg: dict, dry_run: bool = False):
    """Run a single config via run_single.py."""
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "run_single.py"),
        "--K", str(cfg["K"]),
        "--n_b", str(cfg["n_b"]),
        "--data_seed", str(cfg["data_seed"]),
        "--model_seed", str(cfg["model_seed"]),
        "--experiment_name", cfg["experiment_name"],
    ]
    D = cfg["K"] * cfg["n_b"]
    seed = cfg["model_seed"]
    print(f"  D={D}, K={cfg['K']}, seed={seed}")

    if dry_run:
        print(f"  [dry run] {' '.join(cmd)}")
        return

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED: D={D}, seed={seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True,
                        choices=["1", "1.5", "all"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        configs = phase1_configs()
        print(f"Phase 1: {len(configs)} runs")
        for cfg in configs:
            run_config(cfg, args.dry_run)

    if args.phase in ("1.5", "all"):
        configs = phase1_5_configs()
        print(f"Phase 1.5: {len(configs)} runs")
        for cfg in configs:
            run_config(cfg, args.dry_run)


if __name__ == "__main__":
    main()
