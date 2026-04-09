#!/usr/bin/env python3
"""Parallel experiment launcher for single-GPU deployment.

Runs multiple independent training experiments concurrently using a process pool.
Each worker gets its own model/optimizer/dataset and shares the GPU via CUDA's
built-in kernel scheduling.

Memory budget per worker on GPU:
  - Model params:     ~3.2 MB  (803K x 4 bytes)
  - Adam state (m,v): ~6.4 MB
  - Gradients:        ~3.2 MB
  - Activations:      ~2 MB per batch
  - Total:            ~15 MB per worker
  - CUDA context:     ~500 MB per process (spawned)

  RTX 5090 (32 GB VRAM, 32 vCPU, 30 GB disk):
    24 workers: ~12.4 GB VRAM, checkpoints ~3.4 GB disk

  H100 SXM (80 GB VRAM, 8 vCPU):
    8 workers: ~4.1 GB VRAM

Usage:
  # RTX 5090: all phases, 24 workers
  python scripts/run_parallel.py --phase all --workers 24

  # Quick smoke test
  python scripts/run_parallel.py --phase 1 --workers 4 --max-steps 2000

  # Dry run to preview
  python scripts/run_parallel.py --phase all --dry-run
"""

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import get_context

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Config generation (must be importable at module level for spawn pickling)
# ---------------------------------------------------------------------------

def _make_config_dict(K, n_b, data_seed, model_seed, experiment_name,
                      max_steps=50_000, checkpoint_every=2500,
                      eval_every=100, results_root=None):
    """Build a config dict that ExperimentConfig.from_dict can consume."""
    d = {
        "task": {
            "K": K,
            "n_b": n_b,
            "len_b": 6,
            "len_a": 4,
            "len_z": 2,
            "vocab_size": 36,
            "data_seed": data_seed,
        },
        "model": {
            "n_layers": 4,
            "n_heads": 4,
            "d_model": 128,
            "d_mlp": 512,
            "vocab_size": 40,
            "max_seq_len": 16,
        },
        "training": {
            "optimizer": "adamw",
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "eps": 1e-8,
            "batch_size": 128,
            "warmup_steps": 500,
            "max_steps": max_steps,
            "model_seed": model_seed,
        },
        "eval": {
            "eval_every": eval_every,
            "checkpoint_every": checkpoint_every,
            "z_shuffle_batch_size": 1024,
            "group_eval_n_groups": 200,
            "hessian_enabled": False,
            "d7_enabled": False,
        },
        "experiment_name": experiment_name,
    }
    if results_root:
        d["_results_root"] = results_root
    return d


def generate_phase1_configs(max_steps=50_000, checkpoint_every=2500,
                            eval_every=100, results_root=None):
    """Phase 1: D-sweep. K=20, 7 D values x 5 seeds = 35 runs."""
    K = 20
    D_values = [1000, 3000, 5000, 10000, 20000, 50000, 100000]
    seeds = [0, 1, 2, 3, 4]
    configs = []
    for D in D_values:
        n_b = D // K
        for seed in seeds:
            configs.append(_make_config_dict(
                K=K, n_b=n_b, data_seed=D, model_seed=seed,
                experiment_name="phase1_d_sweep",
                max_steps=max_steps, checkpoint_every=checkpoint_every,
                eval_every=eval_every, results_root=results_root,
            ))
    return configs


def generate_phase1_5_configs(max_steps=50_000, checkpoint_every=2500,
                              eval_every=100, results_root=None):
    """Phase 1.5: K-sweep at fixed D=10000. 3 K values x 5 seeds = 15 runs."""
    D = 10000
    K_values = [5, 10, 20]
    seeds = [0, 1, 2, 3, 4]
    configs = []
    for K in K_values:
        n_b = D // K
        for seed in seeds:
            configs.append(_make_config_dict(
                K=K, n_b=n_b, data_seed=10000, model_seed=seed,
                experiment_name="phase1_5_k_sweep",
                max_steps=max_steps, checkpoint_every=checkpoint_every,
                eval_every=eval_every, results_root=results_root,
            ))
    return configs


# ---------------------------------------------------------------------------
# Worker function (top-level for pickle compatibility with spawn)
# ---------------------------------------------------------------------------

def _run_single_experiment(args):
    """Execute one training run inside a spawned worker process.

    Returns (run_label, success: bool, message: str, elapsed_seconds: float).
    """
    config_dict, run_index, total_runs = args
    t0 = time.time()

    K = config_dict["task"]["K"]
    D = K * config_dict["task"]["n_b"]
    seed = config_dict["training"]["model_seed"]
    run_label = f"D={D}_K={K}_seed={seed}"

    try:
        import torch
        from src.config import ExperimentConfig
        from src.trainer import Trainer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = ExperimentConfig.from_dict(config_dict)

        # Allow overriding results root (e.g. for remote storage)
        results_root = config_dict.get("_results_root")
        trainer = Trainer(config, device=device, quiet=True)
        if results_root:
            # Re-point output directory
            run_dir = os.path.join(
                results_root, config.experiment_name, "runs",
                f"D{D}_seed{seed}"
            )
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "hessian"), exist_ok=True)
            trainer.run_dir = run_dir
            trainer.metrics_path = os.path.join(run_dir, "metrics.jsonl")
            open(trainer.metrics_path, "w").close()

        final_metrics = trainer.train()
        elapsed = time.time() - t0
        msg = (f"loss={final_metrics['train_loss']:.4f}  "
               f"gap={final_metrics['z_shuffle_gap']:.4f}  "
               f"{elapsed:.0f}s")
        return (run_label, True, msg, elapsed)

    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        return (run_label, False, f"{e}\n{tb}", elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run parallel training experiments on a single GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", choices=["1", "1.5", "all"], default="all",
        help="Which experimental phase(s) to run (default: all).",
    )
    parser.add_argument(
        "--workers", type=int, default=24,
        help="Number of parallel worker processes (default: 24).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50_000,
        help="Training steps per run (default: 50000).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=2500,
        help="Save model checkpoint every N steps (default: 2500). "
             "200 = full fidelity but ~40 GB disk for 50 runs. "
             "2500 = ~3.4 GB disk, 21 checkpoints per run.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=100,
        help="Run diagnostics every N steps (default: 100).",
    )
    parser.add_argument(
        "--results-root", type=str, default=None,
        help="Override results output directory (default: ./results/).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configs without running.",
    )
    args = parser.parse_args()

    # Collect configs
    sweep_kwargs = dict(
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        results_root=args.results_root,
    )
    configs = []
    if args.phase in ("1", "all"):
        configs.extend(generate_phase1_configs(**sweep_kwargs))
    if args.phase in ("1.5", "all"):
        configs.extend(generate_phase1_5_configs(**sweep_kwargs))

    # Disk budget estimate
    n_ckpts = args.max_steps // args.checkpoint_every + 1
    ckpt_bytes = 803584 * 4  # params * float32
    disk_gb = len(configs) * n_ckpts * ckpt_bytes / 1e9
    metrics_mb = len(configs) * (args.max_steps // args.eval_every + 1) * 500 / 1e6

    print(f"Experiment plan: {len(configs)} runs, {args.workers} workers, "
          f"{args.max_steps} steps/run")
    print(f"  Phase 1:   {sum(1 for c in configs if c['experiment_name'] == 'phase1_d_sweep')} runs")
    print(f"  Phase 1.5: {sum(1 for c in configs if c['experiment_name'] == 'phase1_5_k_sweep')} runs")
    print(f"  Checkpoints: every {args.checkpoint_every} steps "
          f"({n_ckpts}/run, ~{disk_gb:.1f} GB disk)")
    print(f"  Metrics: every {args.eval_every} steps (~{metrics_mb:.0f} MB disk)")

    if args.dry_run:
        for i, c in enumerate(configs):
            D = c["task"]["K"] * c["task"]["n_b"]
            print(f"  [{i+1:3d}] D={D:>6d}  K={c['task']['K']:>2d}  "
                  f"seed={c['training']['model_seed']}  "
                  f"exp={c['experiment_name']}")
        return

    # Launch pool
    os.chdir(PROJECT_ROOT)
    ctx = get_context("spawn")
    work_items = [(c, i, len(configs)) for i, c in enumerate(configs)]

    completed = 0
    failed = 0
    total = len(configs)
    t_start = time.time()

    print(f"\nStarting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 72)

    with ctx.Pool(processes=args.workers) as pool:
        for run_label, success, msg, elapsed in pool.imap_unordered(
            _run_single_experiment, work_items
        ):
            completed += 1
            status = "OK" if success else "FAIL"
            print(f"[{completed:3d}/{total}] {status}  {run_label:30s}  {msg}")
            if not success:
                failed += 1

    wall = time.time() - t_start
    print("-" * 72)
    print(f"Finished: {completed - failed}/{total} succeeded, "
          f"{failed} failed, total wall time {wall:.0f}s "
          f"({wall/60:.1f} min)")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
