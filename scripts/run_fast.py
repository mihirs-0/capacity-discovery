#!/usr/bin/env python3
"""Fast sequential runner — single process, no multiprocessing overhead.

For small models on powerful GPUs, multiprocessing HURTS because:
  - Each process creates a ~800 MB CUDA context
  - CUDA driver serializes kernel launches across contexts
  - torch.compile cache is per-process (can't share)

This script runs all experiments sequentially in ONE process.
torch.compile compiles once on the first run, then reuses for all
subsequent runs (same architecture). CUDA context stays warm.

Usage:
  python scripts/run_fast.py --phase all
  python scripts/run_fast.py --phase 1 --max-steps 2000  # quick test
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from src.config import ExperimentConfig, TaskConfig, ModelConfig, TrainingConfig, EvalConfig
from src.trainer import Trainer


def make_config(K, n_b, data_seed, model_seed, experiment_name,
                max_steps=50000, checkpoint_every=2500, eval_every=100):
    return ExperimentConfig(
        task=TaskConfig(K=K, n_b=n_b, data_seed=data_seed),
        model=ModelConfig(),
        training=TrainingConfig(max_steps=max_steps, model_seed=model_seed),
        eval=EvalConfig(eval_every=eval_every, checkpoint_every=checkpoint_every),
        experiment_name=experiment_name,
    )


def phase1_configs(max_steps, checkpoint_every, eval_every):
    K = 20
    D_values = [1000, 3000, 5000, 10000, 20000, 50000, 100000]
    seeds = [0, 1, 2, 3, 4]
    configs = []
    for D in D_values:
        n_b = D // K
        for seed in seeds:
            configs.append(make_config(
                K=K, n_b=n_b, data_seed=D, model_seed=seed,
                experiment_name="phase1_d_sweep",
                max_steps=max_steps, checkpoint_every=checkpoint_every,
                eval_every=eval_every,
            ))
    return configs


def phase1_5_configs(max_steps, checkpoint_every, eval_every):
    D = 10000
    K_values = [5, 10, 20]
    seeds = [0, 1, 2, 3, 4]
    configs = []
    for K in K_values:
        n_b = D // K
        for seed in seeds:
            configs.append(make_config(
                K=K, n_b=n_b, data_seed=10000, model_seed=seed,
                experiment_name="phase1_5_k_sweep",
                max_steps=max_steps, checkpoint_every=checkpoint_every,
                eval_every=eval_every,
            ))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "1.5", "all"], default="all")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-every", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = []
    if args.phase in ("1", "all"):
        configs.extend(phase1_configs(args.max_steps, args.checkpoint_every, args.eval_every))
    if args.phase in ("1.5", "all"):
        configs.extend(phase1_5_configs(args.max_steps, args.checkpoint_every, args.eval_every))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"Runs: {len(configs)}, steps/run: {args.max_steps}, eval every: {args.eval_every}")
    print(f"torch.compile: {'yes' if hasattr(torch, 'compile') else 'no'}")
    print()

    if args.dry_run:
        for i, c in enumerate(configs):
            print(f"  [{i+1:3d}] D={c.task.D:>6d}  K={c.task.K:>2d}  seed={c.training.model_seed}")
        return

    completed = 0
    failed = 0
    t_global = time.time()

    for i, config in enumerate(configs):
        label = f"D={config.task.D}_K={config.task.K}_seed={config.training.model_seed}"
        print(f"[{i+1}/{len(configs)}] {label} ... ", end="", flush=True)

        t0 = time.time()
        try:
            trainer = Trainer(config, device=device, quiet=True)
            final = trainer.train()
            elapsed = time.time() - t0
            print(f"loss={final['train_loss']:.4f}  gap={final['z_shuffle_gap']:.4f}  {elapsed:.0f}s")
            completed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.0f}s): {e}")
            failed += 1

        # Free memory between runs
        del trainer
        torch.cuda.empty_cache()

    wall = time.time() - t_global
    print(f"\nDone: {completed}/{len(configs)} succeeded, {failed} failed, "
          f"{wall:.0f}s ({wall/60:.1f} min)")


if __name__ == "__main__":
    main()
