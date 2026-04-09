#!/usr/bin/env python3
"""Experiment A: Small model test.

Runs a small model (d_model=32, d_mlp=128, ~40K params) at D=10000, K=20
to test behavior near the capacity limit.

Two noise conditions:
  - lr=1e-3 (baseline), 5 seeds
  - lr=3e-3 (high noise), 5 seeds

Total: 10 runs, stored under results/small_model_test/
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


def make_config(lr: float, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskConfig(K=20, n_b=500, data_seed=10000),
        model=ModelConfig(
            n_layers=4,
            n_heads=4,
            d_model=32,
            d_mlp=128,
        ),
        training=TrainingConfig(
            lr=lr,
            max_steps=50_000,
            model_seed=seed,
        ),
        eval=EvalConfig(
            eval_every=100,
            checkpoint_every=2500,
        ),
        experiment_name=f"small_model_test",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--early-stop", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Print small model param count
    from src.model import Transformer
    test_model = Transformer(n_layers=4, n_heads=4, d_model=32, d_mlp=128)
    print(f"Small model parameters: {test_model.count_parameters():,}")
    print(f"D=10000, params/example = {test_model.count_parameters() / 10000:.1f}")
    del test_model

    configs = []
    # 5 seeds at lr=1e-3
    for seed in range(5):
        cfg = make_config(lr=1e-3, seed=seed)
        cfg.training.max_steps = args.max_steps
        cfg.eval.eval_every = args.eval_every
        configs.append(("lr=1e-3", cfg))
    # 5 seeds at lr=3e-3
    for seed in range(5):
        cfg = make_config(lr=3e-3, seed=seed)
        cfg.training.max_steps = args.max_steps
        cfg.eval.eval_every = args.eval_every
        configs.append(("lr=3e-3", cfg))

    early = args.early_stop if args.early_stop > 0 else None
    print(f"\nRuns: {len(configs)}, steps/run: {args.max_steps}, "
          f"eval every: {args.eval_every}")
    print(f"Early stop: {'loss < ' + str(early) if early else 'disabled'}")
    print()

    completed = 0
    failed = 0
    t_global = time.time()

    for i, (label, config) in enumerate(configs):
        # Override run directory to include lr in the path
        lr_str = f"lr{config.training.lr:.0e}".replace("+", "")
        run_label = f"{label}_seed={config.training.model_seed}"
        print(f"[{i+1}/{len(configs)}] {run_label} ... ", end="", flush=True)

        t0 = time.time()
        try:
            # Customize run_dir to separate lr conditions
            config.experiment_name = f"small_model_test/lr_{lr_str}"
            trainer = Trainer(config, device=device, quiet=True,
                              early_stop_loss=early)
            final = trainer.train()
            elapsed = time.time() - t0
            stopped = final.get("step", args.max_steps)
            print(f"loss={final['train_loss']:.4f}  gap={final['z_shuffle_gap']:.4f}  "
                  f"step={stopped}  {elapsed:.0f}s")
            completed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.0f}s): {e}")
            failed += 1

        if "trainer" in dir():
            del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall = time.time() - t_global
    print(f"\nDone: {completed}/{len(configs)} succeeded, {failed} failed, "
          f"{wall:.0f}s ({wall/60:.1f} min)")


if __name__ == "__main__":
    main()
