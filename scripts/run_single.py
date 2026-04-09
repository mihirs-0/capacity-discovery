"""Run a single experiment from CLI args or a config file."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import ExperimentConfig, TaskConfig, ModelConfig, TrainingConfig, EvalConfig
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Run a single training experiment")
    parser.add_argument("--config", type=str, help="Path to config JSON file")

    # Task params
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n_b", type=int, default=500)
    parser.add_argument("--data_seed", type=int, default=42)

    # Model params
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_mlp", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)

    # Training params
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Eval params
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=200)

    # Experiment name
    parser.add_argument("--experiment_name", type=str, default="default")

    # Device
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig(
            task=TaskConfig(
                K=args.K,
                n_b=args.n_b,
                data_seed=args.data_seed,
            ),
            model=ModelConfig(
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                d_mlp=args.d_mlp,
            ),
            training=TrainingConfig(
                lr=args.lr,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                model_seed=args.model_seed,
                warmup_steps=args.warmup_steps,
            ),
            eval=EvalConfig(
                eval_every=args.eval_every,
                checkpoint_every=args.checkpoint_every,
            ),
            experiment_name=args.experiment_name,
        )

    import torch
    device = None
    if args.device:
        device = torch.device(args.device)

    trainer = Trainer(config, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
