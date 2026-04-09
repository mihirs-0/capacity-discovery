"""Training loop with eval hooks and checkpointing."""

import json
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .model import Transformer
from .task import SurjectiveMap
from .diagnostics import (
    compute_train_loss,
    compute_z_shuffle_gap,
    compute_group_accuracy,
    compute_stable_ranks,
)


def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """Cosine warmup, then constant."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


class Trainer:
    def __init__(self, config: ExperimentConfig, device: torch.device | None = None,
                 quiet: bool = False, early_stop_loss: float | None = None):
        self.config = config
        self.quiet = quiet
        self.early_stop_loss = early_stop_loss
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Build dataset
        self.dataset = SurjectiveMap(
            K=config.task.K,
            n_b=config.task.n_b,
            len_b=config.task.len_b,
            len_a=config.task.len_a,
            len_z=config.task.len_z,
            vocab_size=config.task.vocab_size,
            seed=config.task.data_seed,
        )

        # Build model
        torch.manual_seed(config.training.model_seed)
        self.model = Transformer.from_config(config.model).to(self.device)
        param_count = self.model.count_parameters()
        if not self.quiet:
            print(f"Model parameters: {param_count:,}")

        # Compile model for faster training (fuses small CUDA kernels)
        if self.device.type == "cuda" and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Build optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
            eps=config.training.eps,
        )

        # Batch RNG (separate from data RNG)
        self.batch_rng = np.random.RandomState(config.training.model_seed + 10000)

        # Output directory
        D = config.task.D
        seed = config.training.model_seed
        self.run_dir = os.path.join(
            "results", config.experiment_name, "runs",
            f"D{D}_seed{seed}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "hessian"), exist_ok=True)

        # Save config
        config.save(os.path.join(self.run_dir, "..", "..", "config.json"))

        # Metrics file
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        # Clear if exists
        open(self.metrics_path, "w").close()

        self.start_time = time.time()

    def _log_metrics(self, metrics: dict):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def _save_checkpoint(self, step: int):
        path = os.path.join(self.run_dir, "checkpoints", f"step_{step}.pt")
        torch.save(self.model.state_dict(), path)

    def _run_eval(self, step: int) -> dict:
        """Run all diagnostics and return metrics dict."""
        full_data = self.dataset.get_full()

        metrics = {"step": step}

        # D1: training loss
        loss_metrics = compute_train_loss(self.model, full_data, self.device)
        metrics.update(loss_metrics)

        # D3: z-shuffle gap
        shuffle_metrics = compute_z_shuffle_gap(
            self.model, full_data, self.device,
            batch_size=self.config.eval.z_shuffle_batch_size,
        )
        metrics.update(shuffle_metrics)

        # D4: group accuracy
        acc_metrics = compute_group_accuracy(
            self.model, self.dataset, self.device,
            n_groups=self.config.eval.group_eval_n_groups,
        )
        metrics.update(acc_metrics)

        # D6: stable ranks
        rank_metrics = compute_stable_ranks(self.model)
        metrics.update(rank_metrics)

        # Timing
        metrics["wall_time_seconds"] = time.time() - self.start_time

        # GPU memory
        if self.device.type == "cuda":
            metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
        else:
            metrics["gpu_memory_mb"] = 0.0

        return metrics

    def train(self):
        """Run training loop."""
        cfg = self.config
        max_steps = cfg.training.max_steps
        eval_every = cfg.eval.eval_every
        ckpt_every = cfg.eval.checkpoint_every

        self.model.train()

        if self.quiet:
            step_iter = range(max_steps)
        else:
            step_iter = tqdm(range(max_steps), desc="Training")

        for step in step_iter:
            # Eval
            if step % eval_every == 0:
                metrics = self._run_eval(step)
                self._log_metrics(metrics)
                if not self.quiet and hasattr(step_iter, "set_postfix"):
                    step_iter.set_postfix(
                        loss=f"{metrics['train_loss']:.4f}",
                        gap=f"{metrics['z_shuffle_gap']:.4f}",
                    )
                # Early stopping
                if (self.early_stop_loss is not None
                        and metrics["train_loss"] < self.early_stop_loss):
                    self._save_checkpoint(step)
                    if not self.quiet:
                        print(f"\nEarly stop at step {step}: "
                              f"loss {metrics['train_loss']:.4f} < {self.early_stop_loss}")
                    return metrics
                self.model.train()

            # Checkpoint
            if step % ckpt_every == 0:
                self._save_checkpoint(step)

            # LR schedule
            lr = get_lr(step, cfg.training.warmup_steps, cfg.training.lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Sample batch
            batch = self.dataset.get_batch(cfg.training.batch_size, self.batch_rng)
            batch = batch.to(self.device)

            # Forward
            loss, _ = self.model(batch, batch)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Final eval and checkpoint
        metrics = self._run_eval(max_steps)
        self._log_metrics(metrics)
        self._save_checkpoint(max_steps)

        if not self.quiet:
            print(f"\nTraining complete. Results saved to {self.run_dir}")
            print(f"Final loss: {metrics['train_loss']:.4f}")
            print(f"Final z-shuffle gap: {metrics['z_shuffle_gap']:.4f}")

        return metrics
