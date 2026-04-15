#!/usr/bin/env python3
"""Train the forward task (A -> B) — custom loop.

Cannot use src.trainer.Trainer because model.py hardcodes the backward task's
loss positions (10..14). The forward task computes loss on positions 5..11
predicting 6..12 (B tokens + EOS), so we run the loop here and compute loss
manually from the logits.

Mirrors the backward trainer in everything else: same model architecture,
same AdamW settings, cosine warmup, full-dataset diagnostic evals.

Usage:
    python scripts/run_forward.py --model_seed 0 --max_steps 50000
    python scripts/run_forward.py --model_seed 0 --max_steps 500 \
        --experiment_name forward_task_sanity --eval_every 100 --checkpoint_every 100
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
from tqdm import tqdm

from src.config import ModelConfig
from src.model import Transformer
from src.task_forward import ForwardSurjectiveMap


def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    return base_lr


def compute_forward_loss(model: Transformer, batch: torch.Tensor,
                         reduction: str = "mean") -> torch.Tensor:
    """Forward-task loss: predict positions 6..12 (b1..b6, EOS) from logits
    at positions 5..11 (SEP..b6)."""
    _, logits = model(batch)
    ls = ForwardSurjectiveMap.LOSS_START       # 5
    le = ForwardSurjectiveMap.LOSS_END         # 11 inclusive
    loss_logits = logits[:, ls:le + 1]                 # (B, 7, V)
    loss_targets = batch[:, ls + 1:le + 2]             # (B, 7)
    return F.cross_entropy(
        loss_logits.reshape(-1, logits.size(-1)),
        loss_targets.reshape(-1),
        reduction=reduction,
    )


@torch.no_grad()
def eval_full(model: Transformer, dataset: ForwardSurjectiveMap,
              device: torch.device, batch_size: int = 2048) -> dict:
    model.eval()
    data = dataset.get_full()
    D = data.shape[0]
    ls = ForwardSurjectiveMap.LOSS_START
    le = ForwardSurjectiveMap.LOSS_END
    n_pos = le - ls + 1                        # 7

    total_loss = 0.0
    per_pos = [0.0] * n_pos                    # b1..b6 + EOS
    correct_all = 0
    total_examples = 0

    for start in range(0, D, batch_size):
        batch = data[start:start + batch_size].to(device)
        n = batch.shape[0]
        _, logits = model(batch)
        loss_logits = logits[:, ls:le + 1]             # (n, 7, V)
        loss_targets = batch[:, ls + 1:le + 2]         # (n, 7)
        loss = F.cross_entropy(
            loss_logits.reshape(-1, logits.size(-1)),
            loss_targets.reshape(-1),
        )
        total_loss += loss.item() * n
        for i in range(n_pos):
            l = F.cross_entropy(loss_logits[:, i], loss_targets[:, i])
            per_pos[i] += l.item() * n
        preds = loss_logits.argmax(dim=-1)              # (n, 7)
        correct_all += (preds == loss_targets).all(dim=-1).sum().item()
        total_examples += n

    metrics = {
        "train_loss": total_loss / total_examples,
        "train_acc_exact": correct_all / total_examples,
    }
    labels = ["b1", "b2", "b3", "b4", "b5", "b6", "eos"]
    for label, v in zip(labels, per_pos):
        metrics[f"train_loss_{label}"] = v / total_examples
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n_b", type=int, default=500)
    parser.add_argument("--data_seed", type=int, default=10000)
    parser.add_argument("--model_seed", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--experiment_name", type=str, default="forward_task")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--early_stop_key", type=str, default="train_loss_b1",
                        help="Which eval-metric to watch for early stopping. "
                             "train_loss_b1 is the right signal because b2..b6 + eos "
                             "become trivial once the model can autoregress inside B.")
    parser.add_argument("--early_stop_thresh", type=float, default=1e-3,
                        help="Stop when early_stop_key < thresh for `patience` "
                             "consecutive evals. Set to 0 or negative to disable.")
    parser.add_argument("--early_stop_patience", type=int, default=2,
                        help="Consecutive evals below thresh before stopping.")

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    # Dataset
    dataset = ForwardSurjectiveMap(K=args.K, n_b=args.n_b, seed=args.data_seed)
    D_nominal = args.K * args.n_b
    print(f"Forward dataset: D={dataset.D} (nominal {D_nominal}, "
          f"{dataset.n_collisions} colliding A strings dropped)")

    # Model
    torch.manual_seed(args.model_seed)
    model_cfg = ModelConfig()
    model = Transformer.from_config(model_cfg).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Device: {device}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay, eps=args.eps,
    )

    # Output directory — mirrors trainer.py layout:
    #   results/{experiment_name}/runs/D{D_nominal}_seed{seed}/
    run_dir = os.path.join(
        "results", args.experiment_name, "runs",
        f"D{D_nominal}_seed{args.model_seed}"
    )
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    # Save config alongside run (useful for later analysis)
    config_path = os.path.join("results", args.experiment_name, "config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({
            "task": {
                "direction": "forward",
                "K": args.K, "n_b": args.n_b, "D": D_nominal,
                "D_actual": dataset.D, "n_collisions": dataset.n_collisions,
                "data_seed": args.data_seed,
            },
            "model": {
                "n_layers": model_cfg.n_layers, "n_heads": model_cfg.n_heads,
                "d_model": model_cfg.d_model, "d_mlp": model_cfg.d_mlp,
                "vocab_size": model_cfg.vocab_size, "max_seq_len": model_cfg.max_seq_len,
            },
            "training": {
                "lr": args.lr, "batch_size": args.batch_size,
                "max_steps": args.max_steps, "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "beta1": args.beta1, "beta2": args.beta2, "eps": args.eps,
                "model_seed": args.model_seed,
            },
            "eval": {
                "eval_every": args.eval_every,
                "checkpoint_every": args.checkpoint_every,
            },
        }, f, indent=2)

    batch_rng = np.random.RandomState(args.model_seed + 10000)
    t_start = time.time()
    last_ckpt_step = -1

    def save_checkpoint(step):
        path = os.path.join(run_dir, "checkpoints", f"step_{step}.pt")
        torch.save(model.state_dict(), path)

    def log(step):
        model.eval()
        metrics = {"step": step}
        metrics.update(eval_full(model, dataset, device))
        metrics["wall_time_seconds"] = time.time() - t_start
        if device.type == "cuda":
            metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
        else:
            metrics["gpu_memory_mb"] = 0.0
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        model.train()
        return metrics

    # Initial checkpoint + eval
    save_checkpoint(0)
    last_ckpt_step = 0
    init_metrics = log(0)
    print(f"[step 0] loss={init_metrics['train_loss']:.4f}")

    step_iter = range(1, args.max_steps + 1)
    if not args.quiet:
        step_iter = tqdm(step_iter, desc=f"Forward seed={args.model_seed}")

    early_stop_below = 0          # consecutive evals with metric below thresh
    early_stopped = False
    early_stop_step = None

    model.train()
    for step in step_iter:
        # LR schedule
        lr = get_lr(step, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward training step
        batch = dataset.get_batch(args.batch_size, batch_rng).to(device)
        loss = compute_forward_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval
        if step % args.eval_every == 0:
            m = log(step)
            if not args.quiet and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(
                    loss=f"{m['train_loss']:.4f}",
                    acc=f"{m['train_acc_exact']:.3f}",
                )
            # Early stopping check
            if args.early_stop_thresh > 0:
                v = m.get(args.early_stop_key, float("inf"))
                if v < args.early_stop_thresh:
                    early_stop_below += 1
                else:
                    early_stop_below = 0
                if early_stop_below >= args.early_stop_patience:
                    early_stopped = True
                    early_stop_step = step
                    save_checkpoint(step)
                    last_ckpt_step = step
                    if not args.quiet:
                        print(f"\n[early stop] {args.early_stop_key}={v:.2e} "
                              f"< {args.early_stop_thresh} for "
                              f"{args.early_stop_patience} consecutive evals "
                              f"at step {step}")
                    break

        # Checkpoint
        if step % args.checkpoint_every == 0:
            save_checkpoint(step)
            last_ckpt_step = step

    # Final eval + checkpoint — skip if we already stopped early (everything
    # was captured at the stopping step). Otherwise ensure step == max_steps is
    # logged and checkpointed exactly once.
    if not early_stopped:
        if last_ckpt_step != args.max_steps:
            save_checkpoint(args.max_steps)
        if args.max_steps % args.eval_every == 0:
            with open(metrics_path) as f:
                final = json.loads(f.readlines()[-1])
        else:
            final = log(args.max_steps)
    else:
        with open(metrics_path) as f:
            final = json.loads(f.readlines()[-1])
    wall = time.time() - t_start
    print(f"\nDone in {wall:.0f}s ({wall/60:.1f} min). "
          f"{'Early-stopped at step ' + str(early_stop_step) if early_stopped else 'Full run.'}")
    print(f"Final loss: {final['train_loss']:.4f}  "
          f"acc_exact: {final['train_acc_exact']:.3f}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
