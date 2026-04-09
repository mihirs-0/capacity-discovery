#!/usr/bin/env python3
"""Experiment B: Z-attention tracking during plateau.

Single training run (D=10K, K=20, default model, seed=0) that logs
per-head attention to z-positions every 10 steps. Reveals whether
the routing circuit builds incrementally or nucleates suddenly.

Output:
  results/attention_tracking/seed_0/metrics.jsonl     (every 100 steps)
  results/attention_tracking/seed_0/attention.jsonl   (every 10 steps)
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
from tqdm import tqdm

from src.config import ExperimentConfig, TaskConfig, ModelConfig, TrainingConfig, EvalConfig
from src.model import Transformer
from src.task import SurjectiveMap
from src.tokenizer import CharTokenizer
from src.diagnostics import (
    compute_train_loss,
    compute_z_shuffle_gap,
    compute_group_accuracy,
    compute_stable_ranks,
)


def compute_z_attention(model, batch, n_layers, n_heads):
    """Compute per-head attention scores from A-prediction positions to z-positions,
    and from z-positions to B-positions.

    Sequence layout:
        [BOS] b1 b2 b3 b4 b5 b6 [SEP] z1 z2 [SEP] a1 a2 a3 a4 [EOS]
        pos:  0   1  2  3  4  5  6    7    8  9   10  11 12 13 14  15

    A-prediction positions: 10, 11, 12, 13
        (logit at pos 10 predicts a1 at pos 11, etc.)
    z-positions: 8, 9
    B-positions: 1, 2, 3, 4, 5, 6
    """
    z_positions = [8, 9]
    a_pred_positions = [10, 11, 12, 13]
    b_positions = [1, 2, 3, 4, 5, 6]

    model.eval()
    with torch.no_grad():
        # Use the underlying model (unwrap torch.compile if needed)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        logits, all_attn = raw_model.forward_with_attention(batch)

    result = {}
    for li, attn_weights in enumerate(all_attn):
        # attn_weights: (batch, n_heads, seq_len, seq_len)
        for hi in range(n_heads):
            # How much A-prediction positions attend to z-positions
            # attn[batch, head, a_pred_pos, z_pos]
            z_attn = attn_weights[:, hi][:, a_pred_positions][:, :, z_positions]
            # shape: (batch, 4, 2) -> sum over z-positions, mean over a-positions, mean over batch
            score = z_attn.sum(dim=-1).mean(dim=-1).mean(dim=0).item()
            result[f"z_attn_L{li}H{hi}"] = score

            # How much z-positions attend to B-positions (z "reading" B)
            z_reads_b = attn_weights[:, hi][:, z_positions][:, :, b_positions]
            # shape: (batch, 2, 6) -> sum over b-positions, mean over z-positions, mean over batch
            score_b = z_reads_b.sum(dim=-1).mean(dim=-1).mean(dim=0).item()
            result[f"z_reads_b_L{li}H{hi}"] = score_b

    return result


def get_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = ExperimentConfig(
        task=TaskConfig(K=20, n_b=500, data_seed=10000),
        model=ModelConfig(),
        training=TrainingConfig(max_steps=50_000, model_seed=0),
        eval=EvalConfig(eval_every=100, checkpoint_every=2500),
        experiment_name="attention_tracking",
    )

    # Build dataset
    dataset = SurjectiveMap(
        K=config.task.K, n_b=config.task.n_b,
        len_b=config.task.len_b, len_a=config.task.len_a,
        len_z=config.task.len_z, vocab_size=config.task.vocab_size,
        seed=config.task.data_seed,
    )

    # Verify token layout
    tok = CharTokenizer()
    sample = dataset.data[0].tolist()
    print("\nToken layout verification:")
    print(f"  Sequence: {[tok.decode_id(t) for t in sample]}")
    print(f"  Positions: {list(range(16))}")
    print(f"  z-positions: 8, 9 (tokens: {tok.decode_id(sample[8])}, {tok.decode_id(sample[9])})")
    print(f"  A-prediction positions: 10, 11, 12, 13")
    print(f"  B-positions: 1-6")
    print()

    # Build model (no torch.compile — we need forward_with_attention)
    torch.manual_seed(config.training.model_seed)
    model = Transformer.from_config(config.model).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    n_layers = config.model.n_layers
    n_heads = config.model.n_heads

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay, eps=config.training.eps,
    )

    batch_rng = np.random.RandomState(config.training.model_seed + 10000)
    # Fixed batch for attention tracking (consistent across steps)
    attn_batch = dataset.get_batch(256, np.random.RandomState(999)).to(device)

    # Output
    run_dir = os.path.join("results", "attention_tracking", "seed_0")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    attention_path = os.path.join(run_dir, "attention.jsonl")
    open(metrics_path, "w").close()
    open(attention_path, "w").close()

    config.save(os.path.join(run_dir, "..", "config.json"))

    max_steps = config.training.max_steps
    t_start = time.time()

    print(f"Training {max_steps} steps, attention logged every 10 steps\n")
    pbar = tqdm(range(max_steps), desc="Training")

    for step in pbar:
        # Attention tracking every 10 steps
        if step % 10 == 0:
            attn_metrics = {"step": step}
            attn_metrics.update(compute_z_attention(model, attn_batch, n_layers, n_heads))
            with open(attention_path, "a") as f:
                f.write(json.dumps(attn_metrics) + "\n")
            model.train()

        # Full diagnostics every 100 steps
        if step % 100 == 0:
            model.eval()
            full_data = dataset.get_full()
            metrics = {"step": step}
            metrics.update(compute_train_loss(model, full_data, device))
            metrics.update(compute_z_shuffle_gap(model, full_data, device))
            metrics.update(compute_group_accuracy(model, dataset, device))
            metrics.update(compute_stable_ranks(model))
            metrics["wall_time_seconds"] = time.time() - t_start
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            pbar.set_postfix(
                loss=f"{metrics['train_loss']:.4f}",
                gap=f"{metrics['z_shuffle_gap']:.4f}",
            )
            model.train()

        # Checkpoint
        if step % 2500 == 0:
            path = os.path.join(run_dir, "checkpoints", f"step_{step}.pt")
            torch.save(model.state_dict(), path)

        # LR schedule
        lr = get_lr(step, config.training.warmup_steps, config.training.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Training step
        batch = dataset.get_batch(config.training.batch_size, batch_rng).to(device)
        loss, _ = model(batch, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final eval
    model.eval()
    full_data = dataset.get_full()
    metrics = {"step": max_steps}
    metrics.update(compute_train_loss(model, full_data, device))
    metrics.update(compute_z_shuffle_gap(model, full_data, device))
    metrics.update(compute_group_accuracy(model, dataset, device))
    metrics["wall_time_seconds"] = time.time() - t_start
    with open(metrics_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    # Final attention
    attn_metrics = {"step": max_steps}
    attn_metrics.update(compute_z_attention(model, attn_batch, n_layers, n_heads))
    with open(attention_path, "a") as f:
        f.write(json.dumps(attn_metrics) + "\n")

    # Final checkpoint
    torch.save(model.state_dict(), os.path.join(run_dir, "checkpoints", f"step_{max_steps}.pt"))

    wall = time.time() - t_start
    print(f"\nDone. {wall:.0f}s ({wall/60:.1f} min)")
    print(f"Final loss: {metrics['train_loss']:.4f}")
    print(f"Final z-shuffle gap: {metrics['z_shuffle_gap']:.4f}")
    print(f"Attention log: {attention_path} ({max_steps // 10 + 1} entries)")
    print(f"Metrics log: {metrics_path}")


if __name__ == "__main__":
    main()
