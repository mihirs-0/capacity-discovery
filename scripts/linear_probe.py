#!/usr/bin/env python3
"""Linear probe the D=100K plateau checkpoint.

Question: at step 50K of D=100K (loss frozen at log K, model stuck), is
the correct A token information present in the residual stream after
layer 0?

If probe accuracy >> chance: routing info is there, downstream can't read it.
                              Bottleneck = unembedding/MLP integration.
If probe accuracy ≈ chance:  attention is rising but values are random.
                              Bottleneck = OV circuit hasn't formed yet.

We probe at the A-prediction positions (10, 11, 12, 13) — the positions
where the model needs to predict A given B and z.

Method:
1. Load D=100K seed=0, step=50000 checkpoint.
2. Run forward, capturing residual stream after each block.
3. For each (D=100K) example, get the residual at the A-prediction positions.
4. Train a linear classifier (logistic regression) to predict the
   correct A-token at each position from the residual.
5. Train/test split: 80/20 on the 100K examples.
6. Compare to chance (1/36 for the alphabet) and to the random-init
   baseline.

Also probe at multiple layer depths to see WHERE the information enters.
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap
from src.tokenizer import CharTokenizer


def strip_compiled_prefix(state_dict):
    new = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new[k[len("_orig_mod."):]] = v
        else:
            new[k] = v
    return new


@torch.no_grad()
def collect_residual_streams(model, dataset_tensor, device, batch_size=1024):
    """Run forward, capturing the residual stream after each transformer block.

    Returns:
        residuals: list of (D, T, d_model) tensors, one per layer (after block).
        Layer 0 is "after block 0", etc.
    """
    n_layers = len(model.blocks)
    D = dataset_tensor.shape[0]

    # Pre-allocate storage on CPU to avoid OOM
    residuals = [
        torch.zeros(D, model.max_seq_len, model.d_model, dtype=torch.float32)
        for _ in range(n_layers + 1)  # +1 for "after embedding, before block 0"
    ]

    model.eval()
    for start in range(0, D, batch_size):
        batch = dataset_tensor[start:start + batch_size].to(device)
        T = batch.shape[1]
        pos = torch.arange(T, device=device)
        x = model.tok_embed(batch) + model.pos_embed(pos)
        residuals[0][start:start + batch.shape[0]] = x.detach().cpu().float()
        for li, block in enumerate(model.blocks):
            x = block(x)
            residuals[li + 1][start:start + batch.shape[0]] = x.detach().cpu().float()

    return residuals  # length n_layers + 1


def linear_probe(X_train, y_train, X_test, y_test, n_classes, device, n_epochs=200):
    """Fit a linear classifier (logistic regression) on residual features.

    Returns:
        accuracy on test set.
    """
    d = X_train.shape[1]
    W = torch.zeros(d, n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    optimizer = torch.optim.Adam([W, b], lr=0.05)

    for epoch in range(n_epochs):
        logits = X_train @ W + b
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = X_test @ W + b
        preds = test_logits.argmax(dim=-1)
        acc = (preds == y_test).float().mean().item()
    return acc


def probe_checkpoint(ckpt_path, D, K, label, device):
    """Load a checkpoint and probe each layer's residual stream."""
    print(f"\n=== {label} ===")
    print(f"  Checkpoint: {ckpt_path}")

    # Build dataset (data_seed = D in our run_fast.py convention)
    n_b = D // K
    dataset = SurjectiveMap(K=K, n_b=n_b, seed=D)
    print(f"  Dataset: D={dataset.D}, K={K}, n_b={n_b}")

    # Build model and load weights
    model_cfg = ModelConfig()
    model = Transformer.from_config(model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compiled_prefix(state)
    model.load_state_dict(state)

    # Collect residual streams (D x T x d_model) for every layer
    print(f"  Running forward to collect residuals...")
    residuals = collect_residual_streams(model, dataset.data, device, batch_size=1024)
    n_layer_outputs = len(residuals)  # = n_layers + 1

    # Targets: A tokens at positions 11, 12, 13, 14
    # Probe predicts target at position p+1 from residual at position p.
    a_pred_positions = [10, 11, 12, 13]
    a_target_positions = [11, 12, 13, 14]

    # Use vocabulary 4..39 (alphanumeric) — 36 classes
    n_classes = 36
    alpha_offset = CharTokenizer.ALPHA_OFFSET  # 4

    # Train/test split
    rng = np.random.RandomState(42)
    n = dataset.D
    perm = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = torch.tensor(perm[:split], dtype=torch.long)
    test_idx = torch.tensor(perm[split:], dtype=torch.long)

    # Targets per A position
    targets = dataset.data  # (D, 16)

    print(f"  Probing each layer at each A position...")
    print(f"  {'layer':>10}  {'pos1':>7}  {'pos2':>7}  {'pos3':>7}  {'pos4':>7}  {'mean':>7}")
    print(f"  {'-' * 50}")

    layer_names = ["embed"] + [f"after_L{i}" for i in range(n_layer_outputs - 1)]

    results = {}
    for li in range(n_layer_outputs):
        accs = []
        for pred_pos, target_pos in zip(a_pred_positions, a_target_positions):
            # Features: residual at pred_pos
            X = residuals[li][:, pred_pos, :]  # (D, d_model)
            # Targets: token at target_pos, shifted to 0..35
            y = targets[:, target_pos] - alpha_offset  # (D,)

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Center features
            mean = X_train.mean(dim=0, keepdim=True)
            X_train = X_train - mean
            X_test = X_test - mean

            acc = linear_probe(X_train, y_train, X_test, y_test, n_classes, device)
            accs.append(acc)

        mean_acc = float(np.mean(accs))
        results[layer_names[li]] = {"per_pos": accs, "mean": mean_acc}
        print(f"  {layer_names[li]:>10}  "
              f"{accs[0]:>7.3f}  {accs[1]:>7.3f}  {accs[2]:>7.3f}  {accs[3]:>7.3f}  "
              f"{mean_acc:>7.3f}")

    chance = 1.0 / n_classes
    print(f"\n  Chance accuracy (1/36): {chance:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str,
                        default="results/phase1_d_sweep")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Device: {device}")

    runs_dir = os.path.join(args.experiment_dir, "runs")

    # Find available D=100K checkpoints (latest step)
    targets = []
    # Primary: D=100K at the latest step
    for D in [100000]:
        for seed in range(5):
            ckpt_dir = os.path.join(runs_dir, f"D{D}_seed{seed}", "checkpoints")
            if not os.path.isdir(ckpt_dir):
                continue
            ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
                key=lambda f: int(re.search(r"step_(\d+)", f).group(1)),
            )
            if not ckpts:
                continue
            latest = ckpts[-1]
            step = int(re.search(r"step_(\d+)", latest).group(1))
            ckpt_path = os.path.join(ckpt_dir, latest)
            targets.append((ckpt_path, D, f"D={D} seed={seed} step={step}"))
            break  # one seed is enough for the headline result

    # Comparison: D=20K converged checkpoint (latest)
    for D in [20000]:
        for seed in range(5):
            ckpt_dir = os.path.join(runs_dir, f"D{D}_seed{seed}", "checkpoints")
            if not os.path.isdir(ckpt_dir):
                continue
            ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
                key=lambda f: int(re.search(r"step_(\d+)", f).group(1)),
            )
            if not ckpts:
                continue
            latest = ckpts[-1]
            step = int(re.search(r"step_(\d+)", latest).group(1))
            ckpt_path = os.path.join(ckpt_dir, latest)
            targets.append((ckpt_path, D, f"D={D} seed={seed} step={step} (converged)"))
            break

    # Baseline: D=100K at step 0 (random init)
    for D in [100000]:
        for seed in range(5):
            ckpt_dir = os.path.join(runs_dir, f"D{D}_seed{seed}", "checkpoints")
            step0 = os.path.join(ckpt_dir, "step_0.pt")
            if os.path.exists(step0):
                targets.append((step0, D, f"D={D} seed={seed} step=0 (RANDOM INIT)"))
                break

    if not targets:
        print("No checkpoints found.")
        return

    all_results = {}
    for ckpt_path, D, label in targets:
        all_results[label] = probe_checkpoint(ckpt_path, D, args.K, label, device)

    # Save results
    out_path = os.path.join(args.experiment_dir, "linear_probe_results.json")
    import json
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
