#!/usr/bin/env python3
"""Compute Hessian eigenvalues at the plateau for D=20K vs D=100K.

The decisive question: at the loss=log K plateau, is the geometry a
saddle (lambda_min < 0, escape direction exists) or a true minimum
(lambda_min > 0, no escape direction)?

If sign(lambda_min) is the same at D=20K and D=100K plateau, the wall
is just a slowdown -- same topology, longer escape time.

If sign(lambda_min) flips between D=20K (saddle, escapes) and D=100K
(minimum, stuck), the loss landscape topology has phase-transitioned.
That's the finding.

Comparison points:
  D=20K seed=0 step=2500   (plateau phase, BEFORE snap)
  D=20K seed=0 step=17500  (post-snap, converged, sanity check)
  D=100K seed=0 step=50000 (plateau, never escapes)
  D=100K seed=0 step=0     (random init, baseline)
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap
from src.hessian import compute_hessian_eigenvalues, _hvp, _flatten_params, \
    _random_vector_like, _vector_norm, _vector_dot, _normalize


def strip_compiled_prefix(state_dict):
    new = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new[k[len("_orig_mod."):]] = v
        else:
            new[k] = v
    return new


def power_iteration_robust(model, data, device, n_iter=100, seed=0):
    """Largest eigenvalue via power iteration. Tracks Rayleigh quotient."""
    model.eval()
    rng = np.random.RandomState(seed)
    params = _flatten_params(model)
    v = _random_vector_like(params, rng)

    eigenvalue = 0.0
    history = []
    for i in range(n_iter):
        hv = _hvp(model, data, device, v)
        eigenvalue = _vector_dot(v, hv)
        history.append(eigenvalue)
        v = _normalize(hv)

    # Final residual
    hv = _hvp(model, data, device, v)
    residual_vecs = [hvi - eigenvalue * vi for hvi, vi in zip(hv, v)]
    residual = _vector_norm(residual_vecs)

    return {
        "lambda": eigenvalue,
        "history": history,
        "residual": residual,
        "n_iter": n_iter,
    }


def lobpcg_min_eigenvalue(model, data, device, n_iter=100, seed=0):
    """Smallest eigenvalue via shifted power iteration on (sigma*I - H).

    We pick sigma = lambda_max + small margin, so (sigma*I - H) has its
    largest eigenvalue corresponding to lambda_min of H. Then run power
    iteration on this shifted operator and recover lambda_min via the
    final Rayleigh quotient with H.

    This is more numerically stable than running power iteration on -H.
    """
    model.eval()
    rng = np.random.RandomState(seed + 1)
    params = _flatten_params(model)

    # First, get a rough lambda_max for shifting
    res_max = power_iteration_robust(model, data, device, n_iter=30, seed=seed)
    lam_max = res_max["lambda"]
    sigma = abs(lam_max) * 1.1 + 1.0  # comfortable margin

    # Shifted operator: (sigma * I - H) v
    v = _random_vector_like(params, rng)
    history = []

    for i in range(n_iter):
        hv = _hvp(model, data, device, v)
        # (sigma * I - H) v = sigma * v - H v
        shifted = [sigma * vi - hvi for vi, hvi in zip(v, hv)]
        # Rayleigh quotient with original H (so we can read lambda_min directly)
        ray = _vector_dot(v, hv)
        history.append(ray)
        v = _normalize(shifted)

    # Final eigenvalue with H
    hv = _hvp(model, data, device, v)
    lam_min = _vector_dot(v, hv)
    residual_vecs = [hvi - lam_min * vi for hvi, vi in zip(hv, v)]
    residual = _vector_norm(residual_vecs)

    return {
        "lambda": lam_min,
        "history": history,
        "residual": residual,
        "n_iter": n_iter,
        "sigma": sigma,
    }


def compute_loss_at_checkpoint(model, dataset_tensor, device, batch_size=2048):
    """Compute mean training loss for context."""
    model.eval()
    D = dataset_tensor.shape[0]
    total = 0.0
    n = 0
    with torch.no_grad():
        for start in range(0, D, batch_size):
            batch = dataset_tensor[start:start + batch_size].to(device)
            loss, _ = model(batch, batch)
            total += loss.item() * batch.shape[0]
            n += batch.shape[0]
    return total / n


def analyze_checkpoint(ckpt_path, D, K, label, device, n_iter, batch_size, seed):
    print(f"\n=== {label} ===")
    print(f"  Checkpoint: {ckpt_path}")

    # Build dataset
    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)

    # Build model and load (NO torch.compile -- need autograd to track)
    model_cfg = ModelConfig()
    model = Transformer.from_config(model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compiled_prefix(state)
    model.load_state_dict(state)

    # Need parameters to require grad
    for p in model.parameters():
        p.requires_grad_(True)

    # Loss for context
    train_loss = compute_loss_at_checkpoint(model, dataset.data, device)
    print(f"  Train loss: {train_loss:.4f}  (log K = {np.log(K):.4f})")

    # Sample a fixed batch for HVP
    rng = np.random.RandomState(seed)
    n = min(batch_size, dataset.D)
    indices = rng.choice(dataset.D, size=n, replace=False)
    hvp_batch = dataset.data[indices]

    # Largest eigenvalue
    print(f"  Power iteration for lambda_max ({n_iter} iters)...")
    t0 = time.time()
    res_max = power_iteration_robust(model, hvp_batch, device, n_iter=n_iter, seed=seed)
    print(f"    lambda_max = {res_max['lambda']:>+12.6e}  (residual {res_max['residual']:.3e}, {time.time()-t0:.0f}s)")

    # Smallest eigenvalue (could be negative)
    print(f"  Shifted power iteration for lambda_min ({n_iter} iters)...")
    t0 = time.time()
    res_min = lobpcg_min_eigenvalue(model, hvp_batch, device, n_iter=n_iter, seed=seed)
    print(f"    lambda_min = {res_min['lambda']:>+12.6e}  (residual {res_min['residual']:.3e}, {time.time()-t0:.0f}s)")

    sign = "POSITIVE (true local minimum)" if res_min["lambda"] > 0 else "NEGATIVE (saddle, escape direction exists)"
    print(f"    sign(lambda_min) = {sign}")

    return {
        "label": label,
        "ckpt_path": ckpt_path,
        "D": D,
        "train_loss": train_loss,
        "log_K": float(np.log(K)),
        "lambda_max": res_max["lambda"],
        "lambda_min": res_min["lambda"],
        "lambda_max_history": res_max["history"],
        "lambda_min_history": res_min["history"],
        "residual_max": res_max["residual"],
        "residual_min": res_min["residual"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, default="results/phase1_d_sweep")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        # MPS doesn't reliably support double backward; prefer CPU for correctness
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Device: {device}")

    runs_dir = os.path.join(args.experiment_dir, "runs")

    targets = []

    # Find a D=20K seed with a "plateau-phase" checkpoint (~step 2500)
    for seed in range(5):
        ckpt_dir = os.path.join(runs_dir, f"D20000_seed{seed}", "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        plateau = os.path.join(ckpt_dir, "step_2500.pt")
        final = None
        all_ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda f: int(re.search(r"step_(\d+)", f).group(1)),
        )
        if all_ckpts:
            final = os.path.join(ckpt_dir, all_ckpts[-1])
            final_step = int(re.search(r"step_(\d+)", all_ckpts[-1]).group(1))
        if os.path.exists(plateau):
            targets.append((plateau, 20000,
                            f"D=20K seed={seed} step=2500 (plateau, BEFORE snap)"))
        if final:
            targets.append((final, 20000,
                            f"D=20K seed={seed} step={final_step} (post-snap, converged)"))
        break

    # D=100K stuck plateau (latest step) and random init baseline
    for seed in range(5):
        ckpt_dir = os.path.join(runs_dir, f"D100000_seed{seed}", "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        all_ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda f: int(re.search(r"step_(\d+)", f).group(1)),
        )
        if not all_ckpts:
            continue
        latest = os.path.join(ckpt_dir, all_ckpts[-1])
        latest_step = int(re.search(r"step_(\d+)", all_ckpts[-1]).group(1))
        targets.append((latest, 100000,
                        f"D=100K seed={seed} step={latest_step} (stuck plateau)"))
        init = os.path.join(ckpt_dir, "step_0.pt")
        if os.path.exists(init):
            targets.append((init, 100000,
                            f"D=100K seed={seed} step=0 (random init)"))
        break

    if not targets:
        print("No checkpoints found.")
        return

    print(f"\nWill analyze {len(targets)} checkpoints:")
    for _, _, label in targets:
        print(f"  {label}")

    results = []
    for ckpt_path, D, label in targets:
        try:
            r = analyze_checkpoint(
                ckpt_path, D, args.K, label, device,
                n_iter=args.n_iter, batch_size=args.batch_size, seed=args.seed
            )
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    out_path = os.path.join(args.experiment_dir, "hessian_at_plateau.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'label':<55}  {'loss':>8}  {'lam_max':>12}  {'lam_min':>12}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:<55}  {r['train_loss']:>8.4f}  "
              f"{r['lambda_max']:>+12.4e}  {r['lambda_min']:>+12.4e}")

    print("\nKey question: is sign(lambda_min) the same for D=20K plateau and D=100K plateau?")
    plateau_20k = next((r for r in results if "D=20K" in r["label"] and "BEFORE" in r["label"]), None)
    plateau_100k = next((r for r in results if "D=100K" in r["label"] and "stuck" in r["label"]), None)
    if plateau_20k and plateau_100k:
        s20 = "POSITIVE" if plateau_20k["lambda_min"] > 0 else "NEGATIVE"
        s100 = "POSITIVE" if plateau_100k["lambda_min"] > 0 else "NEGATIVE"
        print(f"  D=20K  plateau (step 2500):  lambda_min = {plateau_20k['lambda_min']:+.4e}  ({s20})")
        print(f"  D=100K plateau (step 50000): lambda_min = {plateau_100k['lambda_min']:+.4e}  ({s100})")
        if s20 != s100:
            print("\n  ** SIGN FLIP. The loss landscape topology has phase-transitioned. **")
        else:
            print(f"\n  Same sign ({s20}). The wall is a slowdown, not a topological change.")


if __name__ == "__main__":
    main()
