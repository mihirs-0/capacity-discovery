#!/usr/bin/env python3
"""Properly compute extreme Hessian eigenvalues via Lanczos.

Uses scipy.sparse.linalg.eigsh with a LinearOperator that wraps
HVP. HVP is averaged across mini-batches that exactly tile the
full dataset, giving the true full-Hessian eigenvalues (not a
subsample's).

Lanczos is the right tool for near-zero eigenvalues. Power
iteration converges as (|lambda_2|/|lambda_1|)^k which is
useless when both are tiny.

Usage:
  python scripts/hessian_lanczos.py --device cpu --tol 1e-4
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh, LinearOperator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap


def strip_compiled_prefix(state_dict):
    new = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new[k[len("_orig_mod."):]] = v
        else:
            new[k] = v
    return new


def get_param_list(model):
    return [p for p in model.parameters() if p.requires_grad]


def flat_to_params(flat, params):
    """Unflatten a 1D tensor into a list of parameter-shaped tensors."""
    out = []
    offset = 0
    for p in params:
        n = p.numel()
        out.append(flat[offset:offset + n].view(p.shape))
        offset += n
    return out


def params_to_flat(plist):
    return torch.cat([p.reshape(-1) for p in plist])


def hvp_full_dataset(model, dataset_tensor, device, vector_flat,
                     batch_size=2000, params=None, n_loss_pos=5):
    """Compute (H_full) @ v by averaging H_batch @ v across mini-batches.

    Cross-entropy with reduction='mean' divides by (batch_size * n_loss_pos).
    For correct averaging across batches of equal size, we just average the
    HVPs (each batch contributes its loss-mean Hessian).

    Requires the dataset size to be divisible by batch_size — caller
    should pad or trim. We trim to the largest multiple of batch_size
    so that all batches are equal-size.
    """
    D = dataset_tensor.shape[0]
    n_batches = D // batch_size
    usable = n_batches * batch_size
    data_used = dataset_tensor[:usable]

    if params is None:
        params = get_param_list(model)

    vector = flat_to_params(vector_flat, params)

    accumulated = [torch.zeros_like(p) for p in params]

    for start in range(0, usable, batch_size):
        batch = data_used[start:start + batch_size].to(device)

        model.zero_grad()
        loss, _ = model(batch, batch)

        grads = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((g * v).sum() for g, v in zip(grads, vector))
        hvp = torch.autograd.grad(gv, params)

        for acc, h in zip(accumulated, hvp):
            acc += h.detach()

        del loss, grads, gv, hvp

    # Average over batches
    accumulated = [a / n_batches for a in accumulated]
    return params_to_flat(accumulated)


def make_linear_operator(model, dataset_tensor, device, batch_size, dtype=np.float64,
                         verbose=True):
    """Build a scipy LinearOperator that computes H_full @ v."""
    params = get_param_list(model)
    n_params = sum(p.numel() for p in params)
    print(f"  n_params: {n_params:,}", flush=True)

    n_calls = [0]
    t_start = [time.time()]

    def matvec(v_np):
        n_calls[0] += 1
        t0 = time.time()
        v_t = torch.from_numpy(v_np.astype(np.float32)).to(device)
        result = hvp_full_dataset(model, dataset_tensor, device, v_t,
                                  batch_size=batch_size, params=params)
        elapsed = time.time() - t0
        if verbose:
            print(f"    HVP #{n_calls[0]}: {elapsed:.1f}s "
                  f"(total {time.time() - t_start[0]:.0f}s)", flush=True)
        return result.cpu().numpy().astype(dtype)

    op = LinearOperator(shape=(n_params, n_params), matvec=matvec, dtype=dtype)
    return op, n_calls


def compute_loss(model, dataset_tensor, device, batch_size=2000):
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


def analyze_checkpoint(ckpt_path, D, K, label, device, batch_size, tol, maxiter,
                       ncv, seed, max_samples=None):
    print(f"\n=== {label} ===", flush=True)
    print(f"  Checkpoint: {ckpt_path}", flush=True)

    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
    print(f"  Full dataset: D={dataset.D}, K={K}", flush=True)
    if max_samples is not None and max_samples < dataset.D:
        rng_sub = np.random.RandomState(123)
        subset_idx = rng_sub.choice(dataset.D, size=max_samples, replace=False)
        data_for_hvp = dataset.data[subset_idx]
        print(f"  Using HVP subset of {max_samples} samples (16x previous run)", flush=True)
    else:
        data_for_hvp = dataset.data
        print(f"  Using full dataset for HVP", flush=True)

    # Build model
    model = Transformer.from_config(ModelConfig()).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compiled_prefix(state)
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()  # ensure dropout etc. consistent (none here, but be safe)

    # Loss for context (always on full dataset)
    train_loss = compute_loss(model, dataset.data, device, batch_size=batch_size)
    print(f"  Train loss: {train_loss:.4f}  (log K = {np.log(K):.4f})", flush=True)

    # Build operator on the chosen data
    op, n_calls = make_linear_operator(model, data_for_hvp, device, batch_size)

    # Largest algebraic eigenvalue
    print(f"  Lanczos for lambda_max (tol={tol}, maxiter={maxiter}, ncv={ncv})...", flush=True)
    n_calls[0] = 0
    t0 = time.time()
    lam_max = float("nan")
    lam_max_converged = False
    lam_max_residual = float("nan")
    try:
        from scipy.sparse.linalg import ArpackNoConvergence
        rng = np.random.RandomState(seed)
        v0 = rng.randn(op.shape[0]).astype(np.float64)
        try:
            vals_max, vecs_max = eigsh(op, k=1, which="LA", tol=tol, maxiter=maxiter,
                                       ncv=ncv, v0=v0, return_eigenvectors=True)
            lam_max = float(vals_max[0])
            lam_max_converged = True
        except ArpackNoConvergence as e:
            print(f"    ARPACK did NOT converge: {e}", flush=True)
            if len(e.eigenvalues) > 0:
                lam_max = float(e.eigenvalues[0])
                vecs_max = e.eigenvectors
                print(f"    (Best so far: {lam_max:+.6e})", flush=True)
            else:
                vecs_max = None

        # Independent residual check: ||H v - lam v||
        if vecs_max is not None and not np.isnan(lam_max):
            v = vecs_max[:, 0]
            hv = op.matvec(v)
            res_vec = hv - lam_max * v
            lam_max_residual = float(np.linalg.norm(res_vec))

        marker = "CONVERGED" if lam_max_converged else "MAXITER"
        ratio = lam_max_residual / abs(lam_max) if abs(lam_max) > 0 else float("nan")
        print(f"    lambda_max = {lam_max:>+12.6e}  "
              f"residual = {lam_max_residual:.3e}  ratio = {ratio:.3f}  "
              f"[{marker}]  ({n_calls[0]} HVP calls, {time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"    FAILED: {e}", flush=True)

    # Smallest algebraic eigenvalue
    print(f"  Lanczos for lambda_min (tol={tol}, maxiter={maxiter}, ncv={ncv})...", flush=True)
    n_calls[0] = 0
    t0 = time.time()
    lam_min = float("nan")
    lam_min_converged = False
    lam_min_residual = float("nan")
    try:
        from scipy.sparse.linalg import ArpackNoConvergence
        rng = np.random.RandomState(seed + 1)
        v0 = rng.randn(op.shape[0]).astype(np.float64)
        try:
            vals_min, vecs_min = eigsh(op, k=1, which="SA", tol=tol, maxiter=maxiter,
                                       ncv=ncv, v0=v0, return_eigenvectors=True)
            lam_min = float(vals_min[0])
            lam_min_converged = True
        except ArpackNoConvergence as e:
            print(f"    ARPACK did NOT converge: {e}", flush=True)
            if len(e.eigenvalues) > 0:
                lam_min = float(e.eigenvalues[0])
                vecs_min = e.eigenvectors
                print(f"    (Best so far: {lam_min:+.6e})", flush=True)
            else:
                vecs_min = None

        if vecs_min is not None and not np.isnan(lam_min):
            v = vecs_min[:, 0]
            hv = op.matvec(v)
            res_vec = hv - lam_min * v
            lam_min_residual = float(np.linalg.norm(res_vec))

        marker = "CONVERGED" if lam_min_converged else "MAXITER"
        ratio = lam_min_residual / abs(lam_min) if abs(lam_min) > 0 else float("nan")
        print(f"    lambda_min = {lam_min:>+12.6e}  "
              f"residual = {lam_min_residual:.3e}  ratio = {ratio:.3f}  "
              f"[{marker}]  ({n_calls[0]} HVP calls, {time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"    FAILED: {e}", flush=True)

    sign = "POSITIVE (true local minimum)" if lam_min > 0 else \
           "NEGATIVE (saddle, escape direction exists)"
    print(f"    sign(lambda_min) = {sign}", flush=True)

    return {
        "label": label,
        "ckpt_path": ckpt_path,
        "D": D,
        "train_loss": train_loss,
        "log_K": float(np.log(K)),
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "lambda_max_residual": lam_max_residual,
        "lambda_min_residual": lam_min_residual,
        "lambda_max_converged": lam_max_converged,
        "lambda_min_converged": lam_min_converged,
        "ncv": ncv,
        "tol": tol,
        "max_samples": max_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, default="results/phase1_d_sweep")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="HVP mini-batch size (must divide D)")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--ncv", type=int, default=20,
                        help="Lanczos vector count (more = better but slower)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples used for HVP (default: full dataset)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one of: d100k_stuck, d20k_plateau, d20k_converged, d100k_init")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"tol: {args.tol}, maxiter: {args.maxiter}, ncv: {args.ncv}")
    print(f"HVP batch size: {args.batch_size}")

    runs_dir = os.path.join(args.experiment_dir, "runs")

    targets = []

    # D=100K stuck plateau (the headline number)
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
        targets.append(("d100k_stuck", latest, 100000,
                        f"D=100K seed={seed} step={latest_step} (stuck plateau)"))
        break

    # D=20K plateau (BEFORE snap) — sanity check (we already trust this one)
    for seed in range(5):
        ckpt_dir = os.path.join(runs_dir, f"D20000_seed{seed}", "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        plateau = os.path.join(ckpt_dir, "step_2500.pt")
        if os.path.exists(plateau):
            targets.append(("d20k_plateau", plateau, 20000,
                            f"D=20K seed={seed} step=2500 (plateau, BEFORE snap)"))
        break

    # D=20K converged
    for seed in range(5):
        ckpt_dir = os.path.join(runs_dir, f"D20000_seed{seed}", "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        all_ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda f: int(re.search(r"step_(\d+)", f).group(1)),
        )
        if all_ckpts:
            latest = os.path.join(ckpt_dir, all_ckpts[-1])
            latest_step = int(re.search(r"step_(\d+)", all_ckpts[-1]).group(1))
            targets.append(("d20k_converged", latest, 20000,
                            f"D=20K seed={seed} step={latest_step} (post-snap, converged)"))
        break

    if args.only:
        targets = [t for t in targets if t[0] == args.only]

    if not targets:
        print("No checkpoints found.")
        return

    print(f"\nWill analyze {len(targets)} checkpoints:")
    for tag, _, _, label in targets:
        print(f"  [{tag}] {label}")

    results = []
    for tag, ckpt_path, D, label in targets:
        try:
            r = analyze_checkpoint(
                ckpt_path, D, args.K, label, device,
                batch_size=args.batch_size, tol=args.tol,
                maxiter=args.maxiter, ncv=args.ncv, seed=args.seed,
                max_samples=args.max_samples,
            )
            r["tag"] = tag
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    out_path = os.path.join(args.experiment_dir, "hessian_lanczos.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    print(f"\n{'=' * 90}")
    print("LANCZOS RESULTS")
    print(f"{'=' * 90}")
    print(f"{'label':<55}  {'loss':>8}  {'lam_max':>14}  {'lam_min':>14}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:<55}  {r['train_loss']:>8.4f}  "
              f"{r['lambda_max']:>+14.4e}  {r['lambda_min']:>+14.4e}")


if __name__ == "__main__":
    main()
