#!/usr/bin/env python3
"""Push the D=100K stuck model along its Hessian escape eigenvector
and see whether it actually escapes.

Phases:
  1. Lanczos for v_min and v_max at the D=100K seed=0 step=50000 checkpoint
     (loss 2.87, lambda_min = -0.747 from prior run). Keep the eigenvectors.
  2. Eval-only sweep: perturb theta by alpha*v_min for alpha in a fixed list
     (both signs), report train_loss, pos1, z_shuffle_gap, group_accuracy.
  3. Training from perturbed starts: 4 alphas along v_min.
  4. Controls: -v_min @ alpha=2.0, v_max @ alpha=2.0, 3 random unit vectors
     @ alpha=2.0.

Output: results/push_escape/*

Important: the Hessian-vector product goes through torch.autograd.grad with
create_graph=True (double backward). MPS double-backward is historically
flaky, so Lanczos runs on CPU. The training phase can run on MPS since it
only needs single backward.
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
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap
from src.diagnostics import (
    compute_train_loss,
    compute_z_shuffle_gap,
    compute_group_accuracy,
)

# Reuse Lanczos machinery from the existing hessian script
from scripts.hessian_lanczos import (
    strip_compiled_prefix,
    get_param_list,
    hvp_full_dataset,
    make_linear_operator,
    compute_loss,
)


DEFAULT_CKPT = "results/phase1_d_sweep/runs/D100000_seed0/checkpoints/step_50000.pt"
DEFAULT_D = 100000
DEFAULT_K = 20

SWEEP_ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
TRAIN_ALPHAS_VMIN = [0.5, 1.0, 2.0, 5.0]
CONTROL_ALPHA = 2.0

LOG_K = math.log(DEFAULT_K)
HALF_LOG_K = 0.5 * LOG_K  # 1.4979


# -----------------------------------------------------------------------------
# Parameter plumbing
# -----------------------------------------------------------------------------

def flatten_params(model: Transformer) -> torch.Tensor:
    # Always walk all parameters so the ordering matches whether or not
    # requires_grad is on (Lanczos needs requires_grad=True; eval-only sweep
    # doesn't, but the flat layout must be the same).
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_params_from_flat(model: Transformer, flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view(p.shape))
            offset += n


def load_model(ckpt_path: str, device: torch.device,
               requires_grad: bool = False) -> Transformer:
    model = Transformer.from_config(ModelConfig()).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compiled_prefix(state)
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(requires_grad)
    return model


# -----------------------------------------------------------------------------
# Lanczos — return actual eigenvectors, not just eigenvalues
# -----------------------------------------------------------------------------

def lanczos_extreme(op, n_params: int, which: str, ncv: int, tol: float,
                    maxiter: int, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.RandomState(seed)
    v0 = rng.randn(n_params).astype(np.float64)
    vals, vecs = eigsh(op, k=1, which=which, tol=tol, maxiter=maxiter,
                       ncv=ncv, v0=v0, return_eigenvectors=True)
    return float(vals[0]), vecs[:, 0]


def compute_extreme_eigenpairs(ckpt_path: str, D: int, K: int,
                               device: torch.device,
                               ncv: int, tol: float, max_samples: int,
                               batch_size: int, maxiter: int,
                               cache_path: str | None,
                               recompute: bool) -> dict:
    """Returns dict with numeric eigenvalues and torch tensors for eigenvectors
    on CPU (float32, unit norm)."""
    if cache_path and os.path.exists(cache_path) and not recompute:
        print(f"[cache] loading v_min/v_max from {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        return {
            "lambda_max": float(blob["lambda_max"]),
            "lambda_min": float(blob["lambda_min"]),
            "v_max": blob["v_max"],
            "v_min": blob["v_min"],
            "n_params": int(blob["n_params"]),
        }

    print(f"[lanczos] loading {ckpt_path}", flush=True)
    model = load_model(ckpt_path, device, requires_grad=True)
    model.train()

    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)
    # Subsample matching earlier run's ncv=60, tol=1e-5, max_samples=4096
    if max_samples is not None and max_samples < dataset.D:
        rng_sub = np.random.RandomState(123)
        subset_idx = rng_sub.choice(dataset.D, size=max_samples, replace=False)
        data_for_hvp = dataset.data[subset_idx]
        print(f"  HVP subset: {max_samples} samples")
    else:
        data_for_hvp = dataset.data
        print(f"  HVP on full dataset: {dataset.D}")

    # Context loss on full dataset
    ctx_loss = compute_loss(model, dataset.data, device, batch_size=batch_size)
    print(f"  model loss at plateau ckpt: {ctx_loss:.4f}  (log K = {LOG_K:.4f})")

    op, n_calls = make_linear_operator(model, data_for_hvp, device,
                                        batch_size=batch_size, verbose=True)
    n_params = op.shape[0]

    print(f"\n[lanczos] lambda_max (ncv={ncv}, tol={tol})...", flush=True)
    n_calls[0] = 0
    t0 = time.time()
    lam_max, v_max = lanczos_extreme(op, n_params, "LA", ncv, tol, maxiter, 0)
    print(f"  lambda_max = {lam_max:+.6e}   "
          f"({n_calls[0]} HVPs, {time.time()-t0:.0f}s)", flush=True)

    print(f"\n[lanczos] lambda_min (ncv={ncv}, tol={tol})...", flush=True)
    n_calls[0] = 0
    t0 = time.time()
    lam_min, v_min = lanczos_extreme(op, n_params, "SA", ncv, tol, maxiter, 1)
    print(f"  lambda_min = {lam_min:+.6e}   "
          f"({n_calls[0]} HVPs, {time.time()-t0:.0f}s)", flush=True)

    # Unit-norm float32 torch tensors on CPU (we'll move to device as needed)
    v_max_t = torch.from_numpy((v_max / np.linalg.norm(v_max)).astype(np.float32))
    v_min_t = torch.from_numpy((v_min / np.linalg.norm(v_min)).astype(np.float32))

    result = {
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "v_max": v_max_t,
        "v_min": v_min_t,
        "n_params": n_params,
        "plateau_loss": ctx_loss,
    }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "lambda_max": result["lambda_max"],
            "lambda_min": result["lambda_min"],
            "v_max": result["v_max"],
            "v_min": result["v_min"],
            "n_params": result["n_params"],
            "plateau_loss": result["plateau_loss"],
        }, cache_path)
        print(f"[cache] saved to {cache_path}")

    del model
    return result


# -----------------------------------------------------------------------------
# Eval-only sweep (Phase 2)
# -----------------------------------------------------------------------------

@torch.no_grad()
def eval_perturbed(model: Transformer, theta0: torch.Tensor,
                    direction: torch.Tensor, alpha: float,
                    dataset: SurjectiveMap, device: torch.device,
                    full_data: torch.Tensor,
                    do_group_acc: bool = True) -> dict:
    """In-place perturb, run all diagnostics, restore."""
    set_params_from_flat(model, theta0 + alpha * direction)
    model.eval()

    loss_metrics = compute_train_loss(model, full_data, device)
    z_metrics = compute_z_shuffle_gap(model, full_data, device, batch_size=1024)
    result = {
        "alpha": alpha,
        "train_loss": loss_metrics["train_loss"],
        "train_loss_pos1": loss_metrics["train_loss_pos1"],
        "z_shuffle_gap": z_metrics["z_shuffle_gap"],
    }
    if do_group_acc:
        acc_metrics = compute_group_accuracy(model, dataset, device, n_groups=200)
        result["group_acc_frac_80"] = acc_metrics["group_accuracy_frac_80"]
        result["group_acc_mean"] = acc_metrics["group_accuracy_mean"]

    set_params_from_flat(model, theta0)
    return result


def run_sweep_phase(ckpt_path: str, v_min: torch.Tensor, v_max: torch.Tensor,
                    device: torch.device, out_dir: str) -> list[dict]:
    print("\n" + "=" * 78)
    print("PHASE 2: Eval-only sweep along v_min (both signs)")
    print("=" * 78)

    dataset = SurjectiveMap(K=DEFAULT_K, n_b=DEFAULT_D // DEFAULT_K, seed=DEFAULT_D)
    full_data = dataset.get_full()
    model = load_model(ckpt_path, device, requires_grad=False)
    theta0 = flatten_params(model).clone()
    base_loss = compute_train_loss(model, full_data, device)["train_loss"]
    print(f"base loss (alpha=0): {base_loss:.4f}")

    v_min_dev = v_min.to(device)
    v_max_dev = v_max.to(device)

    print(f"\n{'alpha':>8}  {'loss':>8}  {'pos1':>8}  {'z_gap':>8}  {'acc80':>8}")
    print("-" * 52)

    results = []
    # alpha=0 baseline
    r0 = eval_perturbed(model, theta0, v_min_dev, 0.0, dataset, device, full_data,
                        do_group_acc=True)
    r0["direction"] = "baseline"
    r0["sign"] = 0
    results.append(r0)
    print(f"{'0.0':>8}  {r0['train_loss']:>8.4f}  {r0['train_loss_pos1']:>8.4f}  "
          f"{r0['z_shuffle_gap']:>8.4f}  {r0['group_acc_frac_80']:>8.4f}")

    # Positive and negative v_min sweep
    for sign, label in [(+1, "+v_min"), (-1, "-v_min")]:
        print(f"\n-- {label} --")
        for a in SWEEP_ALPHAS:
            r = eval_perturbed(model, theta0, v_min_dev, sign * a,
                                dataset, device, full_data, do_group_acc=True)
            r["direction"] = label
            r["sign"] = sign
            results.append(r)
            print(f"{sign*a:>+8.3f}  {r['train_loss']:>8.4f}  "
                  f"{r['train_loss_pos1']:>8.4f}  {r['z_shuffle_gap']:>8.4f}  "
                  f"{r['group_acc_frac_80']:>8.4f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_dir}/sweep.json")
    return results


# -----------------------------------------------------------------------------
# Training from perturbed start (Phases 3 & 4)
# -----------------------------------------------------------------------------

def get_lr(step, warmup, base):
    return base * step / max(1, warmup) if step < warmup else base


def train_from_perturbed(src_ckpt: str, v_min: torch.Tensor,
                          direction: torch.Tensor, alpha: float,
                          label: str, out_dir: str,
                          device: torch.device, max_steps: int,
                          eval_every: int, early_stop_thresh: float,
                          early_stop_patience: int) -> dict:
    print(f"\n=== TRAINING: {label}  alpha={alpha}  ===")
    run_dir = os.path.join(out_dir, "runs", label)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    dataset = SurjectiveMap(K=DEFAULT_K, n_b=DEFAULT_D // DEFAULT_K, seed=DEFAULT_D)
    full_data = dataset.get_full()

    # Fresh model, load src weights, then perturb
    model = load_model(src_ckpt, device, requires_grad=True)
    theta0 = flatten_params(model).clone()
    direction_dev = direction.to(device)
    set_params_from_flat(model, theta0 + alpha * direction_dev)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        weight_decay=0.01, eps=1e-8,
    )
    batch_rng = np.random.RandomState(42)  # fixed across all training runs

    t0 = time.time()
    below_streak = 0
    tau_total = None
    tau_pos1 = None
    escaped = False
    early_stopped_at = None

    @torch.no_grad()
    def run_eval(step):
        model.eval()
        m = {"step": step}
        loss_metrics = compute_train_loss(model, full_data, device)
        m.update(loss_metrics)
        z_metrics = compute_z_shuffle_gap(model, full_data, device, batch_size=1024)
        m.update(z_metrics)
        m["wall_time_seconds"] = time.time() - t0
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        model.train()
        return m

    # Initial eval (step 0)
    m0 = run_eval(0)
    print(f"[{label:>14}] step 0    loss={m0['train_loss']:.4f}  "
          f"pos1={m0.get('train_loss_pos1', -1):.4f}")

    for step in range(1, max_steps + 1):
        lr = get_lr(step, 500, 1e-3)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = dataset.get_batch(128, batch_rng).to(device)
        loss, _ = model(batch, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == max_steps:
            m = run_eval(step)
            if tau_total is None and m["train_loss"] < HALF_LOG_K:
                tau_total = step
                escaped = True
            if tau_pos1 is None and m.get("train_loss_pos1", 99) < HALF_LOG_K:
                tau_pos1 = step
            # early stop: loss < thresh for patience consecutive evals
            if m["train_loss"] < early_stop_thresh:
                below_streak += 1
            else:
                below_streak = 0
            if below_streak >= early_stop_patience:
                early_stopped_at = step
                print(f"[{label:>14}] early stop at step {step}  "
                      f"(train_loss={m['train_loss']:.4f})")
                break

            if step % (eval_every * 5) == 0:
                print(f"[{label:>14}] step {step:>5} loss={m['train_loss']:.4f}  "
                      f"pos1={m.get('train_loss_pos1', -1):.4f}  "
                      f"z_gap={m['z_shuffle_gap']:.4f}")

    final = m
    wall = time.time() - t0
    print(f"[{label:>14}] done {wall:.0f}s  "
          f"final_loss={final['train_loss']:.4f}  escaped={escaped}  "
          f"tau_total={tau_total}  tau_pos1={tau_pos1}")

    return {
        "label": label,
        "alpha": alpha,
        "final_loss": final["train_loss"],
        "final_pos1": final.get("train_loss_pos1"),
        "final_z_gap": final["z_shuffle_gap"],
        "escaped": bool(escaped),
        "tau_total": tau_total,
        "tau_pos1": tau_pos1,
        "early_stopped_at": early_stopped_at,
        "wall_time_seconds": wall,
        "metrics_path": metrics_path,
    }


# -----------------------------------------------------------------------------
# Plot and table
# -----------------------------------------------------------------------------

def plot_training(runs: list[dict], out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    # Color by condition type
    cmap = {
        "v_min": "#1f77b4",      # blue
        "-v_min": "#17becf",     # cyan
        "v_max": "#d62728",      # red
        "random": "#7f7f7f",     # gray
    }
    for r in runs:
        label = r["label"]
        if label.startswith("v_min_alpha"):
            color, kind = cmap["v_min"], "v_min"
        elif label.startswith("neg_v_min"):
            color, kind = cmap["-v_min"], "-v_min"
        elif label.startswith("v_max"):
            color, kind = cmap["v_max"], "v_max"
        elif label.startswith("random"):
            color, kind = cmap["random"], "random"
        else:
            color = "black"; kind = label

        with open(r["metrics_path"]) as f:
            m = [json.loads(l) for l in f if l.strip()]
        steps = [x["step"] for x in m]
        losses = [x["train_loss"] for x in m]
        ax.plot(steps, losses, "-", color=color, lw=1.5, alpha=0.85,
                label=f"{label} (α={r['alpha']})")

    ax.axhline(LOG_K, color="black", ls=":", lw=0.8,
               label=f"log K = {LOG_K:.3f}")
    ax.axhline(HALF_LOG_K, color="red", ls="--", lw=0.8,
               label=f"0.5 log K = {HALF_LOG_K:.3f}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("train_loss")
    ax.set_yscale("log")
    ax.set_title("Training from perturbed-start: does D=100K escape?")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_summary(runs: list[dict]):
    print("\n" + "=" * 88)
    print(f"{'condition':>14}  {'alpha':>6}  {'escaped':>8}  "
          f"{'tau_total':>10}  {'tau_pos1':>10}  {'final_loss':>12}")
    print("-" * 88)
    for r in runs:
        esc = "YES" if r["escaped"] else "no"
        tt = str(r["tau_total"]) if r["tau_total"] is not None else "-"
        tp = str(r["tau_pos1"]) if r["tau_pos1"] is not None else "-"
        print(f"{r['label']:>14}  {r['alpha']:>6.2f}  {esc:>8}  "
              f"{tt:>10}  {tp:>10}  {r['final_loss']:>12.4f}")
    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--D", type=int, default=DEFAULT_D)
    ap.add_argument("--K", type=int, default=DEFAULT_K)

    # Lanczos
    ap.add_argument("--lanczos-device", default="cpu")
    ap.add_argument("--ncv", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-samples", type=int, default=4096)
    ap.add_argument("--hvp-batch-size", type=int, default=2000)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--cache-path", default="results/push_escape/eigenpairs.pt")
    ap.add_argument("--recompute", action="store_true",
                    help="Ignore cached eigenpairs and recompute")

    # Training
    ap.add_argument("--train-device", default=None,
                    help="Default: mps if available, else cpu")
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--early-stop-thresh", type=float, default=0.05)
    ap.add_argument("--early-stop-patience", type=int, default=3,
                    help="Consecutive evals below thresh before stopping. "
                         "With eval_every=200, patience=3 ≈ 600 training steps.")

    ap.add_argument("--skip-phase", type=int, nargs="*", default=[],
                    help="Skip phases by number: 2=sweep, 3=train_vmin, 4=controls")
    ap.add_argument("--out-dir", default="results/push_escape")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    lanczos_device = torch.device(args.lanczos_device)
    train_device = torch.device(args.train_device) if args.train_device else torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Lanczos device: {lanczos_device}")
    print(f"Training device: {train_device}")

    # ================= Phase 1: eigenpairs =================
    eig = compute_extreme_eigenpairs(
        args.ckpt, args.D, args.K, lanczos_device,
        ncv=args.ncv, tol=args.tol, max_samples=args.max_samples,
        batch_size=args.hvp_batch_size, maxiter=args.maxiter,
        cache_path=args.cache_path, recompute=args.recompute,
    )
    print(f"\nv_min ||·|| = {eig['v_min'].norm().item():.6f}  "
          f"v_max ||·|| = {eig['v_max'].norm().item():.6f}")
    print(f"lambda_min = {eig['lambda_min']:+.6e}  "
          f"lambda_max = {eig['lambda_max']:+.6e}")

    v_min = eig["v_min"]
    v_max = eig["v_max"]

    # ================= Phase 2: eval-only sweep =================
    if 2 not in args.skip_phase:
        run_sweep_phase(args.ckpt, v_min, v_max, train_device, args.out_dir)

    # ================= Phase 3+4: training =================
    training_runs = []

    if 3 not in args.skip_phase:
        for a in TRAIN_ALPHAS_VMIN:
            r = train_from_perturbed(
                args.ckpt, v_min, v_min, a,
                label=f"v_min_alpha_{a:g}", out_dir=args.out_dir,
                device=train_device, max_steps=args.max_steps,
                eval_every=args.eval_every,
                early_stop_thresh=args.early_stop_thresh,
                early_stop_patience=args.early_stop_patience,
            )
            training_runs.append(r)

    if 4 not in args.skip_phase:
        # -v_min at alpha=2
        r = train_from_perturbed(
            args.ckpt, v_min, v_min, -CONTROL_ALPHA,
            label=f"neg_v_min_alpha_{CONTROL_ALPHA:g}", out_dir=args.out_dir,
            device=train_device, max_steps=args.max_steps,
            eval_every=args.eval_every,
            early_stop_thresh=args.early_stop_thresh,
            early_stop_patience=args.early_stop_patience,
        )
        training_runs.append(r)
        # v_max at alpha=2
        r = train_from_perturbed(
            args.ckpt, v_min, v_max, CONTROL_ALPHA,
            label=f"v_max_alpha_{CONTROL_ALPHA:g}", out_dir=args.out_dir,
            device=train_device, max_steps=args.max_steps,
            eval_every=args.eval_every,
            early_stop_thresh=args.early_stop_thresh,
            early_stop_patience=args.early_stop_patience,
        )
        training_runs.append(r)
        # random unit vectors at alpha=2
        n_params = eig["n_params"]
        for ri in range(3):
            rng = np.random.RandomState(5000 + ri)
            rv = rng.randn(n_params).astype(np.float32)
            rv /= np.linalg.norm(rv)
            rv_t = torch.from_numpy(rv)
            r = train_from_perturbed(
                args.ckpt, v_min, rv_t, CONTROL_ALPHA,
                label=f"random_{ri+1}_alpha_{CONTROL_ALPHA:g}",
                out_dir=args.out_dir,
                device=train_device, max_steps=args.max_steps,
                eval_every=args.eval_every,
                early_stop_thresh=args.early_stop_thresh,
                early_stop_patience=args.early_stop_patience,
            )
            training_runs.append(r)

    # ================= Outputs =================
    if training_runs:
        print_summary(training_runs)
        plot_training(training_runs, os.path.join(args.out_dir,
                                                  "push_escape_training.png"))
        with open(os.path.join(args.out_dir, "training_summary.json"), "w") as f:
            json.dump(training_runs, f, indent=2)
        print(f"saved: {args.out_dir}/push_escape_training.png")
        print(f"saved: {args.out_dir}/training_summary.json")


if __name__ == "__main__":
    main()
