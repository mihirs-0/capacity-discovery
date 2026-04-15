#!/usr/bin/env python3
"""Decompose per-example gradients into marginal (group-mean) and
conditional (per-example deviation) components through the D=10K
plateau → escape trajectory.

For each group b (K=20 examples sharing the same B string):
  g_j            = per-example gradient for example j
  group_mean_b   = (1/K) Σ_{j∈b} g_j            ("marginals" component)
  cond_j         = g_j − group_mean_b             ("conditionals" component)

Key identity: E[‖g‖²] = E[‖group_mean‖²] + E[‖cond‖²]
  (cross terms vanish because Σ_{j∈b} cond_j = 0 by construction)

Also:
  batch_gradient = mean over groups of group_mean_b
  (conditionals sum to zero within each group, so they vanish from the
  batch gradient entirely — the batch gradient IS the marginals signal)

Tracks marginals_rms and conditionals_rms over training steps to see
whether the 5× RMS growth during the plateau comes from marginals
(model learning the B-group structure) or conditionals (model learning
z-specific routing) or both.
"""

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
import matplotlib.pyplot as plt

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap


TARGET_STEPS = [100, 300, 500, 700, 900, 1000, 1100, 1200, 1300, 1500,
                1800, 2500]
CKPT_DIR = "results/basin_plateau_retrain/runs/D10000_seed3/checkpoints"
D = 10000
K = 20
N_GROUPS = 100    # sample 100 out of 500 groups = 2000 examples
GROUP_SEED = 77
OUT_DIR = "results/gradient_trajectory"
HALF_LOG_K = 0.5 * math.log(K)


def strip_prefix(s):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in s.items()}


def load_model(ckpt, device):
    m = Transformer.from_config(ModelConfig()).to(device)
    m.load_state_dict(strip_prefix(
        torch.load(ckpt, map_location=device, weights_only=True)))
    for p in m.parameters():
        p.requires_grad_(True)
    m.train()
    return m


@torch.no_grad()
def eval_loss(model, data, device, bs=2000):
    model.eval()
    t, n = 0.0, 0
    for s in range(0, data.shape[0], bs):
        b = data[s:s + bs].to(device)
        l, _ = model(b, b)
        t += l.item() * b.shape[0]; n += b.shape[0]
    model.train()
    return t / n


@torch.no_grad()
def compute_marginal_loss(model, group_examples, device):
    """Loss when using group-averaged logits (marginal prediction)."""
    n_g = group_examples.shape[0]  # n_groups
    k = group_examples.shape[1]    # K
    flat = group_examples.view(-1, 16).to(device)
    _, logits = model(flat)  # (n_g*K, 16, V)
    V = logits.size(-1)
    logits = logits.view(n_g, k, 16, V)
    avg_logits = logits.mean(dim=1, keepdim=True).expand_as(logits)
    loss_logits = avg_logits[:, :, 10:15].reshape(-1, V)
    loss_targets = group_examples.to(device)[:, :, 11:16].reshape(-1)
    return F.cross_entropy(loss_logits, loss_targets).item()


@torch.no_grad()
def per_example_loss_stats(model, flat_examples, device):
    """Per-example loss mean and std across examples."""
    B = flat_examples.shape[0]
    _, logits = model(flat_examples.to(device))
    V = logits.size(-1)
    per_token = F.cross_entropy(
        logits[:, 10:15].reshape(-1, V),
        flat_examples.to(device)[:, 11:16].reshape(-1),
        reduction="none",
    ).reshape(B, 5)
    per_ex = per_token.mean(dim=1).cpu().numpy()
    return float(per_ex.mean()), float(per_ex.std())


def compute_gradient(model, example, device):
    """Single-example gradient as a flat CPU tensor."""
    for p in model.parameters():
        p.grad = None
    x = example.unsqueeze(0).to(device)
    loss, _ = model(x, x)
    loss.backward()
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()]).cpu()


def analyze_checkpoint(step, ckpt, dataset, group_indices, device):
    model = load_model(ckpt, device)
    full_data = dataset.data

    # Full-dataset loss
    full_loss = eval_loss(model, full_data, device)

    # Build group-structured sample: (N_GROUPS, K, 16)
    group_examples = torch.stack([
        dataset.get_group_examples(int(gi)) for gi in group_indices
    ])  # (N_GROUPS, K, 16)
    flat_examples = group_examples.view(-1, 16)  # (N_GROUPS*K, 16)

    # Per-example loss stats on the sample
    loss_mean, loss_std = per_example_loss_stats(model, flat_examples, device)

    # Marginal loss (group-averaged logits)
    marginal_loss = compute_marginal_loss(model, group_examples, device)

    # Per-example gradients → decompose per group
    n_g = len(group_indices)
    marginals_sq_norms = np.zeros(n_g)
    conditionals_sq_norms_all = []
    batch_grad_accum = None  # running sum of group means

    for bi in range(n_g):
        # Compute K per-example gradients for this group
        grads = []
        for ki in range(K):
            idx = bi * K + ki
            g = compute_gradient(model, flat_examples[idx], device)
            grads.append(g)
        grads_t = torch.stack(grads)  # (K, n_params)

        # Group mean
        gm = grads_t.mean(dim=0)  # (n_params,)
        marginals_sq_norms[bi] = float((gm * gm).sum().item())

        # Conditional deviations
        for ki in range(K):
            cond = grads[ki] - gm
            conditionals_sq_norms_all.append(float((cond * cond).sum().item()))

        # Accumulate for batch gradient
        if batch_grad_accum is None:
            batch_grad_accum = gm.clone()
        else:
            batch_grad_accum += gm

    conditionals_sq_norms_all = np.array(conditionals_sq_norms_all)

    # Statistics
    marginals_rms = float(np.sqrt(marginals_sq_norms.mean()))
    conditionals_rms = float(np.sqrt(conditionals_sq_norms_all.mean()))
    ratio = marginals_rms / conditionals_rms if conditionals_rms > 0 else float("inf")

    batch_grad = batch_grad_accum / n_g  # mean of group means
    marginals_batch = float(batch_grad.norm().item())

    # Total per-example RMS (for consistency check)
    total_sq = marginals_sq_norms.mean() + conditionals_sq_norms_all.mean()
    total_rms = float(np.sqrt(total_sq))

    del model
    return {
        "step": step,
        "loss": full_loss,
        "loss_sample_mean": loss_mean,
        "loss_sample_std": loss_std,
        "marginal_loss": marginal_loss,
        "marginals_rms": marginals_rms,
        "conditionals_rms": conditionals_rms,
        "ratio_m_over_c": ratio,
        "marginals_batch_norm": marginals_batch,
        "total_rms": total_rms,
        "n_groups": n_g,
        "K": K,
    }


def phase_label(loss):
    if loss > 2.3: return "plateau"
    if loss > 0.5: return "escape"
    return "converged"


def main():
    device = torch.device("cpu")
    dataset = SurjectiveMap(K=K, n_b=D // K, seed=D)

    # Verify group structure
    for gi in [0, 7, 42]:
        ex = dataset.get_group_examples(gi)
        b_tokens = ex[:, 1:7]
        assert (b_tokens[0:1] == b_tokens).all(), \
            f"group {gi}: B tokens not identical across K examples!"
    print("Group structure verified: B tokens identical within groups ✓")

    # Fixed group sample
    rng = np.random.RandomState(GROUP_SEED)
    group_indices = rng.choice(dataset.n_b, size=N_GROUPS, replace=False)
    print(f"Sampled {N_GROUPS} groups ({N_GROUPS*K} examples)")

    results = []
    print(f"\n{'step':>5}  {'loss':>7}  {'loss_std':>8}  {'marg_rms':>10}  "
          f"{'cond_rms':>10}  {'ratio':>7}  {'marg_batch':>11}  "
          f"{'marg_loss':>10}  {'phase':>9}  {'wall':>5}")
    print("-" * 105)

    for step in TARGET_STEPS:
        ckpt = os.path.join(CKPT_DIR, f"step_{step}.pt")
        if not os.path.exists(ckpt):
            print(f"{step:>5}  [checkpoint not found]"); continue
        t0 = time.time()
        r = analyze_checkpoint(step, ckpt, dataset, group_indices, device)
        dt = time.time() - t0
        r["phase"] = phase_label(r["loss"])
        r["wall_seconds"] = dt
        results.append(r)
        print(f"{step:>5}  {r['loss']:>7.4f}  {r['loss_sample_std']:>8.4f}  "
              f"{r['marginals_rms']:>10.4e}  {r['conditionals_rms']:>10.4e}  "
              f"{r['ratio_m_over_c']:>7.3f}  {r['marginals_batch_norm']:>11.4e}  "
              f"{r['marginal_loss']:>10.4f}  {r['phase']:>9}  {dt:>4.0f}s",
              flush=True)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "gradient_decomposition_D10K.json")
    with open(json_path, "w") as f:
        json.dump({"D": D, "K": K, "n_groups": N_GROUPS,
                   "group_seed": GROUP_SEED, "results": results}, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ---- Plot: marginals_rms vs conditionals_rms vs step, loss on right ----
    steps = np.array([r["step"] for r in results])
    losses = np.array([r["loss"] for r in results])
    m_rms = np.array([r["marginals_rms"] for r in results])
    c_rms = np.array([r["conditionals_rms"] for r in results])
    ratios = np.array([r["ratio_m_over_c"] for r in results])
    marg_loss = np.array([r["marginal_loss"] for r in results])
    loss_std = np.array([r["loss_sample_std"] for r in results])

    fig, ax = plt.subplots(figsize=(10, 6))
    c_m = "#1f77b4"
    c_c = "#ff7f0e"
    c_loss = "#d62728"

    ax.plot(steps, m_rms, "o-", color=c_m, lw=2, ms=7, label="marginals RMS")
    ax.plot(steps, c_rms, "s-", color=c_c, lw=2, ms=6, label="conditionals RMS")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient component RMS")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(steps, losses, "d-", color=c_loss, lw=1.3, ms=4, alpha=0.7,
             label="train loss")
    ax2.axhline(HALF_LOG_K, color=c_loss, ls=":", lw=0.8, alpha=0.5)
    ax2.set_ylabel("train loss", color=c_loss)
    ax2.tick_params(axis="y", labelcolor=c_loss)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

    # Mark crossover if it exists
    cross_idx = None
    for i in range(1, len(steps)):
        if ratios[i-1] > 1 and ratios[i] < 1:
            cross_idx = i
            break
    if cross_idx:
        ax.axvline(steps[cross_idx], color="gray", ls="--", lw=1, alpha=0.5)
        ax.text(steps[cross_idx], ax.get_ylim()[1]*0.9,
                f" crossover @ step {steps[cross_idx]}", fontsize=8, color="gray")

    ax.set_title("D=10K: marginals vs conditionals gradient decomposition")
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "gradient_decomposition_D10K.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()
