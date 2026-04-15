#!/usr/bin/env python3
"""PHASE A: Unbundle muP — which component enables circuit discovery?

5 configurations isolating muP's three components:
  1. SP_muP_lr:      SP init + SP attention + layer-wise lr (muP lr only)
  2. muP_init_only:  muP init + SP attention + uniform lr (muP init only)
  3. muP_attn_only:  SP init + muP attention + uniform lr (muP attn only)
  4. muP_init_attn:  muP init + muP attention + uniform lr (init+attn, no lr)
  5. SP_large_eps:   SP everything + Adam eps=1e-4 (epsilon control)

Fully autonomous. No user interaction. Saves all results, plots, interpretation.
"""

import csv
import json
import math
import os
import sys
import time
import traceback
import types
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from src.model import Transformer, CausalSelfAttention
from src.task import SurjectiveMap
from src.diagnostics import compute_train_loss, compute_z_shuffle_gap

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
K = 2
N_B = 5000
DATA_SEED = 42
MODEL_SEED = 0
BATCH_SIZE = 128
WARMUP = 500
MAX_STEPS = 3000
EVAL_EVERY = 100
CKPT_STEPS = [1000, 3000]

D_BASE = 128
D_TARGET = 768
M = D_TARGET / D_BASE  # = 6.0

OUT_DIR = "results/overparameterization/unbundle_mup"

# Early stopping
CONVERGE_THRESH = 0.1
DIVERGE_THRESH = 100.0
SUBCRITICAL_ZGAP = 0.01
SUBCRITICAL_PATIENCE = 2000

# Reference values from prior runs (hardcoded)
REFERENCES = {
    "SP_baseline": {
        "z_gap_500": 0.0008, "z_gap_1000": 0.0002,
        "loss_1000": 2.875, "nucleated": False,
    },
    "full_muP": {
        "z_gap_500": 0.0111, "z_gap_1000": 0.9594,
        "loss_1000": 0.875, "nucleated": True,
    },
    "SGD_SP": {
        "z_gap_500": 0.0019, "z_gap_1000": 0.0214,
        "loss_1000": 2.814, "nucleated": True,
    },
}

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

CONFIGS = [
    {
        "name": "SP_muP_lr",
        "desc": "SP init + SP attention + layer-wise lr (muP lr only)",
        "init": "sp",
        "attn_scale": "sqrt",     # / sqrt(d_head)
        "lr_mode": "layerwise",   # hidden weights at lr/m
        "eps": 1e-8,
        "base_lr": 1e-3,
    },
    {
        "name": "muP_init_only",
        "desc": "muP init + SP attention + uniform lr",
        "init": "mup",
        "attn_scale": "sqrt",
        "lr_mode": "uniform",
        "eps": 1e-8,
        "base_lr": 1e-3,
    },
    {
        "name": "muP_attn_only",
        "desc": "SP init + muP attention + uniform lr",
        "init": "sp",
        "attn_scale": "linear",   # / d_head
        "lr_mode": "uniform",
        "eps": 1e-8,
        "base_lr": 1e-3,
    },
    {
        "name": "muP_init_attn",
        "desc": "muP init + muP attention + uniform lr (no lr scaling)",
        "init": "mup",
        "attn_scale": "linear",
        "lr_mode": "uniform",
        "eps": 1e-8,
        "base_lr": 1e-3,
    },
    {
        "name": "SP_large_eps",
        "desc": "SP everything + Adam eps=1e-4",
        "init": "sp",
        "attn_scale": "sqrt",
        "lr_mode": "uniform",
        "eps": 1e-4,
        "base_lr": 1e-3,
    },
]


# ═══════════════════════════════════════════════════════════════════════
# MODEL SETUP HELPERS
# ═══════════════════════════════════════════════════════════════════════

def patched_attn_forward(self, x, return_attention=False):
    """Attention forward with configurable scaling."""
    B, T, C = x.shape
    q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    k = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    v = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    attn = (q @ k.transpose(-2, -1)) / self._attn_scale
    attn = attn.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
    attn = F.softmax(attn, dim=-1)
    out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
    if return_attention:
        return self.W_O(out), attn
    return self.W_O(out)


def build_model(cfg, device):
    """Create model, apply init and attention scaling per config."""
    torch.manual_seed(MODEL_SEED)
    model = Transformer(
        n_layers=12, n_heads=12, d_model=768, d_mlp=3072,
        vocab_size=40, max_seq_len=16,
    ).to(device)

    # Patch attention scaling
    for block in model.blocks:
        attn = block.attn
        if cfg["attn_scale"] == "linear":
            attn._attn_scale = float(attn.d_head)  # / d_head (muP)
        else:
            attn._attn_scale = math.sqrt(float(attn.d_head))  # / sqrt(d_head) (SP)
        attn.forward = types.MethodType(patched_attn_forward, attn)

    # Apply initialization
    if cfg["init"] == "mup":
        with torch.no_grad():
            nn.init.normal_(model.tok_embed.weight, 0.0, 1.0)
            nn.init.normal_(model.pos_embed.weight, 0.0, 1.0)
            nn.init.normal_(model.unembed.weight, 0.0, 1.0 / math.sqrt(D_TARGET))
            for block in model.blocks:
                for W in [block.attn.W_Q, block.attn.W_K,
                          block.attn.W_V, block.attn.W_O]:
                    nn.init.normal_(W.weight, 0.0, 1.0 / math.sqrt(D_TARGET))
                nn.init.normal_(block.mlp_in.weight, 0.0, 1.0 / math.sqrt(D_TARGET))
                nn.init.normal_(block.mlp_out.weight, 0.0, 1.0 / math.sqrt(D_TARGET))
                nn.init.ones_(block.ln1.weight)
                nn.init.zeros_(block.ln1.bias)
                nn.init.ones_(block.ln2.weight)
                nn.init.zeros_(block.ln2.bias)
            nn.init.ones_(model.ln_final.weight)
            nn.init.zeros_(model.ln_final.bias)
    # else: keep default SP init from Transformer.__init__

    return model


def build_optimizer(model, cfg):
    """Create AdamW with appropriate lr per parameter group."""
    base_lr = cfg["base_lr"]

    if cfg["lr_mode"] == "layerwise":
        embed_params = (list(model.tok_embed.parameters()) +
                        list(model.pos_embed.parameters()))
        output_params = list(model.unembed.parameters())
        ln_params = list(model.ln_final.parameters())
        hidden_params = []
        for block in model.blocks:
            hidden_params.extend(list(block.attn.W_Q.parameters()))
            hidden_params.extend(list(block.attn.W_K.parameters()))
            hidden_params.extend(list(block.attn.W_V.parameters()))
            hidden_params.extend(list(block.attn.W_O.parameters()))
            hidden_params.extend(list(block.mlp_in.parameters()))
            hidden_params.extend(list(block.mlp_out.parameters()))
            ln_params.extend(list(block.ln1.parameters()))
            ln_params.extend(list(block.ln2.parameters()))

        param_groups = [
            {"params": embed_params, "lr": base_lr, "label": "embed"},
            {"params": hidden_params, "lr": base_lr / M, "label": "hidden"},
            {"params": output_params, "lr": base_lr, "label": "output"},
            {"params": ln_params, "lr": base_lr, "label": "layernorm"},
        ]
    else:
        param_groups = [
            {"params": list(model.parameters()), "lr": base_lr, "label": "all"},
        ]

    # Store base_lr for warmup
    for pg in param_groups:
        pg["base_lr"] = pg["lr"]

    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.999),
        weight_decay=0.01, eps=cfg["eps"],
    )
    return optimizer, param_groups


# ═══════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def verify_model(model, cfg, param_groups, device):
    """Compute init stats and forward-pass diagnostics."""
    # Parameter stats
    pg_stats = []
    for pg in param_groups:
        total = sum(p.numel() for p in pg["params"])
        stds = [p.data.std().item() for p in pg["params"] if p.numel() > 1]
        avg_std = sum(stds) / len(stds) if stds else 0
        sample_val = pg["params"][0].data.flatten()[0].item() if pg["params"] else 0
        pg_stats.append({
            "group": pg["label"], "n_params": total,
            "init_std": round(avg_std, 6), "lr": pg["lr"],
            "sample_weight": round(sample_val, 6),
        })

    # Forward-pass diagnostics
    model.eval()
    x = torch.randint(0, 40, (4, 16)).to(device)
    with torch.no_grad():
        logits, all_attn = model.forward_with_attention(x)

    # Mean absolute attention logit (proxy via attention entropy)
    entropies = []
    for attn_w in all_attn:
        p = attn_w.mean(dim=0)  # avg over batch → (n_heads, T, T)
        ent = -(p * (p + 1e-10).log()).sum(dim=-1)
        entropies.append(ent.mean().item())
    avg_entropy = sum(entropies) / len(entropies)
    logit_std = logits.std().item()

    model.train()
    return {
        "config": cfg["name"],
        "param_groups": pg_stats,
        "avg_attn_entropy": round(avg_entropy, 4),
        "output_logit_std": round(logit_std, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train_one(cfg, device, dataset, full_data):
    """Train one configuration. Returns result dict."""
    name = cfg["name"]
    run_dir = os.path.join(OUT_DIR, name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.csv")

    print(f"\n{'='*65}")
    print(f"  TRAINING: {name}")
    print(f"  {cfg['desc']}")
    print(f"  init={cfg['init']} attn={cfg['attn_scale']} "
          f"lr={cfg['lr_mode']} eps={cfg['eps']}")
    print(f"{'='*65}", flush=True)

    model = build_model(cfg, device)
    optimizer, param_groups = build_optimizer(model, cfg)

    # Verification
    v = verify_model(model, cfg, param_groups, device)
    print(f"  Init verification:")
    for pg in v["param_groups"]:
        print(f"    {pg['group']:>12s}: {pg['n_params']:>10,} params, "
              f"std={pg['init_std']:.4f}, lr={pg['lr']:.2e}")
    print(f"  Attn entropy: {v['avg_attn_entropy']:.4f}  "
          f"(max {math.log(16):.4f})")
    print(f"  Logit std: {v['output_logit_std']:.4f}", flush=True)

    batch_rng = np.random.RandomState(MODEL_SEED + 10000)
    t0 = time.time()
    subcritical_steps = 0
    stopped_reason = None
    metrics_rows = []

    # CSV writer
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "z_shuffle_gap", "wall_seconds"])

    for step in range(MAX_STEPS + 1):
        if step % EVAL_EVERY == 0:
            model.eval()
            loss_m = compute_train_loss(model, full_data, device)
            z_m = compute_z_shuffle_gap(model, full_data, device, batch_size=1024)
            wall = time.time() - t0
            row = {
                "step": step,
                "train_loss": loss_m["train_loss"],
                "z_shuffle_gap": z_m["z_shuffle_gap"],
                "wall_seconds": round(wall, 1),
            }
            metrics_rows.append(row)
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, row["train_loss"],
                                 row["z_shuffle_gap"], row["wall_seconds"]])

            loss = row["train_loss"]
            zgap = row["z_shuffle_gap"]

            if step % 500 == 0:
                print(f"  [{name}] step {step:>4}  loss={loss:.4f}  "
                      f"z_gap={zgap:.4f}  wall={wall:.0f}s", flush=True)

            # Early stopping
            if loss < CONVERGE_THRESH:
                stopped_reason = f"converged (loss={loss:.4f})"
                print(f"  ** EARLY STOP: {stopped_reason} at step {step}")
                break
            if loss > DIVERGE_THRESH or math.isnan(loss):
                stopped_reason = f"diverged (loss={loss:.1f})"
                print(f"  ** EARLY STOP: {stopped_reason} at step {step}")
                break
            if zgap < SUBCRITICAL_ZGAP:
                subcritical_steps += EVAL_EVERY
            else:
                subcritical_steps = 0
            if subcritical_steps >= SUBCRITICAL_PATIENCE and step >= SUBCRITICAL_PATIENCE:
                stopped_reason = f"subcritical ({subcritical_steps} steps)"
                print(f"  ** EARLY STOP: {stopped_reason} at step {step}")
                break
            model.train()

        if step == MAX_STEPS:
            break

        # Checkpoint
        if step in CKPT_STEPS:
            ckpt_path = os.path.join(run_dir, "checkpoints", f"step_{step}.pt")
            torch.save(model.state_dict(), ckpt_path)

        # LR warmup
        warmup_mult = min(1.0, step / max(1, WARMUP))
        for pg in optimizer.param_groups:
            pg["lr"] = pg["base_lr"] * warmup_mult

        # Training step
        batch = dataset.get_batch(BATCH_SIZE, batch_rng).to(device)
        loss_val, _ = model(batch, batch)

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            stopped_reason = f"NaN at step {step}"
            print(f"  ** EARLY STOP: {stopped_reason}")
            break

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    # Final checkpoint
    ckpt_path = os.path.join(run_dir, "checkpoints", f"step_final.pt")
    torch.save(model.state_dict(), ckpt_path)

    wall = time.time() - t0
    final = metrics_rows[-1] if metrics_rows else {}

    # Detect nucleation: z_gap > 0.05 sustained after step 500
    nucleated = False
    for r in metrics_rows:
        if r["step"] >= 500 and r["z_shuffle_gap"] > 0.05:
            # Check if any later point also exceeds
            later = [rr for rr in metrics_rows
                     if rr["step"] > r["step"] and rr["z_shuffle_gap"] > 0.03]
            if later:
                nucleated = True
                break

    # Extract key checkpoint values
    def val_at(step_target, key):
        for r in metrics_rows:
            if r["step"] == step_target:
                return r[key]
        return None

    result = {
        "name": name,
        "desc": cfg["desc"],
        "init": cfg["init"],
        "attn_scale": cfg["attn_scale"],
        "lr_mode": cfg["lr_mode"],
        "eps": cfg["eps"],
        "final_step": final.get("step", 0),
        "final_loss": final.get("train_loss"),
        "final_z_gap": final.get("z_shuffle_gap"),
        "z_gap_500": val_at(500, "z_shuffle_gap"),
        "z_gap_1000": val_at(1000, "z_shuffle_gap"),
        "loss_1000": val_at(1000, "train_loss"),
        "nucleated": nucleated,
        "stopped_reason": stopped_reason or "completed",
        "wall_seconds": round(wall, 1),
        "verification": v,
    }

    print(f"  Done: {wall:.0f}s ({wall/60:.1f} min). "
          f"loss={result['final_loss']:.4f} z_gap={result['final_z_gap']:.4f} "
          f"nucleated={'YES' if nucleated else 'no'}", flush=True)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS AND PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def load_ref_metrics(name, path):
    """Try to load reference metrics from a prior run."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def generate_plots(results):
    """Generate all plots."""
    # Load metrics for each config
    config_metrics = {}
    for r in results:
        csv_path = os.path.join(OUT_DIR, r["name"], "metrics.csv")
        if os.path.exists(csv_path):
            steps, losses, zgaps = [], [], []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    steps.append(int(row["step"]))
                    losses.append(float(row["train_loss"]))
                    zgaps.append(float(row["z_shuffle_gap"]))
            config_metrics[r["name"]] = (steps, losses, zgaps)

    # Reference curves
    ref_paths = {
        "SP_baseline": "results/lr_sweep_85M/lr_1e-3/runs/D10000_seed0/metrics.jsonl",
        "full_muP": "results/overparameterization/mup_test/85M_muP/metrics.jsonl",
    }
    ref_metrics = {}
    for name, path in ref_paths.items():
        rows = load_ref_metrics(name, path)
        if rows:
            ref_metrics[name] = (
                [r["step"] for r in rows],
                [r["train_loss"] for r in rows],
                [r["z_shuffle_gap"] for r in rows],
            )

    colors = {
        "SP_muP_lr": "#1f77b4",
        "muP_init_only": "#ff7f0e",
        "muP_attn_only": "#2ca02c",
        "muP_init_attn": "#d62728",
        "SP_large_eps": "#9467bd",
        "SP_baseline": "#888888",
        "full_muP": "#000000",
    }

    # ---- MAIN PLOT: z_gap vs step ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, (steps, losses, zgaps) in config_metrics.items():
        zgaps_clip = [max(z, 1e-4) for z in zgaps]
        ax.plot(steps, zgaps_clip, "o-", color=colors.get(name, "gray"),
                lw=2, ms=4, label=name)
    for name, (steps, losses, zgaps) in ref_metrics.items():
        zgaps_clip = [max(z, 1e-4) for z in zgaps]
        style = "--" if name == "SP_baseline" else "-."
        ax.plot(steps, zgaps_clip, style, color=colors.get(name, "gray"),
                lw=1.5, alpha=0.6, label=f"{name} (ref)")
    ax.axhline(0.05, color="gray", ls=":", lw=0.8, alpha=0.5,
               label="nucleation threshold (0.05)")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("z_shuffle_gap", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 10)
    ax.set_xlim(0, MAX_STEPS)
    ax.set_title("Unbundling muP: Which Component Enables Circuit Discovery?",
                 fontsize=13)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "unbundle_mup.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- LOSS PLOT ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, (steps, losses, zgaps) in config_metrics.items():
        losses_clip = [min(l, 10) for l in losses]
        ax.plot(steps, losses_clip, "o-", color=colors.get(name, "gray"),
                lw=2, ms=4, label=name)
    for name, (steps, losses, zgaps) in ref_metrics.items():
        losses_clip = [min(l, 10) for l in losses]
        style = "--" if name == "SP_baseline" else "-."
        ax.plot(steps, losses_clip, style, color=colors.get(name, "gray"),
                lw=1.5, alpha=0.6, label=f"{name} (ref)")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("train_loss", fontsize=12)
    ax.set_xlim(0, MAX_STEPS)
    ax.set_title("Loss trajectories", fontsize=13)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "unbundle_mup_loss.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- ATTENTION ENTROPY BAR CHART ----
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["name"] for r in results]
    entropies = [r["verification"]["avg_attn_entropy"] for r in results]
    bars = ax.bar(range(len(names)), entropies, color=[colors.get(n, "gray")
                                                        for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Initial attention entropy")
    ax.axhline(math.log(16), color="gray", ls=":", lw=0.8,
               label=f"max entropy (log 16 = {math.log(16):.2f})")
    ax.set_title("Initial attention entropy by configuration")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "init_attention_entropy.png"), dpi=150)
    plt.close(fig)

    print("Plots saved.", flush=True)


def generate_interpretation(results):
    """Automatically generate interpretation based on which configs nucleated."""
    nucleated = {r["name"]: r["nucleated"] for r in results}

    # Count nucleated among configs 1-4
    c1 = nucleated.get("SP_muP_lr", False)
    c2 = nucleated.get("muP_init_only", False)
    c3 = nucleated.get("muP_attn_only", False)
    c4 = nucleated.get("muP_init_attn", False)
    c5 = nucleated.get("SP_large_eps", False)

    singles = []
    if c1: singles.append(("SP_muP_lr", "layer-wise lr scaling"))
    if c2: singles.append(("muP_init_only", "muP initialization (N(0, 1/d) for hidden weights)"))
    if c3: singles.append(("muP_attn_only", "muP attention scaling (1/d_head instead of 1/sqrt(d_head))"))

    lines = []
    lines.append("=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)
    lines.append("")

    lines.append("Nucleation results:")
    for r in results:
        lines.append(f"  {r['name']:>18s}: {'NUCLEATED' if r['nucleated'] else 'no'}")
    lines.append("")

    if len(singles) == 1:
        name, component = singles[0]
        lines.append(f"SINGLE ACTIVE INGREDIENT IDENTIFIED: {name}")
        lines.append(f"The component that rescues nucleation is: {component}.")
        lines.append(f"The other muP components are neither necessary nor sufficient.")
        if name == "SP_muP_lr":
            lines.append("This means the failure is specifically about uniform lr ")
            lines.append("applied to hidden layers that are too wide. Layer-wise lr ")
            lines.append("scaling alone (without muP init or attention) fixes it.")
        elif name == "muP_init_only":
            lines.append("This means the failure is about initialization scale. ")
            lines.append("Standard init creates weight magnitudes that make the ")
            lines.append("z-signal a vanishing fraction of gradient variance.")
        elif name == "muP_attn_only":
            lines.append("This means the failure is about attention logit magnitude. ")
            lines.append("Standard 1/sqrt(d_head) scaling creates peaked initial ")
            lines.append("attention that resists reshaping toward z-routing.")
    elif len(singles) == 0 and c4:
        lines.append("TWO COMPONENTS REQUIRED: init + attention scaling together")
        lines.append("rescue nucleation, but neither alone suffices.")
        lines.append("The mechanism involves an interaction between initialization")
        lines.append("scale and attention entropy.")
    elif len(singles) == 0 and not c4:
        lines.append("ALL THREE COMPONENTS REQUIRED.")
        lines.append("No subset of muP changes suffices.")
        lines.append("Recommendation: use full muP.")
        lines.append("This is the least informative outcome.")
    elif len(singles) >= 2:
        names = [s[0] for s in singles]
        lines.append(f"MULTIPLE SUFFICIENT FIXES: {', '.join(names)}")
        lines.append("Each independently crosses the nucleation threshold.")
        lines.append("The failure has multiple independent remedies.")

    if c5:
        lines.append("")
        lines.append("UNEXPECTED: Adam's epsilon is the bottleneck.")
        lines.append("SP_large_eps (eps=1e-4) nucleates, suggesting the")
        lines.append("issue is Adam's denominator being too small, not")
        lines.append("per-parameter normalization per se.")

    lines.append("")
    lines.append("Context: SGD+SP also nucleates (from prior experiment),")
    lines.append("confirming the failure is optimizer-specific, not architectural.")

    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "interpretation.txt"), "w") as f:
        f.write(text)
    print(text, flush=True)
    return text


def generate_summary_table(results):
    """Print and save summary table."""
    lines = []
    lines.append("")
    header = (f"{'config':>18s}  {'init':>4s}  {'attn':>5s}  {'lr':>9s}  "
              f"{'eps':>6s}  {'z@500':>7s}  {'z@1000':>8s}  {'L@1000':>8s}  "
              f"{'nucleated':>10s}")
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        z500 = f"{r['z_gap_500']:.4f}" if r['z_gap_500'] is not None else "—"
        z1000 = f"{r['z_gap_1000']:.4f}" if r['z_gap_1000'] is not None else "—"
        l1000 = f"{r['loss_1000']:.4f}" if r['loss_1000'] is not None else "—"
        attn = "1/d" if r["attn_scale"] == "linear" else "1/√d"
        lines.append(
            f"{r['name']:>18s}  {r['init']:>4s}  {attn:>5s}  "
            f"{r['lr_mode']:>9s}  {r['eps']:>6.0e}  {z500:>7s}  "
            f"{z1000:>8s}  {l1000:>8s}  "
            f"{'YES' if r['nucleated'] else 'no':>10s}")

    lines.append("")
    lines.append("--- Reference values (from prior runs) ---")
    for name, ref in REFERENCES.items():
        z500 = f"{ref['z_gap_500']:.4f}" if ref['z_gap_500'] is not None else "—"
        z1000 = f"{ref['z_gap_1000']:.4f}" if ref['z_gap_1000'] is not None else "—"
        l1000 = f"{ref['loss_1000']:.4f}" if ref['loss_1000'] is not None else "—"
        nuc = "YES" if ref["nucleated"] else "no"
        lines.append(f"{name:>18s}  {'—':>4s}  {'—':>5s}  {'—':>9s}  "
                      f"{'—':>6s}  {z500:>7s}  {z1000:>8s}  {l1000:>8s}  "
                      f"{nuc:>10s}")

    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(text)
    print(text, flush=True)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    print(f"PHASE A: Unbundle muP")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Max steps per config: {MAX_STEPS}")
    print(f"Early stopping: converge<{CONVERGE_THRESH}, diverge>{DIVERGE_THRESH}, "
          f"subcritical z_gap<{SUBCRITICAL_ZGAP} for {SUBCRITICAL_PATIENCE} steps")
    print()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build dataset once
    dataset = SurjectiveMap(K=K, n_b=N_B, seed=DATA_SEED)
    full_data = dataset.get_full()
    print(f"Dataset: K={K}, D={dataset.D}")

    # ── Step 1: Verification ──────────────────────────────────
    # NOTE: verification now happens inline within each train_one call
    # to avoid MPS memory fragmentation from creating/destroying 5 models
    # before training starts.
    print("\n  (Verification runs inline with each training config)")

    # ── Step 2: Training ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2: TRAINING")
    print("=" * 65)

    results = []
    verifications = []
    for cfg in CONFIGS:
        try:
            # Clear MPS cache between runs to prevent fragmentation
            if device.type == "mps":
                torch.mps.empty_cache()
            r = train_one(cfg, device, dataset, full_data)
            results.append(r)
            if "verification" in r:
                verifications.append(r["verification"])
        except Exception as e:
            print(f"\n  !! ERROR in {cfg['name']}: {e}")
            traceback.print_exc()
            results.append({
                "name": cfg["name"], "desc": cfg["desc"],
                "init": cfg["init"], "attn_scale": cfg["attn_scale"],
                "lr_mode": cfg["lr_mode"], "eps": cfg["eps"],
                "final_step": 0, "final_loss": None, "final_z_gap": None,
                "z_gap_500": None, "z_gap_1000": None, "loss_1000": None,
                "nucleated": False, "stopped_reason": f"ERROR: {e}",
                "wall_seconds": 0, "verification": {},
            })

    # Save combined results + verification
    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if verifications:
        with open(os.path.join(OUT_DIR, "verification.json"), "w") as f:
            json.dump(verifications, f, indent=2)
        print("\nVerification comparison:")
        print(f"{'config':>18s}  {'init_std':>10s}  {'attn_ent':>10s}  {'logit_std':>10s}")
        for v in verifications:
            std = v["param_groups"][0]["init_std"]
            print(f"{v['config']:>18s}  {std:>10.4f}  "
                  f"{v['avg_attn_entropy']:>10.4f}  {v['output_logit_std']:>10.4f}")

    # ── Step 3: Analysis ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 3: ANALYSIS")
    print("=" * 65)

    generate_summary_table(results)
    generate_plots(results)
    interpretation = generate_interpretation(results)

    # ── Step 4: Completion ────────────────────────────────────
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    one_liner = ""
    nucleated_names = [r["name"] for r in results if r["nucleated"]]
    if nucleated_names:
        one_liner = f"Nucleated: {', '.join(nucleated_names)}"
    else:
        one_liner = "No config nucleated — all three muP components required"

    print("\n" + "=" * 65)
    print("  PHASE A COMPLETE")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results in: {OUT_DIR}/")
    print(f"  Summary: {one_liner}")
    print("=" * 65)

    with open(os.path.join(OUT_DIR, "DONE.txt"), "w") as f:
        f.write(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")
        f.write(f"Summary: {one_liner}\n")

    print("\nAll outputs:")
    for fname in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, fname)
        if os.path.isfile(path):
            print(f"  {fname}")


if __name__ == "__main__":
    main()
