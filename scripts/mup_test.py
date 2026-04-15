#!/usr/bin/env python3
"""muP vs Standard Parameterization test for 85M model.

Tests whether Adam's nucleation failure at 85M is due to standard
parameterization (wrong init/lr scaling for wide models) or Adam's
per-parameter normalization itself.

muP (Maximal Update Parameterization, Yang & Hu 2021) makes feature
learning dynamics width-independent by:
  - Scaling init variance as 1/d for hidden weights
  - Scaling attention logits by 1/d_head (not 1/sqrt(d_head))
  - Scaling lr by 1/m for hidden weights (m = d_target/d_base)

4 runs:
  1. 803K SP  (sanity — should nucleate)
  2. 803K muP (sanity — m=1, should be identical to SP)
  3. 85M  SP  (control — should replicate subcritical)
  4. 85M  muP (THE TEST)

Then 2 more based on result of run 4.
"""

import json
import math
import os
import sys
import time
import types

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.model import Transformer, CausalSelfAttention
from src.task import SurjectiveMap
from src.diagnostics import compute_train_loss, compute_z_shuffle_gap

K = 2
N_B = 5000
DATA_SEED = 42
MODEL_SEED = 0
BATCH_SIZE = 128
WARMUP = 500
EVAL_EVERY = 100

D_BASE = 128  # reference width (803K model)

OUT_DIR = "results/overparameterization/mup_test"

# Early stopping
CONVERGE_THRESH = 0.1
DIVERGE_THRESH = 100.0
SUBCRITICAL_ZGAP = 0.01
SUBCRITICAL_PATIENCE = 2000


# ---- muP attention forward (patches 1/sqrt(d_head) → 1/d_head) ----

def mup_attn_forward(self, x, return_attention=False):
    B, T, C = x.shape
    q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    k = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    v = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    # muP: divide by d_head, not sqrt(d_head)
    attn = (q @ k.transpose(-2, -1)) / self._attn_scale
    attn = attn.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
    attn = F.softmax(attn, dim=-1)
    out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
    if return_attention:
        return self.W_O(out), attn
    return self.W_O(out)


def apply_mup(model, d_target, d_base):
    """Apply muP initialization and attention scaling to a model.
    Returns parameter groups with muP-scaled learning rates."""
    m = d_target / d_base

    # muP attention scaling: /d_head instead of /sqrt(d_head)
    for block in model.blocks:
        attn = block.attn
        attn._attn_scale = float(attn.d_head)  # muP: divide by d_head
        attn.forward = types.MethodType(mup_attn_forward, attn)

    # muP initialization
    with torch.no_grad():
        # Embeddings: N(0, 1) — width-independent
        nn.init.normal_(model.tok_embed.weight, mean=0.0, std=1.0)
        nn.init.normal_(model.pos_embed.weight, mean=0.0, std=1.0)

        # Unembed: N(0, 1/d_target)
        nn.init.normal_(model.unembed.weight, mean=0.0, std=1.0 / math.sqrt(d_target))

        for block in model.blocks:
            # Attention Q/K/V/O: N(0, 1/d_target)
            for W in [block.attn.W_Q, block.attn.W_K, block.attn.W_V, block.attn.W_O]:
                nn.init.normal_(W.weight, mean=0.0, std=1.0 / math.sqrt(d_target))

            # MLP: N(0, 1/d_target)
            nn.init.normal_(block.mlp_in.weight, mean=0.0, std=1.0 / math.sqrt(d_target))
            nn.init.normal_(block.mlp_out.weight, mean=0.0, std=1.0 / math.sqrt(d_target))

            # LayerNorm: standard (weight=1, bias=0)
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)

        # Final LayerNorm
        nn.init.ones_(model.ln_final.weight)
        nn.init.zeros_(model.ln_final.bias)

    # Build parameter groups with muP lr scaling
    embed_params = list(model.tok_embed.parameters()) + list(model.pos_embed.parameters())
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

    # hidden weights: lr / m, everything else: lr
    groups = [
        {"params": embed_params, "lr_scale": 1.0, "label": "embeddings"},
        {"params": hidden_params, "lr_scale": 1.0 / m, "label": f"hidden (1/m=1/{m:.1f})"},
        {"params": output_params, "lr_scale": 1.0, "label": "output"},
        {"params": ln_params, "lr_scale": 1.0, "label": "layernorm"},
    ]
    return groups


def apply_sp(model):
    """Standard parameterization — just return flat parameter groups.
    Model uses default init from __init__."""
    # Set the standard attention scale explicitly
    for block in model.blocks:
        attn = block.attn
        attn._attn_scale = math.sqrt(float(attn.d_head))
        attn.forward = types.MethodType(mup_attn_forward, attn)

    return [{"params": list(model.parameters()), "lr_scale": 1.0, "label": "all"}]


def verify_model(model, label, param_groups, device):
    """Print init stats and one-input forward diagnostics."""
    print(f"\n  Verification: {label}")
    for pg in param_groups:
        total = sum(p.numel() for p in pg["params"])
        stds = [p.data.std().item() for p in pg["params"] if p.numel() > 1]
        avg_std = sum(stds) / len(stds) if stds else 0
        print(f"    {pg['label']:>25s}: {total:>10,} params, "
              f"avg init std={avg_std:.4f}, lr_scale={pg['lr_scale']:.4f}")

    # One forward pass for diagnostics
    model.eval()
    x = torch.randint(0, 40, (1, 16)).to(device)
    with torch.no_grad():
        logits, all_attn = model.forward_with_attention(x)

    # Attention entropy (average across heads and layers)
    entropies = []
    for attn in all_attn:
        # attn: (1, n_heads, T, T)
        p = attn[0]  # (n_heads, T, T)
        # Entropy per head per query position
        ent = -(p * (p + 1e-10).log()).sum(dim=-1)  # (n_heads, T)
        entropies.append(ent.mean().item())
    avg_entropy = sum(entropies) / len(entropies)

    # Output logit magnitude
    logit_std = logits.std().item()

    print(f"    avg attention entropy: {avg_entropy:.4f} "
          f"(max possible: {math.log(16):.4f})")
    print(f"    output logit std: {logit_std:.4f}")
    model.train()
    return {"avg_attn_entropy": avg_entropy, "logit_std": logit_std}


def get_lr(step, warmup, base):
    return base * min(1.0, step / max(1, warmup))


def run_one(cfg, device, max_steps):
    label = cfg["label"]
    run_dir = os.path.join(OUT_DIR, label)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  n_layers={cfg['n_layers']}, n_heads={cfg['n_heads']}, "
          f"d_model={cfg['d_model']}, d_mlp={cfg['d_mlp']}")
    print(f"  param_type={cfg['param']}, base_lr={cfg['lr']}")
    print(f"{'='*65}")

    dataset = SurjectiveMap(K=K, n_b=N_B, seed=DATA_SEED)
    full_data = dataset.get_full()

    torch.manual_seed(MODEL_SEED)
    model = Transformer(
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        d_model=cfg["d_model"], d_mlp=cfg["d_mlp"],
        vocab_size=40, max_seq_len=16,
    ).to(device)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    # Apply parameterization
    if cfg["param"] == "mup":
        param_groups = apply_mup(model, cfg["d_model"], D_BASE)
    else:
        param_groups = apply_sp(model)

    # Verify init and forward stats
    verify_model(model, label, param_groups, device)

    # Build optimizer with per-group lr
    base_lr = cfg["lr"]
    opt_groups = []
    for pg in param_groups:
        opt_groups.append({
            "params": pg["params"],
            "lr": base_lr * pg["lr_scale"],
            "base_lr": base_lr * pg["lr_scale"],
        })
    optimizer = torch.optim.AdamW(
        opt_groups, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8,
    )

    batch_rng = np.random.RandomState(MODEL_SEED + 10000)
    t0 = time.time()
    subcritical_steps = 0
    stopped_reason = None

    for step in range(max_steps + 1):
        if step % EVAL_EVERY == 0:
            model.eval()
            loss_m = compute_train_loss(model, full_data, device)
            z_m = compute_z_shuffle_gap(model, full_data, device, batch_size=1024)
            m = {"step": step, **loss_m, **z_m,
                 "wall_time_seconds": time.time() - t0}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(m) + "\n")

            loss = m["train_loss"]
            zgap = m["z_shuffle_gap"]

            if step % 500 == 0:
                print(f"  [{label}] step {step:>5}  loss={loss:.4f}  "
                      f"z_gap={zgap:.4f}  wall={m['wall_time_seconds']:.0f}s",
                      flush=True)

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

        if step == max_steps:
            break

        # LR warmup (per group)
        warmup_mult = min(1.0, step / max(1, WARMUP))
        for pg in optimizer.param_groups:
            pg["lr"] = pg["base_lr"] * warmup_mult

        batch = dataset.get_batch(BATCH_SIZE, batch_rng).to(device)
        loss_val, _ = model(batch, batch)

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            stopped_reason = f"NaN at step {step}"
            print(f"  ** EARLY STOP: {stopped_reason}")
            break

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    wall = time.time() - t0
    with open(metrics_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    final = rows[-1] if rows else {}

    nucleated = any(r["z_shuffle_gap"] > 0.05 for r in rows[5:] if len(rows) > 5)

    result = {
        "label": label,
        "param": cfg["param"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "d_model": cfg["d_model"],
        "lr": cfg["lr"],
        "n_params": n_params,
        "final_step": final.get("step", 0),
        "final_loss": final.get("train_loss"),
        "final_z_gap": final.get("z_shuffle_gap"),
        "stopped_reason": stopped_reason or "completed",
        "nucleated": nucleated,
        "wall_time_seconds": wall,
    }
    print(f"  Done: {wall:.0f}s. loss={result['final_loss']:.4f} "
          f"z_gap={result['final_z_gap']:.4f} nucleated={nucleated}")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def plot_results(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "803K_SP": "#1f77b4",
        "803K_muP": "#aec7e8",
        "85M_SP": "#d62728",
        "85M_muP": "#2ca02c",
    }
    styles = {"SP": "-", "muP": "--"}

    for r in results:
        label = r["label"]
        mp = os.path.join(OUT_DIR, label, "metrics.jsonl")
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        steps = [x["step"] for x in rows]
        zgaps = [max(x["z_shuffle_gap"], 1e-5) for x in rows]
        c = colors.get(label, "gray")
        ax.plot(steps, zgaps, color=c, lw=2, label=label)

    # Add SGD reference
    sgd_path = os.path.join("results/overparameterization/sgd_test/SGD_lr0.1/metrics.jsonl")
    if os.path.exists(sgd_path):
        with open(sgd_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        ax.plot([x["step"] for x in rows],
                [max(x["z_shuffle_gap"], 1e-5) for x in rows],
                "k:", lw=1.5, alpha=0.6, label="85M SGD lr=0.1 (ref)")

    ax.axhline(0.02, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("z_shuffle_gap")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 10)
    ax.set_title("muP vs SP: does parameterization rescue nucleation?")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "mup_test.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Saved: {png}")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Phase 1: 4 core runs
    configs = [
        {"label": "803K_SP",  "param": "sp",  "n_layers": 4,  "n_heads": 4,
         "d_model": 128, "d_mlp": 512, "lr": 1e-3},
        {"label": "803K_muP", "param": "mup", "n_layers": 4,  "n_heads": 4,
         "d_model": 128, "d_mlp": 512, "lr": 1e-3},
        {"label": "85M_SP",   "param": "sp",  "n_layers": 12, "n_heads": 12,
         "d_model": 768, "d_mlp": 3072, "lr": 1e-3},
        {"label": "85M_muP",  "param": "mup", "n_layers": 12, "n_heads": 12,
         "d_model": 768, "d_mlp": 3072, "lr": 1e-3},
    ]

    results = []
    for cfg in configs:
        r = run_one(cfg, device, max_steps=10000)
        results.append(r)

    # Phase 2: follow-up runs based on 85M_muP result
    mup_result = results[3]  # 85M_muP
    if mup_result["nucleated"]:
        print("\n*** 85M muP NUCLEATED — running lower/higher lr ***")
        followups = [
            {"label": "85M_muP_lr3e-4", "param": "mup", "n_layers": 12,
             "n_heads": 12, "d_model": 768, "d_mlp": 3072, "lr": 3e-4},
            {"label": "85M_muP_lr3e-3", "param": "mup", "n_layers": 12,
             "n_heads": 12, "d_model": 768, "d_mlp": 3072, "lr": 3e-3},
        ]
    else:
        print("\n*** 85M muP DID NOT nucleate — pushing lr higher ***")
        followups = [
            {"label": "85M_muP_lr3e-3", "param": "mup", "n_layers": 12,
             "n_heads": 12, "d_model": 768, "d_mlp": 3072, "lr": 3e-3},
            {"label": "85M_muP_lr1e-2", "param": "mup", "n_layers": 12,
             "n_heads": 12, "d_model": 768, "d_mlp": 3072, "lr": 1e-2},
        ]

    for cfg in followups:
        r = run_one(cfg, device, max_steps=10000)
        results.append(r)

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(f"{'label':>20}  {'param':>5}  {'lr':>8}  {'params':>10}  "
          f"{'steps':>6}  {'loss':>8}  {'z_gap':>8}  {'nucleated':>10}  {'reason':>20}")
    print("-" * 95)
    for r in results:
        loss_s = f"{r['final_loss']:.4f}" if r['final_loss'] and not math.isnan(r['final_loss']) else "NaN"
        zgap_s = f"{r['final_z_gap']:.4f}" if r['final_z_gap'] else "—"
        print(f"{r['label']:>20}  {r['param']:>5}  {r['lr']:>8g}  "
              f"{r['n_params']:>10,}  {r['final_step']:>6}  {loss_s:>8}  "
              f"{zgap_s:>8}  {'YES' if r['nucleated'] else 'no':>10}  "
              f"{r['stopped_reason']:>20}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/summary.json")

    plot_results(results)


if __name__ == "__main__":
    main()
