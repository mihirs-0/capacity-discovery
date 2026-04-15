#!/usr/bin/env python3
"""Representation-convergence analysis.

For the forward (A->B) model: do inputs that share an output-B converge to
similar hidden representations at the last input position (the SEP after A,
which is used to emit b1)?

Also runs an analogous probe on the backward (B,z->A) model at the A-prediction
position (the second SEP at pos 10, used to emit a1). In the backward model,
"group" means "shared B input" — B tokens are byte-identical across a group,
so any probe at a B-position is trivially similar.  The non-trivial question
is whether representations *after z-routing* converge for examples whose target
A would come from different members of the same B-group — i.e., they should
NOT converge there (different A targets).  We still report the number for
symmetry, and expect it to be lower than the forward case.

Output:
    results/forward_task/convergence_analysis.txt   (pretty-printed tables)
    results/forward_task/convergence_analysis.json  (machine-readable)

Usage:
    python scripts/analyze_convergence.py
    python scripts/analyze_convergence.py --forward-seed 0 --backward-seed 1 \
        --n-groups 100 --n-between 5
"""

import argparse
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import Transformer
from src.task import SurjectiveMap
from src.task_forward import ForwardSurjectiveMap


# ------------ Probe positions ----------------------------------------------
FORWARD_PROBE_POS = ForwardSurjectiveMap.LAST_INPUT_POS  # 5 (SEP after A)
BACKWARD_PROBE_POS = 10  # second SEP (before a1) in backward layout

LAYER_NAMES = ["embed", "after_L0", "after_L1", "after_L2", "after_L3"]


# ------------ Utilities ----------------------------------------------------

def strip_compiled_prefix(state_dict):
    new = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new[k[len("_orig_mod."):]] = v
        else:
            new[k] = v
    return new


def find_last_checkpoint(run_dir: str) -> tuple[int, str]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints dir at {ckpt_dir}")
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No .pt files in {ckpt_dir}")
    files.sort(key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    last = files[-1]
    step = int(re.search(r"step_(\d+)", last).group(1))
    return step, os.path.join(ckpt_dir, last)


def load_model(ckpt_path: str, device: torch.device) -> Transformer:
    model_cfg = ModelConfig()
    model = Transformer.from_config(model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compiled_prefix(state)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def residual_per_layer(model: Transformer, input_ids: torch.Tensor,
                        probe_pos: int) -> dict[str, torch.Tensor]:
    """Return the residual stream at `probe_pos` after each of:
    embed, block 0, block 1, block 2, block 3.

    Shape per entry: (batch, d_model).  Float32 on CPU.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Capture residual after each block via forward hooks
    captures: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook(_mod, _inp, out):
            # TransformerBlock.forward returns x or (x, attn) in
            # return_attention mode — here it's a plain tensor.
            captures[idx] = out.detach()
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(make_hook(i)))

    try:
        _loss, _logits = model(input_ids)  # triggers hooks
    finally:
        for h in handles:
            h.remove()

    # Recompute embed manually (token + position) — the model adds them in forward
    T = input_ids.shape[1]
    pos = torch.arange(T, device=device)
    embed = (model.tok_embed(input_ids) + model.pos_embed(pos)).detach()

    out = {"embed": embed[:, probe_pos].float().cpu()}
    for i in range(len(model.blocks)):
        out[f"after_L{i}"] = captures[i][:, probe_pos].float().cpu()
    return out


def mean_pairwise_cos(reps: torch.Tensor) -> float:
    """Mean of upper-triangular cosine similarity (excluding diagonal)."""
    if reps.shape[0] < 2:
        return float("nan")
    normed = F.normalize(reps, dim=-1)
    sim = normed @ normed.T                 # (k, k)
    k = sim.shape[0]
    # upper triangle, off-diagonal
    iu = torch.triu_indices(k, k, offset=1)
    return float(sim[iu[0], iu[1]].mean().item())


def mean_cross_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between every row of a and every row of b."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return float((a_n @ b_n.T).mean().item())


# ------------ Analysis -----------------------------------------------------

@torch.no_grad()
def analyze(
    model: Transformer,
    group_inputs: list[torch.Tensor],  # each (K, T), one per sampled group
    probe_pos: int,
    n_between: int,
    rng: np.random.RandomState,
) -> dict:
    """Compute within-group / between-group cosine sim at each layer."""
    n_groups = len(group_inputs)
    # First pass: cache per-group per-layer representations
    cache: list[dict[str, torch.Tensor]] = []
    for inputs in group_inputs:
        reps = residual_per_layer(model, inputs, probe_pos)
        cache.append(reps)

    layer_stats: dict[str, dict] = {}
    for layer in LAYER_NAMES:
        within = []
        between = []
        for i in range(n_groups):
            within.append(mean_pairwise_cos(cache[i][layer]))
            # Pick n_between random other groups
            others = list(range(n_groups))
            others.remove(i)
            pick = rng.choice(len(others), size=min(n_between, len(others)),
                              replace=False)
            cross_vals = []
            for j in pick:
                cross_vals.append(mean_cross_cos(cache[i][layer],
                                                 cache[others[j]][layer]))
            between.append(float(np.mean(cross_vals)))
        within = np.array(within, dtype=np.float64)
        between = np.array(between, dtype=np.float64)
        layer_stats[layer] = {
            "within_mean": float(np.nanmean(within)),
            "within_std":  float(np.nanstd(within)),
            "between_mean": float(np.nanmean(between)),
            "between_std":  float(np.nanstd(between)),
            "ratio": float(np.nanmean(within) / np.nanmean(between))
                if np.nanmean(between) != 0 else float("inf"),
        }

    # Random baseline: cosine between random vectors with matching d_model
    d_model = cache[0][LAYER_NAMES[-1]].shape[-1]
    rand_rng = np.random.RandomState(123)
    rand_reps = torch.tensor(rand_rng.randn(200, d_model), dtype=torch.float32)
    rand_baseline = mean_pairwise_cos(rand_reps)

    return {"layers": layer_stats, "random_baseline": rand_baseline,
            "d_model": d_model, "n_groups": n_groups, "n_between": n_between}


def print_table(title: str, result: dict):
    print(f"\n=== {title} ===")
    print(f"  random baseline (d={result['d_model']}): {result['random_baseline']:+.4f}")
    print(f"  n_groups={result['n_groups']}  n_between={result['n_between']}")
    print()
    print(f"  {'layer':<10} {'within_group_sim':>18} {'between_group_sim':>19} "
          f"{'ratio':>10}")
    print(f"  {'-'*10} {'-'*18} {'-'*19} {'-'*10}")
    for layer in LAYER_NAMES:
        s = result["layers"][layer]
        print(f"  {layer:<10} "
              f"{s['within_mean']:+.4f} ± {s['within_std']:.4f}   "
              f"{s['between_mean']:+.4f} ± {s['between_std']:.4f}   "
              f"{s['ratio']:>7.3f}")


# ------------ Main ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-seed", type=int, default=0)
    ap.add_argument("--forward-experiment", default="forward_task")
    ap.add_argument("--backward-seed", type=int, default=1,
                    help="Which backward seed to analyze (1 or 3 in phase1_d_sweep)")
    ap.add_argument("--backward-experiment", default="phase1_d_sweep")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n_b", type=int, default=500)
    ap.add_argument("--data-seed-forward", type=int, default=10000)
    ap.add_argument("--data-seed-backward", type=int, default=10000,
                    help="Matches phase1_d_sweep convention (data_seed=D).")
    ap.add_argument("--n-groups", type=int, default=100)
    ap.add_argument("--n-between", type=int, default=5)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--skip-backward", action="store_true")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Device: {device}")

    D_nominal = args.K * args.n_b
    out_dir = os.path.join("results", args.forward_experiment)
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    # ---- Forward -----------------------------------------------------------
    fwd_run = os.path.join("results", args.forward_experiment, "runs",
                           f"D{D_nominal}_seed{args.forward_seed}")
    fwd_step, fwd_ckpt = find_last_checkpoint(fwd_run)
    print(f"\nForward: loading {fwd_ckpt} (step {fwd_step})")
    fwd_model = load_model(fwd_ckpt, device)

    fwd_dataset = ForwardSurjectiveMap(
        K=args.K, n_b=args.n_b, seed=args.data_seed_forward
    )
    # Sample n_groups groups that still have all K members after collision drop
    rng = np.random.RandomState(args.sampling_seed)
    valid_groups = [gi for gi in range(fwd_dataset.n_b)
                    if fwd_dataset.group_size(gi) >= 2]
    pick = rng.choice(len(valid_groups),
                      size=min(args.n_groups, len(valid_groups)),
                      replace=False)
    fwd_group_inputs = [fwd_dataset.get_group_members(valid_groups[i])
                        for i in pick]
    print(f"  forward: {len(fwd_group_inputs)} groups, "
          f"mean k={np.mean([g.shape[0] for g in fwd_group_inputs]):.1f}")

    fwd_result = analyze(fwd_model, fwd_group_inputs,
                         probe_pos=FORWARD_PROBE_POS,
                         n_between=args.n_between, rng=rng)
    fwd_result["checkpoint_step"] = fwd_step
    fwd_result["checkpoint_path"] = fwd_ckpt
    fwd_result["probe_position"] = FORWARD_PROBE_POS
    fwd_result["direction"] = "forward"
    summary["forward"] = fwd_result
    print_table(f"FORWARD (A->B), seed={args.forward_seed}, step={fwd_step}, "
                f"probe pos={FORWARD_PROBE_POS}", fwd_result)

    # ---- Backward ----------------------------------------------------------
    if not args.skip_backward:
        bwd_run = os.path.join("results", args.backward_experiment, "runs",
                               f"D{D_nominal}_seed{args.backward_seed}")
        if not os.path.isdir(os.path.join(bwd_run, "checkpoints")):
            print(f"\n[warn] no backward checkpoints at {bwd_run}; skipping")
        else:
            bwd_step, bwd_ckpt = find_last_checkpoint(bwd_run)
            print(f"\nBackward: loading {bwd_ckpt} (step {bwd_step})")
            bwd_model = load_model(bwd_ckpt, device)

            bwd_dataset = SurjectiveMap(
                K=args.K, n_b=args.n_b, seed=args.data_seed_backward
            )
            rng2 = np.random.RandomState(args.sampling_seed)
            bwd_pick = rng2.choice(bwd_dataset.n_b,
                                   size=min(args.n_groups, bwd_dataset.n_b),
                                   replace=False)
            bwd_group_inputs = [bwd_dataset.get_group_examples(int(gi))
                                for gi in bwd_pick]
            bwd_result = analyze(bwd_model, bwd_group_inputs,
                                 probe_pos=BACKWARD_PROBE_POS,
                                 n_between=args.n_between, rng=rng2)
            bwd_result["checkpoint_step"] = bwd_step
            bwd_result["checkpoint_path"] = bwd_ckpt
            bwd_result["probe_position"] = BACKWARD_PROBE_POS
            bwd_result["direction"] = "backward"
            summary["backward"] = bwd_result
            print_table(
                f"BACKWARD (B,z->A), seed={args.backward_seed}, "
                f"step={bwd_step}, probe pos={BACKWARD_PROBE_POS} "
                f"(SEP before a1)",
                bwd_result,
            )

    # Save
    json_path = os.path.join(out_dir, "convergence_analysis.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
