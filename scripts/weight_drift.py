#!/usr/bin/env python3
"""Measure how far each weight matrix has moved from its initial value.

Compares ||W_step - W_init||_F / ||W_init||_F for every parameter
in the model, at D=100K (stuck) vs D=20K (converged).

Predicted result if Q/K vs V/O dissociation is real:
  - At D=100K: W_Q, W_K have moved significantly; W_V, W_O have NOT.
  - At D=20K converged: all of W_Q, W_K, W_V, W_O have moved.
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def strip_compiled_prefix(state_dict):
    new = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new[k[len("_orig_mod."):]] = v
        else:
            new[k] = v
    return new


def relative_drift(w_now, w_init):
    """||w_now - w_init||_F / ||w_init||_F"""
    if w_init.dtype == torch.bool:
        return float("nan")
    diff = (w_now.float() - w_init.float())
    init_norm = w_init.float().norm().item()
    if init_norm == 0:
        return float("nan")
    return (diff.norm() / init_norm).item()


def absolute_drift(w_now, w_init):
    """||w_now - w_init||_F"""
    if w_init.dtype == torch.bool:
        return float("nan")
    diff = (w_now.float() - w_init.float())
    return diff.norm().item()


def compare_run(experiment_dir, D, seed):
    runs_dir = os.path.join(experiment_dir, "runs")
    ckpt_dir = os.path.join(runs_dir, f"D{D}_seed{seed}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
        key=lambda f: int(re.search(r"step_(\d+)", f).group(1)),
    )
    if len(ckpts) < 2:
        return None

    init_path = os.path.join(ckpt_dir, "step_0.pt")
    final_path = os.path.join(ckpt_dir, ckpts[-1])
    final_step = int(re.search(r"step_(\d+)", ckpts[-1]).group(1))

    state_init = strip_compiled_prefix(
        torch.load(init_path, map_location="cpu", weights_only=True))
    state_final = strip_compiled_prefix(
        torch.load(final_path, map_location="cpu", weights_only=True))

    drifts = {}
    for key in state_init:
        if key not in state_final:
            continue
        drifts[key] = {
            "rel": relative_drift(state_final[key], state_init[key]),
            "abs": absolute_drift(state_final[key], state_init[key]),
            "init_norm": state_init[key].float().norm().item(),
        }

    return {
        "D": D,
        "seed": seed,
        "final_step": final_step,
        "drifts": drifts,
    }


def categorize_attention_drifts(drifts):
    """Group attention weights by Q/K/V/O across all layers."""
    cats = {"W_Q": [], "W_K": [], "W_V": [], "W_O": []}
    for key, d in drifts.items():
        for c in cats:
            if c.replace("_", "") in key.replace("_", "") and "attn" in key:
                cats[c].append(d["rel"])
                break
    return {k: np.mean(v) if v else float("nan") for k, v in cats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str,
                        default="results/phase1_d_sweep")
    args = parser.parse_args()

    # Find one valid run for each D
    targets = []
    for D in [20000, 50000, 100000]:
        for seed in range(5):
            r = compare_run(args.experiment_dir, D, seed)
            if r is not None:
                targets.append(r)
                break

    if not targets:
        print("No checkpoint pairs found.")
        return

    # Per-parameter drift table for each run
    for r in targets:
        D = r["D"]
        seed = r["seed"]
        step = r["final_step"]
        print(f"\n=== D={D} seed={seed} step={step} ===")
        print(f"  {'parameter':<35}  {'rel_drift':>10}  {'abs_drift':>10}  {'init_norm':>10}")
        print(f"  {'-' * 70}")
        for key in sorted(r["drifts"].keys()):
            d = r["drifts"][key]
            if "blocks" in key or "embed" in key or "unembed" in key:
                print(f"  {key:<35}  {d['rel']:>10.4f}  {d['abs']:>10.4f}  {d['init_norm']:>10.4f}")

    # Headline: per-category attention drift across runs
    print(f"\n{'=' * 70}")
    print("ATTENTION SUBCIRCUIT DRIFT (mean rel drift across all 4 layers)")
    print(f"{'=' * 70}")
    print(f"\n  {'D':>8} {'step':>8}  {'W_Q':>10}  {'W_K':>10}  {'W_V':>10}  {'W_O':>10}")
    print(f"  {'-' * 60}")
    for r in targets:
        cats = categorize_attention_drifts(r["drifts"])
        print(f"  {r['D']:>8} {r['final_step']:>8}  "
              f"{cats['W_Q']:>10.4f}  {cats['W_K']:>10.4f}  "
              f"{cats['W_V']:>10.4f}  {cats['W_O']:>10.4f}")

    print("\nInterpretation:")
    print("  Q/K should move similarly across D values (routing forms).")
    print("  V/O should be much smaller at D=100K than D=20K (content learning fails).")

    # Per-layer breakdown for the headline runs
    print(f"\n{'=' * 70}")
    print("PER-LAYER ATTENTION DRIFT")
    print(f"{'=' * 70}")
    for r in targets:
        D = r["D"]
        step = r["final_step"]
        print(f"\nD={D} step={step}:")
        print(f"  {'layer':>6}  {'W_Q':>8}  {'W_K':>8}  {'W_V':>8}  {'W_O':>8}")
        for li in range(4):
            cats = {}
            for key, d in r["drifts"].items():
                if f"blocks.{li}.attn." not in key:
                    continue
                if "W_Q" in key:
                    cats["Q"] = d["rel"]
                elif "W_K" in key:
                    cats["K"] = d["rel"]
                elif "W_V" in key:
                    cats["V"] = d["rel"]
                elif "W_O" in key:
                    cats["O"] = d["rel"]
            print(f"  {f'L{li}':>6}  "
                  f"{cats.get('Q', 0):>8.4f}  {cats.get('K', 0):>8.4f}  "
                  f"{cats.get('V', 0):>8.4f}  {cats.get('O', 0):>8.4f}")

    # Save raw results
    import json
    out_path = os.path.join(args.experiment_dir, "weight_drift.json")
    serializable = []
    for r in targets:
        serializable.append({
            "D": r["D"],
            "seed": r["seed"],
            "final_step": r["final_step"],
            "drifts": r["drifts"],
        })
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
