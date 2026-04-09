"""Phase 2 specific plots: noise sweep analysis."""

import argparse
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.analyze import find_runs, compute_tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

    runs = find_runs(args.experiment_dir)
    if not runs:
        print("No runs found.")
        return

    # Phase 2 would parse noise parameters from config
    # Placeholder for when Phase 2 is implemented
    print(f"Found {len(runs)} runs. Phase 2 plotting not yet implemented.")
    print("Run Phase 1 analysis first with plot_d_sweep.py")


if __name__ == "__main__":
    main()
