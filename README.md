# Capacity and Structural Discovery in Gradient-Trained Transformers

Experimental codebase studying how neural networks discover informative input features as a function of task load (D/params).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # verify everything works
```

## Running experiments

**Single run** (local dev / debugging):
```bash
python scripts/run_single.py --K 20 --n_b 500 --max_steps 2000
```

**Full sweep on GPU** (H100 optimized, parallel):
```bash
# All 50 runs (Phase 1 + 1.5), 8 workers, 50K steps each
python scripts/run_parallel.py --phase all --workers 8

# Phase 1 only, aggressive parallelism
python scripts/run_parallel.py --phase 1 --workers 20

# Dry run to see what would launch
python scripts/run_parallel.py --phase all --dry-run
```

**Analysis and plots**:
```bash
python scripts/analyze.py results/phase1_d_sweep --K 20
python scripts/plot_d_sweep.py results/phase1_d_sweep
```

## Architecture

- `src/tokenizer.py` — Fixed 40-token character-level tokenizer
- `src/task.py` — Surjective map dataset generator with constraint verification
- `src/model.py` — Decoder-only Transformer from scratch (803K params default)
- `src/trainer.py` — Training loop with eval hooks and checkpointing
- `src/diagnostics.py` — D1-D6 measurements (loss, z-shuffle gap, group accuracy, stable ranks)
- `src/hessian.py` — Hessian eigenvalue computation via power iteration

## Parallelization strategy

Each training run uses ~15 MB GPU memory (tiny model). On an H100 (80 GB VRAM, 8 vCPU):
- 8 concurrent workers = ~7 GB GPU usage
- 50 runs / 8 workers = ~7 rounds
- Total wall time: ~35 min at 50K steps/run (vs ~11 hours sequential)

Workers are spawned via `torch.multiprocessing` with the `spawn` context to avoid CUDA fork issues. Each worker has its own model, optimizer, and dataset. The GPU handles concurrent kernels from multiple workers via its built-in scheduler.
