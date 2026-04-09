# Capacity and Structural Discovery in Gradient-Trained Transformers

Experimental codebase studying how neural networks discover informative input features as a function of task load (D/params).

## Quick start (GPU container)

```bash
git clone https://github.com/mihirs-0/capacity-discovery.git
cd capacity-discovery
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Running the full experiment

**RTX 5090 (32 GB VRAM, 32 vCPU, 30 GB disk):**
```bash
# All 50 runs (Phase 1 + 1.5), 24 parallel workers
python scripts/run_parallel.py --phase all --workers 24

# Dry run to preview what will launch
python scripts/run_parallel.py --phase all --dry-run
```

**Smaller GPU / tighter disk:**
```bash
# Fewer workers, sparser checkpoints
python scripts/run_parallel.py --phase all --workers 8 --checkpoint-every 5000
```

**Analysis and plots** (after runs complete):
```bash
python scripts/analyze.py results/phase1_d_sweep --K 20
python scripts/plot_d_sweep.py results/phase1_d_sweep
```

## Parallelization strategy

Each training run uses ~15 MB GPU memory (803K param model). On RTX 5090:

| Resource | Budget | Usage (24 workers) |
|----------|--------|--------------------|
| VRAM | 32 GB | ~12.4 GB (500 MB CUDA ctx + 15 MB model per worker) |
| vCPU | 32 | 24 workers + OS headroom |
| Disk | 30 GB | ~3.4 GB checkpoints + ~12 MB metrics |
| RAM | 141 GB | ~2 GB (datasets + tensors) |

Workers are spawned via `torch.multiprocessing` (spawn context). Each has its own model, optimizer, and dataset. CUDA schedules concurrent kernels automatically.

**Expected wall time:** 50 runs / 24 workers = 3 rounds. ~10-15 min total on RTX 5090.

## Architecture

- `src/tokenizer.py` -- Fixed 40-token character-level tokenizer
- `src/task.py` -- Surjective map dataset generator with constraint verification
- `src/model.py` -- Decoder-only Transformer from scratch (803K params default)
- `src/trainer.py` -- Training loop with eval hooks and checkpointing
- `src/diagnostics.py` -- D1-D6 measurements (loss, z-shuffle gap, group accuracy, stable ranks)
- `src/hessian.py` -- Hessian eigenvalue computation via power iteration
- `scripts/run_parallel.py` -- Parallel launcher (main entry point for GPU runs)
- `scripts/analyze.py` -- Compute tau, summary statistics
- `scripts/plot_d_sweep.py` -- Phase 1 plots (loss curves, tau vs D, etc.)
