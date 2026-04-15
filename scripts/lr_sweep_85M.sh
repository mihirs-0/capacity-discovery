#!/bin/bash
# Learning rate sweep for 85M model on K=2 D=10K synthetic task.
# Tests whether increasing lr restores z-routing nucleation.
# All runs use the same model_seed=0 (identical init), same data_seed=42.

set -e
cd "$(dirname "$0")/.."

K=2
N_B=5000
DATA_SEED=42
MODEL_SEED=0
MAX_STEPS=5000
EVAL_EVERY=100
CKPT_EVERY=5000

COMMON="--K $K --n_b $N_B --data_seed $DATA_SEED --model_seed $MODEL_SEED \
  --n_layers 12 --n_heads 12 --d_model 768 --d_mlp 3072 \
  --max_steps $MAX_STEPS --eval_every $EVAL_EVERY --checkpoint_every $CKPT_EVERY \
  --device mps"

for LR in 1e-3 3e-3 1e-2 3e-2 1e-1; do
  echo ""
  echo "===== LR=$LR ====="
  echo ""
  python3 scripts/run_single.py $COMMON --lr $LR \
    --experiment_name "lr_sweep_85M/lr_${LR}"
  echo "===== LR=$LR DONE ====="
done

echo ""
echo "ALL LR SWEEP RUNS COMPLETE"
