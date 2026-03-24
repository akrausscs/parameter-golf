#!/bin/bash
# Parameter Golf — experiment runner
#
# Usage:
#   bash run.sh baseline                         # standard training, full steps
#   bash run.sh fomaml                           # FOMAML K=1, full steps
#   bash run.sh baseline --iters 4000            # 20% steps (directional)
#   bash run.sh fomaml --iters 4000 --k 1       # FOMAML K=1, 20% steps
#   bash run.sh fomaml --iters 4000 --k 3 --inner-lr 0.005
#   bash run.sh baseline --seed 42               # different seed
#   bash run.sh fomaml --gpus 8                  # multi-GPU (8xH100)
#
# Results written to:
#   logs/<run_id>.txt          full training log
#   logs/<run_id>_summary.json one-line result summary

set -euo pipefail

# ---- Defaults ----
MODE="${1:-baseline}"
shift || true

GPUS=1
ITERATIONS=20000
SEED=1337
FOMAML_K=1
FOMAML_INNER_LR=0.01
WALLCLOCK=600.0   # 10 min — set to 0 for no cap

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPUS="$2";           shift 2 ;;
        --iters)      ITERATIONS="$2";     shift 2 ;;
        --seed)       SEED="$2";           shift 2 ;;
        --k)          FOMAML_K="$2";       shift 2 ;;
        --inner-lr)   FOMAML_INNER_LR="$2"; shift 2 ;;
        --wallclock)  WALLCLOCK="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Build run ID ----
TIMESTAMP=$(date +%m%d_%H%M)
if [ "$MODE" = "fomaml" ]; then
    RUN_ID="fomaml_k${FOMAML_K}_lr${FOMAML_INNER_LR}_i${ITERATIONS}_s${SEED}_${TIMESTAMP}"
    K_VAL=$FOMAML_K
else
    RUN_ID="baseline_i${ITERATIONS}_s${SEED}_${TIMESTAMP}"
    K_VAL=0
fi

# ---- Data paths ----
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

# ---- Guard: check data exists ----
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: data not found at $DATA_PATH"
    echo "Run: bash setup.sh"
    exit 1
fi

echo "========================================"
echo "Mode:       $MODE"
echo "Run ID:     $RUN_ID"
echo "Iterations: $ITERATIONS"
echo "Seed:       $SEED"
echo "GPUs:       $GPUS"
if [ "$K_VAL" -gt 0 ]; then
echo "FOMAML K:   $FOMAML_K"
echo "Inner LR:   $FOMAML_INNER_LR"
fi
echo "========================================"
echo ""

mkdir -p logs

# ---- Launch ----
RUN_ID=$RUN_ID \
SEED=$SEED \
ITERATIONS=$ITERATIONS \
MAX_WALLCLOCK_SECONDS=$WALLCLOCK \
FOMAML_K=$K_VAL \
FOMAML_INNER_LR=$FOMAML_INNER_LR \
DATA_PATH=$DATA_PATH \
TOKENIZER_PATH=$TOKENIZER_PATH \
VOCAB_SIZE=1024 \
torchrun \
    --standalone \
    --nproc_per_node=$GPUS \
    train_gpt.py

# ---- Print summary ----
SUMMARY="logs/${RUN_ID}_summary.json"
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "========================================"
    echo "Result:"
    python3 -c "
import json
s = json.load(open('$SUMMARY'))
print(f\"  BPB:      {s['val_bpb']:.6f}\")
print(f\"  Steps:    {s['steps']}\")
print(f\"  Artifact: {s['artifact_bytes'] / 1e6:.2f} MB\")
print(f\"  Log:      logs/{s['run_id']}.txt\")
"
    echo "========================================"
fi
