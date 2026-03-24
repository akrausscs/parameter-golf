#!/bin/bash
# Parameter Golf — experiment runner
#
# Usage:
#   bash run.sh baseline                         # root train_gpt.py (getting started)
#   bash run.sh frontier                         # current SOTA record
#   bash run.sh late-ema                         # Late EMA fix only
#   bash run.sh bos-reset                        # BOS-reset attention only
#   bash run.sh combined                         # BOS-reset + Late EMA (both)
#
# Common flags:
#   --iters 4000    directional probe (20% of full run — fast feedback)
#   --seed 42       different seed
#   --gpus 8        multi-GPU (8xH100 for leaderboard submissions)
#   --wallclock 0   no time cap (full convergence)
#
# Results written to:
#   logs/<run_id>.log          full training log

set -euo pipefail

# ---- Defaults ----
MODE="${1:-baseline}"
shift || true

GPUS=1
ITERATIONS=20000
SEED=1337
WALLCLOCK=600.0   # 10 min — matches leaderboard cap; set to 0 to disable

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)      GPUS="$2";      shift 2 ;;
        --iters)     ITERATIONS="$2"; shift 2 ;;
        --seed)      SEED="$2";      shift 2 ;;
        --wallclock) WALLCLOCK="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Script selection ----
RECORDS="records/track_10min_16mb"
case "$MODE" in
    baseline)
        SCRIPT="train_gpt.py"
        ;;
    frontier)
        SCRIPT="$RECORDS/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py"
        ;;
    late-ema)
        SCRIPT="$RECORDS/2026-03-25_11L_LateEMA_XSA-all_GPTQ/train_gpt.py"
        ;;
    bos-reset)
        SCRIPT="$RECORDS/2026-03-25_11L_BOS-Reset_XSA-all_GPTQ/train_gpt.py"
        ;;
    combined)
        SCRIPT="$RECORDS/2026-03-25_11L_BOS-Reset_LateEMA_XSA-all_GPTQ/train_gpt.py"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: baseline, frontier, late-ema, bos-reset, combined"
        exit 1
        ;;
esac

if [ ! -f "$SCRIPT" ]; then
    echo "Error: script not found: $SCRIPT"
    exit 1
fi

# ---- Data paths ----
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: data not found at $DATA_PATH"
    echo "Run: bash setup.sh"
    exit 1
fi

# ---- Run ID ----
TIMESTAMP=$(date +%m%d_%H%M)
RUN_ID="${MODE}_i${ITERATIONS}_s${SEED}_${TIMESTAMP}"

mkdir -p logs

echo "========================================"
echo "Mode:       $MODE"
echo "Script:     $SCRIPT"
echo "Run ID:     $RUN_ID"
echo "Iterations: $ITERATIONS"
echo "Seed:       $SEED"
echo "GPUs:       $GPUS"
echo "Wallclock:  ${WALLCLOCK}s"
echo "========================================"
echo ""

# ---- Launch ----
RUN_ID=$RUN_ID \
SEED=$SEED \
ITERATIONS=$ITERATIONS \
MAX_WALLCLOCK_SECONDS=$WALLCLOCK \
DATA_PATH=$DATA_PATH \
TOKENIZER_PATH=$TOKENIZER_PATH \
VOCAB_SIZE=1024 \
torchrun \
    --standalone \
    --nproc_per_node=$GPUS \
    "$SCRIPT" \
    2>&1 | tee "logs/${RUN_ID}.log"

echo ""
echo "Log saved to: logs/${RUN_ID}.log"
echo "BPB line:"
grep "final_int6_sliding_window_exact\|final_int8_zlib_roundtrip_exact" "logs/${RUN_ID}.log" | tail -1
