#!/bin/bash
# Parameter Golf — experiment runner
#
# Usage:
#   bash run.sh bos-reset                        # our current best
#   bash run.sh performer-probe                  # Performer linear attention probe
#   bash run.sh frontier                         # accepted SOTA (PR #1019)
#
# Run modes:
#   (default)             4-GPU probe:  --gpus 4 --wallclock 300  → ~940 steps, 1 val reading, ~$1
#   --gpus 4              full 4-GPU:   --gpus 4 --wallclock 600  → ~1880 steps, ~$2
#   --gpus 8              submission:   --gpus 8 --wallclock 600  → ~5700 steps, ~$4
#
# Common flags:
#   --gpus N        number of GPUs
#   --wallclock N   wall clock cap in seconds (default 300 for probe, 600 for submit)
#   --seed N        random seed (default 1337)
#   --val N         val every N steps (auto-set based on wallclock)
#
# Env vars pass through to train script (e.g. LINEAR_ATTN_ENABLED=1)
#
# Results written to:
#   logs/<run_id>.log

set -euo pipefail

# ---- Defaults ----
MODE="${1:-baseline}"
shift || true

GPUS=4
ITERATIONS=20000
SEED=1337
WALLCLOCK=300.0   # 5 min probe by default — cheap directional signal
VAL_EVERY=""      # auto-set below if not specified

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)      GPUS="$2";      shift 2 ;;
        --iters)     ITERATIONS="$2"; shift 2 ;;
        --seed)      SEED="$2";      shift 2 ;;
        --wallclock) WALLCLOCK="$2"; shift 2 ;;
        --val)       VAL_EVERY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Auto-set val frequency based on wallclock ----
# Goal: get 1-2 val readings per run without wasting time
if [[ -z "$VAL_EVERY" ]]; then
    if   (( $(echo "$WALLCLOCK <= 0" | bc -l) )); then VAL_EVERY=500
    elif (( $(echo "$WALLCLOCK <= 300" | bc -l) )); then VAL_EVERY=250
    elif (( $(echo "$WALLCLOCK <= 600" | bc -l) )); then VAL_EVERY=500
    else VAL_EVERY=1000
    fi
fi

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
    bos-reset-slot)
        SCRIPT="$RECORDS/2026-03-25_11L_BOS-Reset_SLOT_XSA-all_GPTQ/train_gpt.py"
        ;;
    bos-reset-slot-gluv)
        SCRIPT="$RECORDS/2026-03-25_11L_BOS-Reset_SLOT_GLUV_XSA-all_GPTQ/train_gpt.py"
        ;;
    combined)
        SCRIPT="$RECORDS/2026-03-25_11L_BOS-Reset_LateEMA_XSA-all_GPTQ/train_gpt.py"
        ;;
    performer-probe)
        SCRIPT="$RECORDS/2026-03-29_11L_Performer_BOS-Reset_XSA-all_GPTQ/train_gpt.py"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: baseline, frontier, late-ema, bos-reset, bos-reset-slot, bos-reset-slot-gluv, combined, performer-probe"
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
echo "Seed:       $SEED"
echo "GPUs:       $GPUS"
echo "Wallclock:  ${WALLCLOCK}s"
echo "Val every:  ${VAL_EVERY} steps"
echo "========================================"
echo ""

# ---- Launch ----
RUN_ID=$RUN_ID \
SEED=$SEED \
ITERATIONS=$ITERATIONS \
MAX_WALLCLOCK_SECONDS=$WALLCLOCK \
VAL_LOSS_EVERY=$VAL_EVERY \
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
grep "final_slot_ttt_exact\|final_int6_sliding_window_exact\|final_int8_zlib_roundtrip_exact" "logs/${RUN_ID}.log" | tail -1
