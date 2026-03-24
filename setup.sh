#!/bin/bash
# Parameter Golf — environment setup
# Run once on a fresh cloud instance. Safe to re-run (idempotent).
#
# Usage:
#   bash setup.sh                  # 2 training shards (~3GB, fast)
#   TRAIN_SHARDS=10 bash setup.sh  # more data for longer runs
#   SKIP_DATA=1 bash setup.sh      # skip data download (already done)

set -euo pipefail

REPO="akrausscs/parameter-golf"
TRAIN_SHARDS="${TRAIN_SHARDS:-2}"
SKIP_DATA="${SKIP_DATA:-0}"

# ---- Detect workspace ----
if   [ -d /workspace ]; then WORKDIR=/workspace      # RunPod
elif [ -d /home/user  ]; then WORKDIR=/home/user      # Lambda Labs
else                          WORKDIR=$HOME
fi
echo "Using workdir: $WORKDIR"
cd "$WORKDIR"

# ---- Clone repo (or pull if already present) ----
if [ ! -d parameter-golf ]; then
    echo "Cloning repo..."
    git clone "https://github.com/$REPO.git"
else
    echo "Repo already present, pulling latest..."
    cd parameter-golf && git pull && cd ..
fi
cd parameter-golf

# ---- Python deps ----
echo "Installing Python dependencies..."
pip install -q sentencepiece huggingface-hub datasets tqdm

# ---- Data download ----
if [ "$SKIP_DATA" = "1" ]; then
    echo "Skipping data download (SKIP_DATA=1)"
else
    echo "Downloading data ($TRAIN_SHARDS training shards)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"
fi

# ---- Verify GPU ----
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, BF16 supported: {torch.cuda.is_bf16_supported()}')"

# ---- Smoke test (10 steps, no val) ----
echo ""
echo "Running smoke test (10 steps)..."
RUN_ID=smoke \
ITERATIONS=10 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tail -5

echo ""
echo "Setup complete. Run experiments with: bash run.sh"
