#!/bin/bash
set -o pipefail
mkdir -p logs
# rank0 writes logs/{run_id}.txt (code + env + step timings); tee captures the torchrun console.
SCRIPT="${1:-train_gpt.py}"
DISABLE_FP8=1 torchrun --standalone --nproc_per_node=8 "$SCRIPT" 2>&1 | tee "logs/run_${SCRIPT%.py}_$(date +%Y%m%d_%H%M%S).log"
