#!/bin/bash
set -o pipefail
mkdir -p logs
# rank0 writes logs/{run_id}.txt (code + env + step timings); tee captures the torchrun console.
DISABLE_FP8=1 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "logs/run_$(date +%Y%m%d_%H%M%S).log"
