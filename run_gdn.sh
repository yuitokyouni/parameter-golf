#!/bin/bash
# exp17-19: GDN (Gated DeltaNet) Weight-Tied variations
# Requires: pip install flash-linear-attention
# Target: 1x RTX 4090 (24GB), 1800s wallclock

set -euo pipefail

pip install flash-linear-attention --break-system-packages 2>/dev/null || pip install flash-linear-attention

COMMON="MAX_WALLCLOCK_SECONDS=1800 VAL_LOSS_EVERY=200 EVAL_STRIDE=64 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024"

echo "=== exp17: Full GDN 6L×2 d=512 ==="
eval $COMMON RUN_ID=exp17_full_gdn MODEL_DIM=512 python train_exp17.py 2>&1 | tee logs/exp17.log

echo "=== exp18: Hybrid 5GDN+1SWA 6L×2 d=512 ==="
eval $COMMON RUN_ID=exp18_hybrid MODEL_DIM=512 python train_exp18.py 2>&1 | tee logs/exp18.log

echo "=== exp19: Hybrid 5GDN+1SWA 6L×2 d=640 ==="
eval $COMMON RUN_ID=exp19_hybrid_wide MODEL_DIM=640 python train_exp19.py 2>&1 | tee logs/exp19.log

git add -A && git commit -m "exp17-19: GDN Weight-Tied variations" && git push origin baseline-try
echo "=== ALL DONE ==="
