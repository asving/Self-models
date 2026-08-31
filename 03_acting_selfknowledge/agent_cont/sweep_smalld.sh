#!/usr/bin/env bash
# Capacity-starve test: small d_model so shallow nets may FAIL to do the inference.
# Two variants packed onto 6 GPUs (1 depth each), depth {1,2,3,4,6,8}.
#   variant cont-square small-d  (d_model=$DM_SQ)
#   variant hmm na=2 small-d     (d_model=$DM_HMM)
# We run them SEQUENTIALLY (square first, hmm second) per GPU.
set -euo pipefail
cd "$HOME/self-models"
PY="$HOME/comp_icl/.venv/bin/python"
GPUS=${1:-2,3,4,5,6,7}; DM_SQ=${2:-16}; DM_HMM=${3:-24}
IFS=',' read -ra GA <<< "$GPUS"; ng=${#GA[@]}
LAYERS=(1 2 3 4 6 8)
mkdir -p logs runs
ts=$(date +%Y%m%d_%H%M%S)
for idx in "${!GA[@]}"; do
  gpu=${GA[$idx]}; L=${LAYERS[$idx]}
  csq="$PY agent_cont_hard.py --obs_nl square --nl_strength 1.0 --d_model $DM_SQ --n_head 4 --n_layer $L --steps 6000 --out runs/sqd${DM_SQ}_L$L"
  chm="$PY agent_hmm_hard.py --N 6 --stay 0.9 --n_alias 2 --sigma_e 0.3 --d_model $DM_HMM --n_head 4 --n_layer $L --L 50 --steps 6000 --out runs/hmmd${DM_HMM}_L$L"
  loga="logs/sqd${DM_SQ}_L${L}_${ts}.log"; logb="logs/hmmd${DM_HMM}_L${L}_${ts}.log"
  tmux kill-session -t "sd$gpu" 2>/dev/null || true
  tmux new-session -d -s "sd$gpu" \
    "CUDA_VISIBLE_DEVICES=$gpu $csq 2>&1 | tee $loga; CUDA_VISIBLE_DEVICES=$gpu $chm 2>&1 | tee $logb"
  echo "GPU $gpu: [sqd$DM_SQ L$L] then [hmmd$DM_HMM L$L]"
done
echo "ts=$ts"
