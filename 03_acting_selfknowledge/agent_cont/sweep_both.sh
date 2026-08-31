#!/usr/bin/env bash
# Launch BOTH depth sweeps {1,2,3,4,6,8} packed onto free GPUs, 2 runs per GPU (sequential per GPU).
# Variant sq: cont nonlinear-obs square (G=$SQG). Variant hmm: hard aliased HMM (n_alias=2).
set -euo pipefail
cd "$HOME/self-models"
PY="$HOME/comp_icl/.venv/bin/python"
GPUS=${1:-2,3,4,5,6,7}; SQG=${2:-1.0}
IFS=',' read -ra GA <<< "$GPUS"; ng=${#GA[@]}
LAYERS=(1 2 3 4 6 8)
mkdir -p logs runs
ts=$(date +%Y%m%d_%H%M%S)
declare -a JOBS
for L in "${LAYERS[@]}"; do
  JOBS+=("sq|$L|$PY agent_cont_hard.py --obs_nl square --nl_strength $SQG --n_layer $L --steps 5000 --out runs/hard_sq_L$L")
done
for L in "${LAYERS[@]}"; do
  JOBS+=("hmm|$L|$PY agent_hmm_hard.py --N 6 --stay 0.9 --n_alias 2 --sigma_e 0.3 --n_layer $L --steps 5000 --out runs/hmm_L$L")
done
for idx in "${!GA[@]}"; do
  gpu=${GA[$idx]}
  IFS='|' read -r ta La ca <<< "${JOBS[$idx]}"
  IFS='|' read -r tb Lb cb <<< "${JOBS[$(( idx + ng ))]}"
  loga="logs/${ta}_L${La}_${ts}.log"; logb="logs/${tb}_L${Lb}_${ts}.log"
  tmux kill-session -t "gpu$gpu" 2>/dev/null || true
  tmux new-session -d -s "gpu$gpu" \
    "CUDA_VISIBLE_DEVICES=$gpu $ca 2>&1 | tee $loga; CUDA_VISIBLE_DEVICES=$gpu $cb 2>&1 | tee $logb"
  echo "GPU $gpu: [$ta L$La -> $loga] then [$tb L$Lb -> $logb]"
done
echo "ts=$ts"
