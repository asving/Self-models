#!/usr/bin/env bash
# Depth sweep for the HARD nonlinear self-model task.
# Usage: ./sweep_hard.sh <obs_nl> <nl_strength> <tag> <gpu0,gpu1,...>
# Launches layers {1,2,3,4,6,8} one per GPU (in given list) in tmux, each tee'd to logs/.
set -euo pipefail
cd "$HOME/self-models"
PY="$HOME/comp_icl/.venv/bin/python"
NL=${1:-square}; G=${2:-1.0}; TAG=${3:-sq1}; GPUS=${4:-2,3,4,5,6,7}
IFS=',' read -ra GA <<< "$GPUS"
LAYERS=(1 2 3 4 6 8)
mkdir -p logs runs
ts=$(date +%Y%m%d_%H%M%S)
for i in "${!LAYERS[@]}"; do
  L=${LAYERS[$i]}; GPU=${GA[$((i % ${#GA[@]}))]}
  out="runs/hard_${TAG}_L${L}"
  log="logs/hard_${TAG}_L${L}_${ts}.log"
  sess="hard_${TAG}_L${L}"
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$GPU $PY agent_cont_hard.py --obs_nl $NL --nl_strength $G \
       --n_layer $L --d_model 128 --n_head 4 --steps 5000 --out $out 2>&1 | tee $log"
  echo "launched L=$L on GPU $GPU -> $log (tmux: $sess)"
done
