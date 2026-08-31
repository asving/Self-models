#!/usr/bin/env bash
# Escalation: harder aliased-HMM settings + smaller d_model, to try to expose depth discrimination.
# Runs depth {1,2,3,4,6,8}. Pick ONE config via args; pack 6 depths onto 6 GPUs (1 each).
# Usage: bash sweep_hmm_escalate.sh <gpus> <N> <stay> <n_alias> <sigma_e> <d_model> <L> <tag>
set -euo pipefail
cd "$HOME/self-models"
PY="$HOME/comp_icl/.venv/bin/python"
GPUS=${1:-2,3,4,5,6,7}; N=${2:-9}; STAY=${3:-0.92}; NA=${4:-3}; SE=${5:-0.3}; DM=${6:-64}; L=${7:-60}; TAG=${8:-esc}
IFS=',' read -ra GA <<< "$GPUS"
LAYERS=(1 2 3 4 6 8)
mkdir -p logs runs
ts=$(date +%Y%m%d_%H%M%S)
for i in "${!LAYERS[@]}"; do
  NL=${LAYERS[$i]}; GPU=${GA[$(( i % ${#GA[@]} ))]}
  out="runs/hmm_${TAG}_L${NL}"; log="logs/hmm_${TAG}_L${NL}_${ts}.log"
  tmux kill-session -t "esc$GPU" 2>/dev/null || true
  tmux new-session -d -s "esc_${TAG}_L${NL}" \
    "CUDA_VISIBLE_DEVICES=$GPU $PY agent_hmm_hard.py --N $N --stay $STAY --n_alias $NA --sigma_e $SE \
       --d_model $DM --n_layer $NL --L $L --steps 6000 --out $out 2>&1 | tee $log"
  echo "GPU $GPU: hmm $TAG L=$NL (N=$N stay=$STAY na=$NA se=$SE d=$DM L=$L) -> $log"
done
echo "ts=$ts"
