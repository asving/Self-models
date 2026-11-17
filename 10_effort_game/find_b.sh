#!/bin/bash
# Search for run b's unrecorded (LAM, ENT) by matching its step-250 eval:
# target reward 1.0307, wmax 0.402, corr 0.479
PY=/Users/asvin/astera-claude/group-belief/.venv/bin/python
for cfg in "1.0 0.05" "1.0 0.07"; do
  set -- $cfg
  tag="bsearch_${1}_${2}"
  echo "LAM=$1 ENT=$2"
  $PY post.py 250 $1 $tag $2 2>&1 | grep '"step": 250'
done
