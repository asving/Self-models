#!/bin/bash
# Pure-RL variants (no prediction loss): c = ENT 0.05, d = ENT 0
PY=/Users/asvin/astera-claude/group-belief/.venv/bin/python
for cfg in "c 0.05" "d 0"; do
  set -- $cfg
  $PY post.py 4000 0 $1 $2 2>&1 | grep --line-buffered -v -e Warn -e stddev > post_$1.log
  $PY analyze.py $1 2>&1 | grep -v -e Warn -e stddev -e "A = np" > analyze_$1.log
done
