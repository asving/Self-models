#!/bin/bash
cd ~/self-models
# wait until all 6 runs print "done ->" in their logs
while true; do
  ndone=$(grep -l "done ->" logs/rpsbig_b*_*.log 2>/dev/null | wc -l)
  [ "$ndone" -ge 6 ] && break
  sleep 30
done
sleep 5
echo "=== ALL 6 BIG RUNS DONE; evaluating ==="
~/comp_icl/.venv/bin/python eval_big.py
