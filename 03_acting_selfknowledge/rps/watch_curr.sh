#!/bin/bash
cd ~/self-models
while [ "$(ls rps_runs/rpscurr_b*.pt 2>/dev/null | wc -l)" -lt 6 ]; do sleep 30; done
sleep 5; echo "=== ALL 6 CURRICULUM RUNS DONE ==="
CUDA_VISIBLE_DEVICES=7 ~/comp_icl/.venv/bin/python cmp_curr.py
