#!/bin/bash
cd ~/self-models
# wait for the big beta=0 base to finish
while [ ! -f rps_runs/rpsbig_b0.0.pt ]; do sleep 30; done
sleep 10; echo "=== base ready; starting graded anneal on GPU0 ==="
prev=rps_runs/rpsbig_b0.0.pt
for b in 0.1 0.2 0.3 0.4 0.5; do
  echo "--- anneal stage beta=$b (init $prev) ---"
  CUDA_VISIBLE_DEVICES=0 ~/comp_icl/.venv/bin/python rps_im.py --out rps_runs/rpsanneal_b$b --beta $b --per_traj \
    --init $prev --n_layer 6 --d_model 256 --n_head 8 --steps 2500 --batch 256 --lr 3e-4 --T 40 --eval_every 250 \
    2>&1 | tee logs/rpsanneal_b$b.log
  prev=rps_runs/rpsanneal_b$b.pt
done
echo "=== anneal chain done ==="
