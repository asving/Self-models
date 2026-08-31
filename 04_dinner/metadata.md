# 04_dinner — the planning-gap game (2026-07-03)
Pretrain->RL on a 3-factor dinner-party world with deadlines; certified
planning gap (greedy 1.10 << packer 1.92 << backtimed 2.03). Headline: RL
snaps to the pretraining packer template in ~150 steps (selection-then-
fusion); deadline arithmetic generalizes OOD; zero rate-adaptive planning
(order-agreement .494 = chance). Files: dinner.py (env+training),
dinner_eval.py, dinner_probe.py (rate-marginalized filter + do-tests);
dinner_runs/v1. Memory: dinner-party-v1-result.
