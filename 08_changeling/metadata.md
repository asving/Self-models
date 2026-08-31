# 08_changeling — who-am-I: per-episode embodiment of one of two coupled actors

Purpose: force a *deliberative* (in-context, watchable) self/other
representation by making the self/other boundary an episode-level latent.
Two coupled ring HMMs; each episode the network IS one of them (hidden coin
iota); composite objective (running reward + KL anchor + prediction loss on
non-emitted tokens) forces each channel's output to depend on the identity
posterior lam (reference form: the Bayes mixture out = lam*plan +
(1-lam)*predict), so the self-localization posterior is legible in the
output layer.

Central lemma (proved + machine-tested): at zero tilt the record's law is
IDENTICAL under both identities, however different the actors — all identity
evidence is goal-directed deviation ("you find out who you are only by
perturbing the stream"). Corollary measured as the delusion gap: embodiment
severs a channel's reports from its true state without changing the record.

Status 2026-08-31: phase 0 (exact oracle + world selection) DONE, CPU-only.
Selected stage-2 world: n=6 T=64 (c_o,c_s)=(0.6,0.35) eps=.05 q0=.9 d=2
running-reward rho=8 — premium G_occ=.169, collapse at t~14/64, 98% correct.
Measured along the way: terminal-only reward concentrates identity evidence
at the deadline (urgency-gating law, proposal Fig.1, as WHEN selfhood is
learnable); slack-vs-control trade-off (privileged self-knowledge needs
dynamics slack; token-determined dynamics kill the delusion gap).

Inventory: DESIGN_changeling_v0_worldselect.md (design + preregistration +
amendments & outcomes) | worlds.py (kernels, h/M/N tables — single source of
truth) | oracle.py (4-belief filter bank + mixture policy + lambda) |
validate.py (V1-V4, all pass) | sweep.py (selection sweeps) | results/
(validation + 3 sweep JSONs + winner npz) | figs/ | logs/.

Run everything with cwd = this folder, ~/comp_icl/.venv/bin/python. All CPU.
Next: stage-2 RNN design doc (pretrain on base joint law; post-train with
iota coin, masked prediction loss, REINFORCE+KL); lambda probes vs this
oracle; wiggle-bonus and reward-on-belief variants queued.
