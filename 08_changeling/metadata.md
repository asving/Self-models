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
Selected stage-2 world: n=6 T=32 (c_o,c_s)=(0.6,0.35) eps=.05 q0=.9 d=2
running-reward rho=8 — premium G_occ=.156, collapse at t~14 (horizon-
independent), 93% correct side; horizon.py measured T=32 as "just enough"
(identify ~14 + herd ~15, overlapping; T=16 starves the identity incentive).
Measured along the way: terminal-only reward concentrates identity evidence
at the deadline (urgency-gating law, proposal Fig.1, as WHEN selfhood is
learnable); slack-vs-control trade-off (privileged self-knowledge needs
dynamics slack; token-determined dynamics kill the delusion gap).

Inventory: DESIGN_changeling_v0_worldselect.md (design + preregistration +
amendments & outcomes) | worlds.py (kernels, h/M/N tables — single source of
truth) | oracle.py (4-belief filter bank + mixture policy + lambda) |
validate.py (V1-V4, all pass) | sweep.py (selection sweeps) | results/
(validation + 3 sweep JSONs + winner npz) | figs/ | logs/.

Run everything with cwd = this folder, ~/comp_icl/.venv/bin/python. Phase 0
all CPU; stage 2 (RNN) single GPU, minutes.

Stage 2 DONE (2026-08-31, seed 0, DESIGN_changeling_v1_rnn.md + measured
outcomes inside): pretrain hits exact filter floor; three-stage curriculum
(pretrain -> flag-given oracle distillation+DAgger -> flag-hidden
REINFORCE+KL+forecast-CE) takes closed-loop occupancy .355 -> .687 (above
the myopic informed oracle .511), and mechanism.py shows genuine
identification with an EXPONENTIAL policy — "everything is me until proven
otherwise": per-channel plan coefficient .99 (self) vs .34 (other), the
other-channel claim decaying 1.0 -> .08 within the episode. v1.0's two
training pathologies (distillation compounding; advantage normalization
shrinking effective rho) documented in the doc. Hidden states saved for the
probe/steering session. Next: hidden-state lambda probe + body-swap
steering; more seeds; wiggle-bonus; reward-on-belief; correlated-actors.
