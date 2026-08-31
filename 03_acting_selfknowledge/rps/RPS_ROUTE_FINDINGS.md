# Recompute vs message-pass: how the RPS nets access their own previous action (2026-07-03)

**Question.** In the entangled-observation games (`o_t=(a_t-b_t)%3`), decoding the opponent requires
the net's own action `a_t`, whose policy `p_t` was computed at the *previous* position. Does the info
reach position `tau=t+1` by (M) **message-passing** — attention reads the previous position's
computed action — or (R) **recomputation** — `tau`'s own circuit re-derives it from context — or
(H) **hardcoding** — it isn't context-dependent at all?

**Verdict: neither a stored efference value nor raw-token recompute-at-`tau` — the previous action
is RE-DERIVED at the read position from a POSITION-DISTRIBUTED aggregate of partial computations
(pooled evidence/counts), plus a hardcoded opening.** There is no action *record* anywhere: the
"efference copy" is the re-derivable policy mode, and the net re-derives it wherever it needs it.
Same verdict in the 2L net (where message-passing is provably infeasible) and the 6L net (where it
is feasible but unused). Analysis: `rps_route.py` (mech/probe/train/attn/patch); full logs in
`logs/rps_route_main.log`, `logs/rps_route_pretrain_bias.log`. Nets: `rps_b0.0` (2L d64),
`rpsbig_b0.0` (6L d256), replication `pretrain_bias` (2L d64). All patching done by recomputing ONLY
row `tau` with per-layer, per-position K/V sources chosen from clean/corrupt cached contexts
(`row_forward`; validated to 2e-6 against the full forward).

## 1. Presence (the easy bar) — yes, with two caveats
- `m_t = argmax p_t` decodes from lnf@`tau` at **0.95–0.99** in all round buckets, vs round-only
  baseline 0.36–0.54 (mid/late). On the **discriminating subset** (`m_t != m_{t+1}`, probe *fit on
  the subset* so it can't inherit the current-policy direction): **0.86** (2L) / **0.94** (6L) vs
  0.64 baseline — a previous-action representation distinct from the current policy exists.
- **Caveat 1 (what the info is):** forced-random control (actions uniform, never fed back): the
  residual decodes the policy **mode at 0.92–0.97** but the **realized sample at chance** (0.33),
  though Bayes-from-`o` could reach 0.70. The self-knowledge is the *reconstructable intention*,
  never the draw — exactly mechanism B from `wb_pretrain.py`, now at the representation level.
- **Caveat 2 (training dynamics):** the probe is already at 0.87 (discrim) on a *random-init* net —
  the mode of a near-linear random policy is trivially decodable from shared context. Signal
  strength does not monotonically "grow with training" (0.87 → 0.96 @1k → 0.79 @2k → 0.86 @4k);
  presence-probing cannot distinguish "represents its past action" from "past action is a simple
  function of shared context". This is why the causal test below is the real bar.

## 2. Attention probes (the tricky middle bar) — no wire, info everywhere
- **No previous-token head exists.** Every head's attention mass on `tau-1` is ≤ 0.09; all heads
  pool broadly (eff# 13–20 of ~20 positions) — accumulator profiles, same as `wb_pretrain`.
- **No single head output carries the previous action on discriminating events** (all heads
  0.35–0.45 vs 0.64 baseline; high all-rounds accs are the `m_t ≈ m_{t+1}` degeneracy).
- Aggregation *creates* the signal: in the 6L net, att1's output at `tau` carries `m_t` at 0.96
  while the individual x1 rows it reads carry only 0.44–0.59. In the 2L net lnf@`tau` (0.97) exceeds
  the best single readable message (x1@`t` = 0.80) — more info at the destination than any source.

## 3. K/V-source patching (the hard bar) — the decisive causal result
Setup: recompute only row `tau`; for each attention layer independently, past positions' K/V come
from the clean or a corrupt context (episode-swap, or 3-token re-randomize); `tau`'s own token stays
clean; readouts = probe-decoded `a_hat`, its induced decode `b_hat` (cyclic consistency stays
0.5–0.7 vs 0.33 chance under all mixed conditions), and the behavioral policy. Sanity: all-clean
reproduces clean exactly; all-corrupt flips `a_hat` to the corrupt mode at 0.96–0.99.

- **The τ-1 "message wire" carries nothing.** Patch position `tau-1` completely (all layers, or its
  computed layers only): `a_hat` stays clean at 0.91–0.98 for `tau`≥12 in every net. This kills
  single-wire message-passing — including in the 6L net where it was *feasible* (`m_t` is fully
  present at its own position from x2 up, 0.96–0.99).
- **Raw-token K/V alone lose control as evidence accumulates.** 2L `XC` (tokens corrupt, computed
  clean): `a_hat` stays clean 0.75/0.86/0.90 at `tau`=12/20/32; 6L `XCCCCC`: 0.98 stays clean. At
  `tau`=6 the token path still carries ~half (0.49/0.46 split) — the duplicate-circuit-at-`tau`
  route operates early, before the pipeline has content.
- **The carrier is the computed content of MANY positions.** Corrupting all past positions'
  computed K/V flips `a_hat` (2L `CX`: 0.75/0.84/0.90; 6L: layer-boundary sweep shows the causal
  load at x3–x5). Position-set decomposition: effect scales with the *share* of positions patched,
  old positions ≥ share, recent positions ≈ nothing (2L `tau`=32: recent-3 → 0.02, old-25 → 0.87;
  6L `tau`=32: recent-3 → 0.02, old-25 → 0.94). Count/evidence aggregation predicts exactly this;
  reading any position's precomputed policy value predicts the opposite (recent/pivotal dominance).

## 4. The mechanism, stated positively
Each position `j` computes shallow local evidence (block-0: pooled token histogram → per-position
evidence components in x_{1..}; deeper layers refine it). The read position aggregates these
broadly through its accumulator heads — over rows `< tau`, i.e. **the aggregate naturally excludes
the current token: this is the "duplicated circuit fed by partial attention"** — and re-applies the
policy readout (BR of aggregated evidence) to obtain "what I did", in parallel with computing the
current policy (same aggregate + current-token correction). Previous-action and current-policy
coexist as two readouts of one belief state, separated by one round of evidence.

**The deepest form of the finding: the net never maintains an action *variable* at all.** Action ≡
BR(belief), so "my previous move" is re-derived from the belief aggregate wherever needed. In the
swap-patch this is visible as `a_hat`, `b_hat`-decode, and policy all moving together with the
belief-carrying K/V — there is no separate thing to desync (choice-slot fusion, in the RPS regime
where the fusion is *correct*: after identification, action is a deterministic function of belief).

### 4b. Dimension cost of the parallel readout (`dims` subcommand)
Does re-deriving the previous action alongside the current policy require duplicating circuitry in
extra dimensions (the inputs differ: aggregate-without vs aggregate-with the current token)? **No —
the measured private footprint is ~1–2 dims:**
- The prev- and curr-mode probe subspaces at lnf@`tau` (fit on the discriminating subset) share
  their leading direction exactly (cos principal angles 2L: [1.0, .86, .30]; 6L: [1.0, .83, .62]) —
  two readouts of one shared belief representation, one partially rotated.
- Projecting out the entire 6-dim current-policy block (act_head + curr-probe) barely dents the
  prev decode (2L: 0.855→0.843; 6L: 0.945→0.934), and the surviving private signal saturates at
  top-1–2 dims (0.82/0.88 with ONE dimension).
- The architecture supplies the two different inputs for free in naturally disjoint subspaces:
  attention output at `tau` (self-weight 0.01–0.06) IS the aggregate-through-t−1, while `o_t`
  arrives via the skip connection in the embedding subspace — no demultiplexing needed.
- What IS duplicated is the readout *function* (BR applied mid-stack for `m_t`, at act_head for
  `p_{t+1}`) — separate weights, but rank-≤3 each, not a pipeline copy. And the `m_t` application
  is not an optional extra: it is a required INTERMEDIATE of the current-policy update (`o_t`'s
  evidence cannot be attributed without `m_t`); keeping it to lnf is free (residuals persist).
  Caveat: linear probes cannot certify the serial order (a probe can finish a linear update
  itself — prev/curr appear together stream-by-stream); the intermediate role rests on the
  architecture argument + `wb_mech` EXP2 (read-position move-patch flips the final policy).
So the dims-vs-layers trade is real but cheap here: the residual must hold pre- and post-update
answers simultaneously (an RNN would overwrite), and that bill is a couple of dimensions because
only the low-rank answer forks, never the computation trunk.

## 5. RPS-specific caveats (why this task can't force an efference channel)
As suspected: (i) the opening is 100% hardcoded (H) — identification-phase self-knowledge is free;
(ii) after identification `m_t ≈ m_{t+1}` (switch rate 3%→0.3%), so the "previous action" is almost
the current policy — the discriminating events are rare, early, and near decision boundaries; (iii)
the policy is a deterministic function of observable history, so re-derivation is always available
and *cheaper than storage* (see §6). A task where the action is NOT a function of the token prefix
(private randomness, or an externally-supplied action the net must ingest) removes the re-derivation
option and is the right place to force a genuine stored efference copy — cf. the private-bits /
notes-to-self designs (`selfmodel-rng-bits-design`, `bounded-self-channel-design`).

## 6. Dimensions-vs-layers tension, resolved empirically here
Message-passing costs **layers**: the value must be finished at some depth k at position t, moved by
one attention layer, consumed above it — and in a 2L net with the policy finished only post-block-1,
that budget doesn't exist (feasibility probe: x1@t carries only 0.80 of `m_t`; lnf is unreadable).
Recomputation costs **dimensions**: the residual at `tau` holds both the pre-update readout (prev
action) and post-update policy — but re-uses the SAME aggregation circuitry with a shifted window,
so the duplicate is cheap in weights. The nets chose to spend dimensions in both regimes — even the
6L net, which had the layer budget (`m_t` readable from x2@t), still re-derives from the distributed
aggregate. Plausibly because the aggregation pipeline must exist anyway (for the belief), making the
re-derivation marginal cost ~zero, while a dedicated wire would need a previous-token head that has
no other use. Consistent with `depth-recurrence-feedforward`: the parallel shortcut is taken
whenever the per-step evidence computation is shallow — here not because the filter forgets (counts
accumulate) but because counting is **order-free** (commutative aggregation ⇒ position-parallel).
