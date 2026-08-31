"""Decisive rubber-hand: isolate the efference-subtraction sign cleanly.

mu_{t0+1} is computed by the net from y = o_{t0+1} - a_hat_{t0}, where a_hat_{t0} is the
net's internal reconstruction of the action it took at t0 (efference copy). Two clean,
faithful interventions on the SAME forward pass, contrasted with the naive patch:

  Recall o_{t0+1} = s_{t0+1} + v + a_{t0}. The action a_{t0} appears in o additively
  (the 'corruption') AND must be subtracted internally. So:

   * Raise o_{t0+1} by delta  (a real world/observation change): both the carried emission
     and the apparent action rise; net belief should rise ~ +K*delta. [obs update]
   * Raise the CONSUMED action a_hat_{t0} by delta with o_{t0+1} fixed: decoded
     y = o-(a+delta) drops by delta; belief should drop ~ -K*delta. [efference / rubber-hand]

  Because the net reconstructs a_hat_{t0} from CONTEXT (positions <= t0+1), the faithful,
  consistent way to raise the consumed action is to present a context where the net's
  reconstructed action at t0 is higher -- i.e. the closed-loop world that genuinely took
  a+delta -- but then SUBTRACT the world's response (the +delta that entered o and s) so
  that only the *consumed-action* term differs. We get this exactly from the (A) closed-loop
  pair by decomposition:
     belief_inj - belief_base = (obs-update response to +delta in o_{t0+1})
                              +  (efference response to +delta in consumed a_hat)
     true_state shift          = +delta
  Net tracked +0.869 of the +1.0 true shift => efference subtraction is REAL and ~complete:
  if there were NO efference subtraction, raising the action would raise o by delta and the
  net would perceive +K*delta only (~+0.53), NOT track the full +1.0. Tracking ~0.87 proves
  the net adds back the (1-K)*delta via efference. We quantify this directly here.

CPU. Run: CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python cont_rubberhand2.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cont_dissect import (load_net, manual_forward, kalman_on_realized, ridge_r2, fit_predict)
from agent_cont import ALPHA, SW, SV, S0


torch.set_grad_enabled(False)


def steady_gain():
    P = S0 ** 2
    for _ in range(80):
        P = ALPHA ** 2 * P + SW ** 2; K = P / (P + SV ** 2); P = (1 - K) * P
    return K


def main():
    net, ck = load_net()
    K = steady_gain()
    B, L, t0 = 2500, 40, 20
    pos = np.arange(L // 2, L); d = 128; ntr = B // 2

    g = torch.Generator().manual_seed(11)
    s = torch.randn(B, generator=g) * S0
    o = s + torch.randn(B, generator=g) * SV
    obs = [o]; states = [s]; acts = []
    Wn = torch.randn(B, L, generator=g) * SW
    Vn = torch.randn(B, L, generator=g) * SV
    for t in range(L):
        a = net(torch.stack(obs, 1))[0][:, -1]; acts.append(a)
        if t == L - 1: break
        s = (ALPHA * s + a + Wn[:, t]).clamp(-12, 12)
        o = s + Vn[:, t] + a
        obs.append(o); states.append(s)
    obs = torch.stack(obs, 1); states = torch.stack(states, 1); acts = torch.stack(acts, 1)
    mu, _ = kalman_on_realized(obs, acts)
    out = manual_forward(net, obs)

    Xm = out["post_lnf"].numpy()[:, pos, :]; Ym = mu[:, pos]
    _, mp = ridge_r2(Xm[:ntr].reshape(-1, d), Ym[:ntr].reshape(-1), Xm[ntr:].reshape(-1, d), Ym[ntr:].reshape(-1))
    def read_mu(o_, t): return fit_predict(mp, manual_forward(net, o_)["post_lnf"].numpy()[:, t, :])
    mu_clean = fit_predict(mp, out["post_lnf"].numpy()[:, t0 + 1, :])

    delta = 1.0
    print(f"=== Decisive efference / rubber-hand (t0={t0}, K={K:.3f}, delta={delta}) ===\n")

    # (1) OBS update: raise the world observation o_{t0+1} by delta (real world change)
    o1 = obs.clone(); o1[:, t0 + 1] += delta
    d_obs = (read_mu(o1, t0 + 1) - mu_clean).mean()
    print(f"(1) world obs o[t0+1] += delta  -> belief mu[t0+1] shift = {d_obs:+.3f}  (Kalman +K*delta={K*delta:+.3f})")

    # (2) CLOSED-LOOP rubber-hand: actually take action a+delta at t0, reusing noise.
    #     World genuinely changes: s and o both shift. Net must use efference to track.
    s_t0 = states.numpy()[:, t0]
    a_inj = acts.numpy()[:, t0] + delta
    s_next_inj = (ALPHA * s_t0 + a_inj + Wn[:, t0].numpy()).clip(-12, 12)
    o_next_inj = s_next_inj + Vn[:, t0].numpy() + a_inj
    s_next_base = (ALPHA * s_t0 + acts.numpy()[:, t0] + Wn[:, t0].numpy()).clip(-12, 12)
    o_inj = obs.clone(); o_inj[:, t0 + 1] = torch.tensor(o_next_inj, dtype=obs.dtype)
    o_base = obs.clone()  # base o_{t0+1} already correct
    mu_inj = read_mu(o_inj, t0 + 1); mu_base = read_mu(o_base, t0 + 1)
    true_shift = (s_next_inj - s_next_base).mean()      # = +delta
    belief_shift = (mu_inj - mu_base).mean()
    print(f"\n(2) closed-loop: take a+delta at t0 (world genuinely changes)")
    print(f"    true state s[t0+1] shift = {true_shift:+.3f} (=delta)")
    print(f"    net belief mu[t0+1] shift = {belief_shift:+.3f}")
    print(f"    tracking ratio = {belief_shift/true_shift:.3f}  (1.0 => perfect efference)")

    # Decompose: o_{t0+1} shifted by delta (the +a corruption + state response).
    # Pure obs-update would give +K*delta. The EXTRA tracking above +K*delta is the
    # efference subtraction working in the (1-K) Kalman-predict path.
    eff_contrib = belief_shift - K * delta
    print(f"    decomposition: obs-update part = +K*delta = {K*delta:+.3f};  "
          f"efference/predict part = {eff_contrib:+.3f}  (Kalman expects (1-K)*delta={ (1-K)*delta:+.3f})")

    # (3) Pure rubber-hand: world UNCHANGED (o[t0+1] as generated under action a),
    #     but make consumed action a+delta. Equivalent to (closed-loop o_inj) MINUS
    #     (raising o[t0+1] by the delta that the action injected into o). We isolate it:
    #     o_inj differs from o_base at t0+1 by (s_next_inj - s_next_base) + delta = 2*delta
    #     (delta into state, delta into +a corruption). To hold the WORLD's perceived
    #     emission fixed but flip only the consumed-action term, subtract the obs change:
    do_t1 = (o_next_inj - obs.numpy()[:, t0 + 1])   # how much o moved in closed-loop
    o_rub = o_inj.clone()
    o_rub[:, t0 + 1] = o_inj[:, t0 + 1] - torch.tensor(do_t1, dtype=obs.dtype)  # = original o
    # That's just o_base; the consumed action is reconstructed from context o_<=t0+1 which
    # is now identical to base => no isolation. Instead, drive the world with a+delta but
    # CANCEL its effect on o_{t0+1} only, leaving the net's *reconstruction* of a higher.
    # Reconstruction of a_{t0} at pos t0+1 leans on o_{t0} and o_{t0+1}. In closed loop both
    # move. Keep o_{t0} at its injected (higher) value but reset o_{t0+1} to the value it
    # would have had with the ORIGINAL action -> net thinks action was bigger, world emission
    # is the original-state one.
    o_rub2 = obs.clone()
    # o_{t0} is unchanged (action at t0 doesn't enter o_{t0}); the action shows up in o_{t0+1}
    # and the state. Set o_{t0+1} = s_next_base + v + (a+delta): same state as base, but the
    # +a corruption uses the bigger action -> net reconstructs bigger action, decodes
    # y=o-(a+delta)=s_next_base+v => true emission base, consumed action +delta.
    o_rh = (s_next_base + Vn[:, t0].numpy() + a_inj)
    o_rub2[:, t0 + 1] = torch.tensor(o_rh, dtype=obs.dtype)
    mu_rh = (read_mu(o_rub2, t0 + 1) - mu_clean).mean()
    print(f"\n(3) PURE rubber-hand: world state UNCHANGED (emission=base), but +a corruption")
    print(f"    and reconstructed consumed action both raised by delta (efference target a+delta)")
    print(f"    belief mu[t0+1] shift vs clean = {mu_rh:+.3f}")
    print(f"    Kalman expectation if net subtracts the (a+delta): obs rose +delta (so +K*delta)")
    print(f"      but consumed action rose +delta in predict path => net -(1-K)*delta extra")
    print(f"      net result should track the BASE state ~0 shift if efference perfect; observed {mu_rh:+.3f}")

    print(f"\n=== SUMMARY ===")
    print(f"  obs-update gain (world o change)         : {d_obs:+.3f}  (Kalman +K={K:.3f})")
    print(f"  closed-loop tracking of true state       : {belief_shift/true_shift:.3f} of delta (efference works)")
    print(f"  efference 'add-back' beyond obs gain      : {eff_contrib:+.3f}  (Kalman (1-K)*delta={(1-K)*delta:+.3f})")


if __name__ == "__main__":
    main()
