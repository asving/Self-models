"""HARDER self-model task via a HARD-TO-DECODE DISCRETE HMM latent (complementary to the nonlinear-
observation route in agent_cont_hard.py). Same self-model structure: smooth continuous action that
additively corrupts the next observation (efference copy), closed loop, regression reward,
observations-only input. The difficulty comes from the LATENT being a slow-mixing, emission-ALIASED
HMM, so a single observation is uninformative and the state can only be pinned by integrating a long
sequence => belief integration + action-conditioning => (hypothesis) depth helps.

Env:
  z_t  in {0..N-1}: discrete hidden state, transition matrix T (slow mixing).
  Each state has a scalar 'value' r(z) in R (a readout embedding of the state). The TARGET the action
  should track is m_t = E[r(z_t) | o_{0:t}] (posterior mean of the readout) -- a SMOOTH readout
  (no argmax), so no collapse confound.
  Emission: each state emits a continuous obs mean μ(z); ALIASED => several states share (nearly) the
  same μ, so one obs can't identify z.  e_t = μ(z_t) + ε,  ε~N(0,σe²).
  o_t = e_t + a_{t-1}    (efference copy: action additively corrupts the NEXT observation).
  Action a_t = net output; reward = −(a_t − m_t)²  (track posterior mean of the readout).
  Closed loop: a_t biases the transition? We keep the action's effect on the latent BENIGN and known:
  the action shifts the EMISSION (pure efference copy) and optionally nudges transition temperature.
  Default: action only corrupts observation (clean efference-copy closed loop via the obs the net must
  de-corrupt; the latent evolves exogenously). This keeps the floor on-policy-invariant (the latent
  trajectory does not depend on the action), so 'excess MSE' is a clean measure.

Floor: EXACT discrete forward Bayes filter (cheap): b_{t} over N states; predict b T; update by
  Gaussian emission likelihood of y_t = o_t − a_{t-1} about μ(z); action a = sum_z b(z) r(z).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import Block
BASE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# HMM definition: N states, slow-mixing transition, ALIASED emissions, readout values.
# ---------------------------------------------------------------------------
def make_hmm(N=6, stay=0.85, n_alias=2, sigma_e=0.4, seed=0):
    """N states on a ring (slow mixing: stay w.p. `stay`, else step to a neighbour).
    Emission means μ(z): only `N//n_alias` DISTINCT values, assigned so that states far apart on the
    ring share an emission (=> a single obs is ambiguous between aliased states; only the transition
    structure + history disambiguates). Readout r(z): distinct per state (so tracking needs the true z)."""
    rng = np.random.default_rng(seed)
    T = np.full((N, N), 0.0)
    for i in range(N):
        T[i, i] = stay
        T[i, (i - 1) % N] = (1 - stay) / 2
        T[i, (i + 1) % N] = (1 - stay) / 2
    n_distinct = max(1, N // n_alias)
    base_mu = np.linspace(-1.5, 1.5, n_distinct)
    mu = np.array([base_mu[i % n_distinct] for i in range(N)])      # aliased: i and i+n_distinct share μ
    r = np.linspace(-1.5, 1.5, N)                                   # distinct readout per state
    pi0 = np.full(N, 1.0 / N)
    return dict(N=N, T=T, mu=mu, r=r, pi0=pi0, sigma_e=float(sigma_e))


class ContAgent(nn.Module):
    def __init__(self, d, nl, nh, L):
        super().__init__()
        self.in_proj = nn.Linear(1, d)
        self.pos = nn.Embedding(L + 2, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.act_head = nn.Linear(d, 1)
        self.obs_head = nn.Linear(d, 1)

    def forward(self, o):
        B, L = o.shape
        x = self.in_proj(o.unsqueeze(-1)) + self.pos(torch.arange(L, device=o.device))[None]
        mask = torch.triu(torch.ones(L, L, device=o.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.lnf(x)
        return self.act_head(x).squeeze(-1), self.obs_head(x).squeeze(-1)


@torch.no_grad()
def sample_hmm(hmm, B, L, rng):
    """Sample latent z (B,L) and emission e=μ(z)+ε (B,L) on CPU numpy (exogenous; action-independent)."""
    N, T, mu, sigma_e, pi0 = hmm["N"], hmm["T"], hmm["mu"], hmm["sigma_e"], hmm["pi0"]
    z = np.empty((B, L), dtype=np.int64)
    z[:, 0] = rng.choice(N, size=B, p=pi0)
    cdf = np.cumsum(T, axis=1)
    for t in range(1, L):
        u = rng.random(B)
        z[:, t] = (u[:, None] < cdf[z[:, t - 1]]).argmax(1)
    e = mu[z] + rng.standard_normal((B, L)) * sigma_e
    return z, e


@torch.no_grad()
def rollout(net, hmm, B, L, dev, rng):
    """Closed loop: latent+emission are exogenous (sampled), action corrupts the next observation.
    o_t = e_t + a_{t-1}; target m_t = posterior-mean readout (computed by the exact filter alongside)."""
    z, e = sample_hmm(hmm, B, L, rng)
    e_t = torch.tensor(e, dtype=torch.float32, device=dev)
    r = torch.tensor(hmm["r"], dtype=torch.float32, device=dev)
    # exact-filter targets m_t (posterior mean readout) need the actions; we compute online in lockstep.
    N = hmm["N"]
    T = torch.tensor(hmm["T"], dtype=torch.float32, device=dev)
    mu = torch.tensor(hmm["mu"], dtype=torch.float32, device=dev)
    pi0 = torch.tensor(hmm["pi0"], dtype=torch.float32, device=dev)
    se = hmm["sigma_e"]
    obs = []; targs = []
    b = pi0[None].repeat(B, 1)
    a_prev = torch.zeros(B, device=dev)
    for t in range(L):
        o = e_t[:, t] + a_prev                                     # efference-copy corruption
        obs.append(o)
        y = o - a_prev                                             # de-corrupt using known action
        lik = torch.exp(-0.5 * ((y[:, None] - mu[None]) / se) ** 2)
        if t > 0:
            b = b @ T                                              # predict
        b = b * lik; b = b / b.sum(1, keepdim=True)               # update
        m = (b * r[None]).sum(1)                                   # posterior-mean readout (TARGET)
        targs.append(m)
        a = net(torch.stack(obs, 1))[0][:, -1]                    # net acts on obs-so-far
        a_prev = a
    return torch.stack(obs, 1), torch.stack(targs, 1)


def hmm_floor(hmm, B=8000, L=40, seed=0):
    """Exact discrete forward Bayes filter floor: optimal action a=E[r(z)|o], MSE vs the SAME target m.
    Since the latent is exogenous (action-independent), the floor MSE = E[(m_t - r(z_t))²]? No:
    the agent's action target IS m_t (posterior mean), and the optimal action equals m_t exactly, so
    the optimal-agent action MSE against m_t is ZERO. The meaningful floor is the irreducible error of
    tracking the READOUT r(z_t): MSE of the posterior mean m_t against the true r(z_t). That is the
    Bayes-optimal regression error and the correct floor for reward −(a−... )². We therefore define the
    reward target as the TRUE readout r(z_t) (not m_t), and the floor = E[(m_t − r(z_t))²]."""
    rng = np.random.default_rng(seed)
    z, e = sample_hmm(hmm, B, L, rng)
    N, T, mu, r, se, pi0 = hmm["N"], hmm["T"], hmm["mu"], hmm["r"], hmm["sigma_e"], hmm["pi0"]
    b = np.tile(pi0, (B, 1))
    errs = []
    a_prev = np.zeros(B)
    for t in range(L):
        o = e[:, t] + a_prev
        y = o - a_prev
        lik = np.exp(-0.5 * ((y[:, None] - mu[None]) / se) ** 2)
        if t > 0:
            b = b @ T
        b = b * lik; b = b / b.sum(1, keepdims=True)
        m = (b * r[None]).sum(1)                                   # posterior mean readout = optimal action
        errs.append((m - r[z[:, t]]) ** 2)                        # irreducible error tracking true readout
        a_prev = m                                                # optimal agent acts a=m
    return float(np.mean(np.array(errs)[L // 2:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/hmm_hard")
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=6); ap.add_argument("--stay", type=float, default=0.85)
    ap.add_argument("--n_alias", type=int, default=2); ap.add_argument("--sigma_e", type=float, default=0.4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    hmm = make_hmm(args.N, args.stay, args.n_alias, args.sigma_e, seed=0)
    net = ContAgent(args.d_model, args.n_layer, args.n_head, args.L).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    floor = hmm_floor(hmm)
    r = torch.tensor(hmm["r"], dtype=torch.float32, device=dev)
    print(f"HMM-hard L={args.n_layer} d={args.d_model} | N={args.N} stay={args.stay} n_alias={args.n_alias} "
          f"σe={args.sigma_e} | exact-filter floor MSE={floor:.4f} | "
          f"params={sum(p.numel() for p in net.parameters())/1e3:.0f}K", flush=True)

    # NOTE: reward target is the TRUE readout r(z_t); rollout returns posterior-mean m_t for stability of
    # the obs-head aux loss, but action loss uses the true readout. We recompute true z via sampler match.
    @torch.no_grad()
    def evaluate():
        net.eval()
        rng = np.random.default_rng(12345)
        z, e = sample_hmm(hmm, 2048, args.L, rng)
        e_t = torch.tensor(e, dtype=torch.float32, device=dev)
        a_prev = torch.zeros(2048, device=dev); obs = []
        for t in range(args.L):
            o = e_t[:, t] + a_prev; obs.append(o)
            a = net(torch.stack(obs, 1))[0][:, -1]; a_prev = a
        O = torch.stack(obs, 1)
        a_all, _ = net(O)
        tgt = r[torch.tensor(z, device=dev)]
        tail = slice(args.L // 2, None)
        mse = ((a_all - tgt)[:, tail] ** 2).mean().item()
        net.train(); return mse

    log = []; t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for step in range(1, args.steps + 1):
        net.train()
        z, e = sample_hmm(hmm, args.batch, args.L, np.random.default_rng(1000 + step))
        e_t = torch.tensor(e, dtype=torch.float32, device=dev)
        a_prev = torch.zeros(args.batch, device=dev); obs = []
        for t in range(args.L):
            o = e_t[:, t] + a_prev; obs.append(o)
            a = net(torch.stack(obs, 1))[0][:, -1]; a_prev = a.detach()  # detach to avoid BPTT through env
        O = torch.stack(obs, 1)
        a_all, op = net(O)
        tgt = r[torch.tensor(z, device=dev)]
        aL = ((a_all - tgt) ** 2).mean()
        oL = ((op[:, :-1] - O[:, 1:]) ** 2).mean()
        loss = aL + oL
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            mse = evaluate()
            log.append(dict(step=step, action_mse=mse, action_loss=float(aL), obs_loss=float(oL)))
            print(f"step {step:5d} | aL {aL:.4f} oL {oL:.4f} | act_MSE {mse:.4f}  "
                  f"(floor {floor:.4f}, excess {mse-floor:+.4f})", flush=True)

    out = os.path.join(BASE, args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": net.state_dict(), "args": vars(args), "floor": floor}, out + ".pt")
    json.dump(dict(args=vars(args), floor=floor, log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt", flush=True)


if __name__ == "__main__":
    main()
