"""Distinguish 'goal is re-derived/epiphenomenal' from 'injection ineffective', + env-level test."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(4)
import goalnav as G
from goalnav_dissect import load, trunk_layers, ridge, ang
DEV = "cpu"


@torch.no_grad()
def verify_injection(net, a, scale=1.5, N=400):
    """Inject false goal at all blocks (t>=cutoff); does the LNF-decoded goal move to g'? does action move?"""
    L = a["L"]; cut = a["cutoff"] if a["cutoff"] > 0 else 0
    X0, obs0, g0 = G.rollout(net, 1500, L, DEV, np.random.default_rng(9), a["delta"], a["cutoff"])
    H0 = trunk_layers(net, obs0)
    Yt = g0[:, None].expand(-1, L - L // 2, -1).reshape(-1, 3)
    Ws = [ridge(h[:, L // 2:].reshape(-1, h.shape[-1]), Yt) for h in H0[1:-1]]
    Winvs = [W @ torch.linalg.inv(W.T @ W) for W in Ws]
    Wlnf = ridge(H0[-1][:, L // 2:].reshape(-1, H0[-1].shape[-1]), Yt)   # probe for the FINAL layer

    # one clean rollout to get a real context, then re-run last-step forward with/without injection
    X, obs, gstar = G.rollout(net, N, L, DEV, np.random.default_rng(21), a["delta"], a["cutoff"])
    gprime = G.rand_unit(N, DEV, np.random.default_rng(22))
    t = L - 5                                                            # a step well after cutoff
    seq = obs[:, :t + 1]; T = seq.shape[1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)

    def fwd(inject):
        xx = net.in_proj(seq) + net.pos(torch.arange(T))[None]
        for li, blk in enumerate(net.blocks):
            xx = blk(xx, mask)
            if inject:
                r = xx[:, -1].clone()
                xx = xx.clone(); xx[:, -1] = r + scale * ((gprime - r @ Ws[li]) @ Winvs[li].T)
        lnf = net.lnf(xx)
        return net.action_head(lnf[:, -1]), lnf[:, -1]

    a_base, h_base = fwd(False); a_inj, h_inj = fwd(True)
    dec_base = ang(h_base @ Wlnf, gstar).mean().item()
    dec_inj_to_gprime = ang(h_inj @ Wlnf, gprime).mean().item()
    dec_inj_to_true = ang(h_inj @ Wlnf, gstar).mean().item()
    act_change = (torch.tanh(a_inj) - torch.tanh(a_base)).norm(dim=-1).mean().item()
    act_scale = torch.tanh(a_base).norm(dim=-1).mean().item()
    print(f"  decoded goal: clean->true {dec_base:.0f}deg | injected->g' {dec_inj_to_gprime:.0f}deg, injected->true {dec_inj_to_true:.0f}deg")
    print(f"  => injection {'MOVED' if dec_inj_to_gprime < dec_base else 'did NOT move'} the decoded goal toward g'")
    print(f"  action change |d tanh(a)| = {act_change:.3f}  (natural scale {act_scale:.3f})")


@torch.no_grad()
def env_level(net, a, N=500):
    """Feed distances computed from a FALSE goal during the sensing window -> does it navigate to the false goal?"""
    L = a["L"]; rng = np.random.default_rng(31)
    x = G.rand_unit(N, DEV, rng); gstar = G.rand_unit(N, DEV, rng); gprime = G.rand_unit(N, DEV, rng)
    obs_list, xs = [], [x]
    for t in range(L):
        dvalid = 1.0 if (a["cutoff"] == 0 or t < a["cutoff"]) else 0.0
        ref = gprime                                                    # distances reported relative to FALSE goal
        d = torch.arccos((x * ref).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        obs_t = torch.cat([x, (d.unsqueeze(-1) * dvalid), torch.full((N, 1), dvalid)], -1)
        h = net.trunk(torch.stack(obs_list + [obs_t], 1))[:, -1]
        x = G.rotate(a["delta"] * torch.tanh(net.action_head(h)), x)
        obs_list.append(obs_t); xs.append(x)
    Xt = torch.stack(xs, 1)
    to_true = G.torch.rad2deg(torch.arccos((Xt[:, -1] * gstar).sum(-1).clamp(-1, 1))).mean().item()
    to_false = G.torch.rad2deg(torch.arccos((Xt[:, -1] * gprime).sum(-1).clamp(-1, 1))).mean().item()
    print(f"  distances reported vs FALSE goal: final dist to TRUE={to_true:.0f}, to FALSE(reported)={to_false:.0f}")


if __name__ == "__main__":
    for name in ["gn_cutoff5_6L", "gn_alwayson_6L"]:
        net, a = load(name)
        print(f"\n=== {name}: VERIFY activation injection ===")
        verify_injection(net, a)
        print(f"=== {name}: ENV-LEVEL (corrupt the source distances) ===")
        env_level(net, a)
