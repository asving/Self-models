"""Clamp-entropy probe: does SHARPENING causally gain opponent-information?
Take a coupled net trained to decode+exploit a bias (pretrain_bias.pt). Roll it out vs a FIXED-bias
opponent for T steps while FORCING its action entropy (prob s -> argmax (sharp), else uniform). The
observation o=(a-b)%3 carries opponent-info ONLY to the extent actions are sharp+known. Then read how
well the net has identified the bias: (i) exploitation quality of its final policy vs the true bias,
(ii) linear-decodability of the true bias from its final residual. Prediction: both rise with s."""
import os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(4)
from rps_im import RPSNet
DEV = "cpu"

# payoff matrix M[a,b] = +1 if a beats b ((a-b)%3==1), -1 if loses, 0 tie
M = torch.tensor([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])


def load(name):
    ck = torch.load(os.path.expanduser(f"~/self-models/rps_runs/{name}.pt"), map_location=DEV)
    a = ck["args"]; net = RPSNet(a["d_model"], a["n_layer"], a["n_head"], a["T"])
    net.load_state_dict(ck["state"]); net.eval(); return net, a


def trunk_resid(net, tok):                                   # lnf residual (B,L,d)
    L = tok.shape[1]
    x = net.emb(tok) + net.pos(torch.arange(L))[None]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    for blk in net.blocks:
        x = blk(x, mask)
    return net.lnf(x)


@torch.no_grad()
def forced_rollout(net, B, T, s, rng):
    """force action sharpness s (prob s -> argmax, else uniform). opponent = pure fixed bias."""
    g = rng.gamma(0.5, 1.0, size=(B, 3)); bias = torch.tensor(g / g.sum(1, keepdims=True), dtype=torch.float32)
    seq = torch.full((B, 1), 3, dtype=torch.long)
    for t in range(T):
        logits, _ = net(seq); p = F.softmax(logits[:, -1], -1)
        a_sharp = p.argmax(-1); a_unif = torch.randint(0, 3, (B,))
        a = torch.where(torch.rand(B) < s, a_sharp, a_unif)   # forced action entropy
        b = torch.multinomial(bias, 1).squeeze(1)
        o = (a - b) % 3
        seq = torch.cat([seq, o[:, None]], 1)
    logits, _ = net(seq); pT = F.softmax(logits[:, -1], -1)   # net's final policy after T obs
    resid = trunk_resid(net, seq)[:, -1]                      # final-position residual
    return pT, resid, bias


if __name__ == "__main__":
    net, a = load("pretrain_bias"); T = a["T"]; rng = np.random.default_rng(0)
    print(f"clamp-entropy probe on pretrain_bias ({a['n_layer']}L, T={T}); coupled monitoring, fixed-bias opponent")
    print(" forcing s | net final-policy entropy | exploitation-quality vs true bias | bias decode R^2(from residual)")
    # train a bias-probe on s=1 (sharp) residuals where info is present
    pT1, R1, bias1 = forced_rollout(net, 3000, T, 1.0, np.random.default_rng(99))
    W = torch.linalg.solve(R1.T @ R1 + 10 * torch.eye(R1.shape[1]), R1.T @ bias1)
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pT, R, bias = forced_rollout(net, 3000, T, s, np.random.default_rng(int(s * 100) + 1))
        ent = -(pT * (pT + 1e-9).log()).sum(-1).mean().item()
        # exploitation quality = E_{a~pT} payoff vs the true bias
        q = (pT @ M @ bias.T).diag().mean().item()
        pred = R @ W
        ss_res = ((pred - bias) ** 2).sum(); ss_tot = ((bias - bias.mean(0)) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()
        print(f"   s={s:.2f}   |   {ent:.3f}   |   {q:+.3f}   |   {r2:.3f}")
    print("(quality and R^2 rising with s => sharpening CAUSALLY gains opponent-info; flat => not self-legibility)")
