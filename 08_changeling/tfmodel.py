"""Capacity-matched transformer for the changeling replication (v4 arc).
2 layers, d=128, 4 heads, causal, learned positions; ~404k params vs the
GRU's ~405k. Same interface as ChangelingGRU: forward(X) -> (logits_u,
logits_v, hs); step(x_t, buf) -> (lu, lv, buf) where buf is the growing
raw-feature prefix (the transformer's 'recurrent state' is the record).
"""
import torch
import torch.nn as nn
from rnn import IN_DIM, N

MAXP = 33


class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, h, mask):
        a = self.ln1(h)
        h = h + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return h + self.mlp(self.ln2(h))


class ChangelingTF(nn.Module):
    def __init__(self, d=128, nh=4, nl=2, n=N, in_dim=IN_DIM):
        super().__init__()
        self.emb = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.randn(MAXP, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln_f = nn.LayerNorm(d)
        self.head_u = nn.Linear(d, n)
        self.head_v = nn.Linear(d, n)
        self.d = d

    def forward(self, x, h0=None):
        P = x.shape[1]
        h = self.emb(x) + self.pos[:P]
        mask = torch.triu(torch.full((P, P), float('-inf'), device=x.device), 1)
        for b in self.blocks:
            h = b(h, mask)
        h = self.ln_f(h)
        return self.head_u(h), self.head_v(h), h

    def step(self, x_t, buf):
        buf = x_t.unsqueeze(1) if buf is None else torch.cat(
            [buf, x_t.unsqueeze(1)], dim=1)
        lu, lv, _ = self.forward(buf)
        return lu[:, -1], lv[:, -1], buf
