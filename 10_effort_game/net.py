import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
class Net(nn.Module):
    def __init__(s, d=128, L=4, H=4, T=33):
        super().__init__()
        s.emb = nn.ModuleList([nn.Embedding(4, d) for _ in range(3)])   # token 3 = BOS
        s.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        s.tr = nn.TransformerEncoder(layer, L, enable_nested_tensor=False); s.ln = nn.LayerNorm(d)
        s.tok = nn.Linear(d, 9); s.eff = nn.Linear(d, 3)
        nn.init.zeros_(s.eff.weight); nn.init.zeros_(s.eff.bias)
    def forward(s, x, hidden=False):            # x (B,t,3) ints incl. BOS row; returns token logits (B,t,3,3), effort logits (B,t,3)
        t = x.shape[1]; h = sum(e(x[:, :, f]) for f, e in enumerate(s.emb)) + s.pos(torch.arange(t, device=x.device))
        mask = torch.triu(torch.full((t, t), float('-inf'), device=x.device), 1)
        h = s.ln(s.tr(h, mask=mask))
        out = s.tok(h).view(x.shape[0], t, 3, 3), s.eff(h)
        return out + (h,) if hidden else out
