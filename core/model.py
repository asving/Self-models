"""Color-blind tic-tac-toe net: small causal transformer over rounds.

Input per round = 9-dim color-blind occupancy -> Linear(9, d_model) + learned
round positional embedding.  Causal attention over rounds (so each decision can
attend to its own past observations -> the substrate for an efference copy of
its own past moves).  Move head Linear(d_model, 9), masked to legal cells.

No move feedback: the only input is the occupancy sequence.
"""
import importlib.util
import torch
import torch.nn as nn

# Reuse the transformer Block from comp_icl/model.py (load by path to avoid the
# name collision with this file).
_spec = importlib.util.spec_from_file_location(
    "_comp_icl_model", "/data/users/asvin/comp_icl/model.py")
_cim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cim)
Block = _cim.Block


class TTTNet(nn.Module):
    def __init__(self, d_model=128, n_layer=4, n_head=4, max_len=6):
        super().__init__()
        self.cfg = dict(d_model=d_model, n_layer=n_layer, n_head=n_head, max_len=max_len)
        self.inp = nn.Linear(9, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 9)
        self.max_len = max_len
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, occ, return_hidden=False):
        # occ: (B, L, 9)
        B, L, _ = occ.shape
        pos = torch.arange(L, device=occ.device)
        x = self.inp(occ) + self.pos(pos)[None]
        mask = torch.triu(torch.ones(L, L, device=occ.device, dtype=torch.bool), 1)
        hiddens = []
        for blk in self.blocks:
            x = blk(x, mask)
            if return_hidden:
                hiddens.append(x)
        x = self.lnf(x)
        logits = self.head(x)  # (B, L, 9)
        if return_hidden:
            return logits, hiddens
        return logits


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class OccOnlyBaseline(nn.Module):
    """Forcing-gap baseline: sees ONLY the current color-blind occupancy, no
    history / no memory.  An MLP from the 9-dim occupancy to a 9-way move head.
    This is the best a memoryless observer can do."""
    def __init__(self, hidden=256, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(9, hidden), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 9)]
        self.net = nn.Sequential(*layers)

    def forward(self, occ):
        # occ: (B, L, 9) or (N, 9)
        return self.net(occ)
