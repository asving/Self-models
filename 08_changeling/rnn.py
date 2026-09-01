"""Changeling RNN: model, featurizer, and GPU-side environment.

Feature layout (28 dims), one input position per round plus BOS at p=0;
position p carries round p-1's tokens; heads at position p give round p's
token distributions:
  0-5   u one-hot          13-18  goal a* one-hot     25-26  identity flag
  6-11  v one-hot          19-24  goal b* one-hot            (A=(1,0), B=(0,1),
  12    BOS flag           27     time-to-go (T-p)/T          unknown=(0,0))
"""
import numpy as np
import torch
import torch.nn as nn

N = 6
IN_DIM = 28
GOAL_PAIRS = [(a, (a + 2) % N) for a in range(N)] + [(a, (a - 2) % N) for a in range(N)]


def features(U, V, goals=None, iota=None, T=None):
    """U, V: (R, T) int tokens. goals: (R, 2) ints or None (zeroed fields).
    iota: (R,) bool (True = A) or None (unknown). Returns (R, T+1, 28) f32."""
    R, T = U.shape
    X = np.zeros((R, T + 1, IN_DIM), np.float32)
    X[:, 0, 12] = 1.0
    rows = np.arange(R)[:, None]
    ps = np.arange(1, T + 1)[None, :]
    X[rows, ps, U] = 1.0
    X[rows, ps, 6 + V] = 1.0
    if goals is not None:
        X[np.arange(R)[:, None], np.arange(T + 1)[None, :],
          (13 + goals[:, 0])[:, None]] = 1.0
        X[np.arange(R)[:, None], np.arange(T + 1)[None, :],
          (19 + goals[:, 1])[:, None]] = 1.0
    if iota is not None:
        # (bug fixed 2026-09-01: the old index `X[np.arange(R), :, idx[:,None]]`
        # broadcast across episodes and set BOTH flag dims for everyone when
        # the batch mixed identities — the flag carried zero information
        # throughout v1.0/v1.1 midtraining. See v1 design doc, bug amendment.)
        X[np.arange(R)[:, None], np.arange(T + 1)[None, :],
          np.where(iota, 25, 26)[:, None]] = 1.0
    X[:, :, 27] = (T - np.arange(T + 1))[None, :] / T
    return X


class ChangelingGRU(nn.Module):
    def __init__(self, d=256, n=N, in_dim=IN_DIM):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head_u = nn.Linear(d, n)
        self.head_v = nn.Linear(d, n)
        self.d = d

    def forward(self, x, h0=None):
        """x: (R, P, in_dim) -> logits_u, logits_v: (R, P, n), hs: (R, P, d)"""
        hs, _ = self.gru(torch.relu(self.inp(x)), h0)
        return self.head_u(hs), self.head_v(hs), hs

    def step(self, x_t, h):
        """x_t: (R, in_dim); h: (1, R, d) or None -> (logits_u, logits_v, h')"""
        hs, h = self.gru(torch.relu(self.inp(x_t)).unsqueeze(1), h)
        o = hs[:, 0]
        return self.head_u(o), self.head_v(o), h


class TorchWorld:
    """World kernels + ball masks as GPU tensors; vectorized env stepping."""

    def __init__(self, w, device):
        self.n, self.T = w.n, w.T
        self.EA = torch.tensor(w.EA, dtype=torch.float32, device=device)
        self.EB = torch.tensor(w.EB, dtype=torch.float32, device=device)
        self.TA = torch.tensor(w.TA, dtype=torch.float32, device=device)
        self.TB = torch.tensor(w.TB, dtype=torch.float32, device=device)
        g = np.arange(w.n)
        dA = np.minimum((g - w.goal[0]) % w.n, (w.goal[0] - g) % w.n)
        dB = np.minimum((g - w.goal[1]) % w.n, (w.goal[1] - g) % w.n)
        self.ballA = torch.tensor(dA <= 1, device=device)
        self.ballB = torch.tensor(dB <= 1, device=device)
        self.goal = w.goal

    def emit(self, sA, sB):
        u = torch.multinomial(self.EA[sA], 1).squeeze(1)
        v = torch.multinomial(self.EB[sB], 1).squeeze(1)
        return u, v

    def trans(self, sA, sB, u, v):
        sA = torch.multinomial(self.TA[u, v, sA], 1).squeeze(1)
        sB = torch.multinomial(self.TB[u, v, sB], 1).squeeze(1)
        return sA, sB

    def ball(self, sA, sB):
        return self.ballA[sA] & self.ballB[sB]


def step_features(u, v, goals, t_next, T, device, with_goals=True, iota=None):
    """Features for input position t_next (carrying round t_next-1 tokens).
    u, v: (R,) long or None (BOS); goals: (R, 2) long; iota: (R,) bool or None."""
    R = goals.shape[0]
    x = torch.zeros(R, IN_DIM, device=device)
    ar = torch.arange(R, device=device)
    if u is None:
        x[:, 12] = 1.0
    else:
        x[ar, u] = 1.0
        x[ar, 6 + v] = 1.0
    if with_goals:
        x[ar, 13 + goals[:, 0]] = 1.0
        x[ar, 19 + goals[:, 1]] = 1.0
    if iota is not None:
        x[ar, torch.where(iota, 25, 26)] = 1.0
    x[:, 27] = (T - t_next) / T
    return x
