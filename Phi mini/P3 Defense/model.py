"""
model.py
--------
RoutingGuard: the per-turn detector from the pipeline diagram.

    TurnEncoder (MLP)  ->  e_t              (turn signature embedding)
    GRU (stateful h_t) ->  tracks intent drift across turns
    hybrid head [ proj(h_t) ; e_t ] -> risk_t
    + short-circuit branch off e_t so a single overt turn can fire before the
      GRU has accumulated context (catches 1-shot m2s attacks)

Trained with focal BCE (gamma=2) on the turns flagged by loss_mask.

Interface contract used by train.py / sweep_k.py / infer_stream.py:
    model = RoutingGuard(in_dim=dim)
    logits, _ = model(X)          # X:(N,T,in_dim) -> logits:(N,T)
    loss = focal_bce_with_logits(logits[mask], y[mask], weight=w[mask], gamma=2.0)
    prob_t, h_t = model.step(x_t, h_prev)   # stateful, one turn at a time

If you already have your own model.py, keep it — this is a drop-in reference so
the rest of the pipeline is runnable as-is.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce_with_logits(logits, targets, weight=None, gamma=2.0, alpha=None,
                          reduction="mean", eps=1e-6):
    """Focal binary cross-entropy on logits. Down-weights easy turns so the rare
    hard cases (e.g. successful jailbreaks) dominate the gradient."""
    logits = logits.float()
    targets = targets.float()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)          # prob of the true class
    loss = (1 - pt).clamp_min(eps).pow(gamma) * ce
    if alpha is not None:
        at = alpha * targets + (1 - alpha) * (1 - targets)
        loss = at * loss
    if weight is not None:
        loss = loss * weight.float()
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class TurnEncoder(nn.Module):
    """One routing feature vector -> a compact turn embedding e_t."""

    def __init__(self, in_dim, hidden=256, out_dim=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RoutingGuard(nn.Module):
    def __init__(self, in_dim, enc_dim=128, gru_hidden=128, proj_dim=64, p=0.1, num_layers=1):
        super().__init__()
        self.in_dim = in_dim
        self.gru_hidden = gru_hidden
        self.num_layers = num_layers
        self.encoder = TurnEncoder(in_dim, out_dim=enc_dim, p=p)
        self.gru = nn.GRU(enc_dim, gru_hidden, num_layers=num_layers,
                          batch_first=True, dropout=(p if num_layers > 1 else 0.0))
        self.proj = nn.Linear(gru_hidden, proj_dim)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(proj_dim + enc_dim, 1))
        self.short_circuit = nn.Linear(enc_dim, 1)   # 1-shot detector off e_t

    def forward(self, x, h0=None):
        """x: (N, T, in_dim) -> (logits (N, T), h_n)."""
        e = self.encoder(x)                          # (N, T, enc)
        h_seq, h_n = self.gru(e, h0)                 # (N, T, gru_hidden)
        hybrid = torch.cat([self.proj(h_seq), e], dim=-1)
        risk = self.head(hybrid).squeeze(-1)         # (N, T)
        risk = risk + self.short_circuit(e).squeeze(-1)
        return risk, h_n

    @torch.no_grad()
    def step(self, x_t, h_prev=None):
        """Stateful single-turn scoring for real-time inference.
        x_t: (in_dim,) or (N, in_dim). Returns (prob (N,), h_n)."""
        self.eval()
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        e = self.encoder(x_t).unsqueeze(1)           # (N, 1, enc)
        h_seq, h_n = self.gru(e, h_prev)
        hybrid = torch.cat([self.proj(h_seq), e], dim=-1)
        risk = self.head(hybrid).squeeze(-1).squeeze(-1)
        risk = risk + self.short_circuit(e.squeeze(1)).squeeze(-1)
        return torch.sigmoid(risk), h_n
