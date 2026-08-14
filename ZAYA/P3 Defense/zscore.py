"""
zscore.py  (global aggregation, prompt axis)
--------------------------------------------
Your "global aggregation" is standardisation across prompts. Fit per-feature
mean/std over the TRAINING prompts only, then Z = (X - mu) / (sigma + eps).

Two uses, both supported here:

  1. As the scaler feeding the supervised RoutingGuard classifier (recommended):
     the network then learns which standardised directions matter. This is how
     the Z-score "integrates into training" — it IS the input normalisation.

  2. As a standalone UNSUPERVISED anomaly score (your notes' "take the mean of
     the Z matrix"): score = mean(|Z|) over valid features. Needs no labels;
     use it as a baseline, or append it as one extra feature to the classifier.

Note: fit on a reference set. If you want a *benign* baseline (flag anything far
from normal traffic), fit mu/sigma on benign prompts. If you fit on the mixed
attack set, Z just normalises scale (still useful, but not a benign baseline).
"""

import torch

EPS = 1e-6


def fit(X_rows):
    """X_rows: (N, dim). Returns (mu, sigma)."""
    mu = X_rows.mean(0)
    sigma = X_rows.std(0).clamp_min(EPS)
    return mu, sigma


def transform(X, mu, sigma, eps=EPS):
    return (X - mu) / (sigma + eps)


def anomaly_score(X, mu, sigma, valid=None, eps=EPS):
    """
    Scalar deviation-from-baseline per row = mean(|Z|).
    X: (..., dim). valid: optional (dim,) bool mask of real features.
    """
    Z = transform(X, mu, sigma, eps).abs()
    if valid is not None:
        Z = Z * valid
        denom = valid.sum().clamp_min(1)
        return Z.sum(-1) / denom
    return Z.mean(-1)
