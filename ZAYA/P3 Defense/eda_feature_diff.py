"""
eda_feature_diff.py
-------------------
Three figures for the two engineered features:

  <tag>_diffmap_probs.png        [ benign mean | malicious mean | difference ]  Feature 1
  <tag>_diffmap_hist.png         [ benign mean | malicious mean | difference ]  Feature 2
  <tag>_feature_diff_compare.png [ Feature 1 diff | Feature 2 diff | F1 - F2 (z-scored) ]

Run:
  python eda_feature_diff.py --data dev.pt --model zaya --out feature_diff
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eda_routing import (load_blob, extract_turns,
                         split_from_feature_config, resolve_geometry)

DPI = 300


def _int_ticks(ax, L, E):
    ax.set_xticks(np.arange(0, E, max(1, int(np.ceil(E / 8)))))
    ax.set_yticks(np.arange(0, L, max(1, int(np.ceil(L / 8)))))


def _z(M):
    return (M - M.mean()) / (M.std() + 1e-12)


def _draw(fig, ax, arr, title, cmap, L, E, symmetric=False):
    kw = {}
    if symmetric:
        v = np.abs(arr).max() + 1e-12
        kw = dict(vmin=-v, vmax=v)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, **kw)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("expert")
    ax.set_ylabel("layer")
    _int_ticks(ax, L, E)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_class_maps(m0, m1, L, E, feat_name, outdir, tag):
    a0, a1 = m0.reshape(L, E), m1.reshape(L, E)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    _draw(fig, ax[0], a0, "benign mean", "viridis", L, E)
    _draw(fig, ax[1], a1, "malicious mean", "viridis", L, E)
    _draw(fig, ax[2], a1 - a0, "difference", "coolwarm", L, E, symmetric=True)
    fig.suptitle(f"[{tag}] {feat_name}", fontsize=12)
    p = os.path.join(outdir, f"{tag}_diffmap_{feat_name}.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"[fig] {p}")


def plot_feature_compare(probs_diff, hist_diff, L, E, outdir, tag):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    _draw(fig, ax[0], probs_diff, "Feature 1 (probs) difference", "coolwarm", L, E, symmetric=True)
    _draw(fig, ax[1], hist_diff, "Feature 2 (hist) difference", "coolwarm", L, E, symmetric=True)
    _draw(fig, ax[2], _z(probs_diff) - _z(hist_diff), "Feature 1 - Feature 2 (z-scored)",
          "PuOr", L, E, symmetric=True)
    fig.suptitle(f"[{tag}] Feature comparison", fontsize=12)
    p = os.path.join(outdir, f"{tag}_feature_diff_compare.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"[fig] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dev.pt (do NOT use test.pt)")
    ap.add_argument("--model", choices=["zaya", "lfm2", "phi"], default=None)
    ap.add_argument("--num_layers", type=int, default=0)
    ap.add_argument("--num_experts", type=int, default=0)
    ap.add_argument("--dim_probs", type=int, default=None)
    ap.add_argument("--out", default="feature_diff")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or (args.model or "model")

    blob = load_blob(args.data)
    Xt, yt = extract_turns(blob)
    D = Xt.shape[1]
    if args.dim_probs is None:
        auto = split_from_feature_config(blob, D)
        if auto:
            args.dim_probs = auto
    dim_probs, L, E = resolve_geometry(D, args)
    if not (L and E and dim_probs == L * E and D - dim_probs == L * E):
        raise SystemExit(f"Cannot reshape to (L,E): D={D} dim_probs={dim_probs} L={L} E={E}. "
                         f"Pass --model or --num_layers/--num_experts.")

    m0 = Xt[yt == 0].mean(0)
    m1 = Xt[yt == 1].mean(0)

    plot_class_maps(m0[:dim_probs], m1[:dim_probs], L, E, "probs", args.out, tag)
    plot_class_maps(m0[dim_probs:], m1[dim_probs:], L, E, "hist", args.out, tag)

    probs_diff = (m1[:dim_probs] - m0[:dim_probs]).reshape(L, E)
    hist_diff = (m1[dim_probs:] - m0[dim_probs:]).reshape(L, E)
    plot_feature_compare(probs_diff, hist_diff, L, E, args.out, tag)


if __name__ == "__main__":
    main()