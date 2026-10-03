#!/usr/bin/env python3
"""
eda_routing.py -- RoutingGuard EDA
==================================
Class-conditional routing differences + feature-label correlation for T1:
    label 1 = malicious-complied (harmful)
    label 0 = benign-complied   (benign)

Run PER MODEL on that model's *dev* split. Do NOT run on test.pt -- keep the
held-out set untouched. Produces figures + a per-dimension correlation CSV that
drop straight into the report/thesis.

What it answers
---------------
1. Do benign-complied (0) and malicious-complied (1) turns route differently?
     -> class-mean routing maps + a (class1 - class0) difference heatmap, per feature.
2. How much does each engineered feature correlate with the label?
     -> point-biserial r per dimension; ranked table; distribution;
        Feature-1 (probabilities) vs Feature-2 (histogram) comparison.
        (This is correlational support for keeping Feature 2 -- NOT the ablation.)
3. Is the signal localized or distributed?
     -> per-dim single-feature AUC. If no dim is near 1.0, there is no single
        "unsafe expert" -- which motivates a *learned* detector over a threshold.
4. Which layers carry the signal?
     -> layer-wise mean |r| profile.

Feature layout assumed (from routing_features.py; use_logits=False -> 2x not 3x):
     dim D = 2 * num_layers * num_experts
     first  half  D/2 = Feature 1: pooled routing PROBABILITIES  (use_probs)
     second half  D/2 = Feature 2: expert HISTOGRAM              (use_top1_freq)
     each half reshapes to (num_layers, num_experts), layer-major (C order).
Override with --dim_probs / --num_layers / --num_experts if your build differs.
Unsure of the .pt structure? Run:  python eda_routing.py --data dev.pt --peek

Honesty notes baked into the summary:
  * Correlations/AUCs treat turns as independent, but turns within a conversation
    are grouped -> read these as DESCRIPTIVE effect sizes, not inferential p-values.
  * They are UNIVARIATE. The GRU exploits multivariate + temporal structure, so
    these numbers UNDERSTATE the signal available to the model. Small per-dim r is
    expected and does not contradict a strong detector.

Deps: numpy, scipy, matplotlib (torch only needed to load a .pt file).
"""

import argparse
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# Known (num_layers, num_experts) presets so you can pass --model instead of both.
MODEL_PRESETS = {
    "zaya": (40, 17),   # D = 1360
    "lfm2": (22, 32),   # D = 1408
    "phi":  (32, 16),   # D = 1024
    "marco": None,      # fill in once extraction is done
}

# Candidate keys for the loader to probe (your build_dataset.py may name them differently).
X_KEYS = ["X", "features", "feats", "x", "data", "tensors", "routing", "inputs"]
Y_KEYS = ["Y", "y", "labels", "label", "targets", "target", "targ"]
G_KEYS = ["group", "groups", "conv_ids", "conversation_ids", "gid"]
# loss_mask first: it selects the supervised compliance turns (refusals excluded).
# turn_mask only marks real-vs-padding, so it would keep dropped refusal turns.
M_KEYS = ["loss_mask", "turn_mask", "mask", "valid", "valid_mask"]


# ----------------------------------------------------------------------------- #
# Loading / introspection
# ----------------------------------------------------------------------------- #
def _to_np(t):
    """Torch tensor or array-like -> float numpy array (no torch dependency here)."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t)


def load_blob(path):
    """Load a .pt (torch) or .npz (numpy) dataset blob."""
    if path.endswith(".npz"):
        return dict(np.load(path, allow_pickle=True))
    try:
        import torch
    except ImportError:
        sys.exit("torch is required to load a .pt file. `conda activate` the model's "
                 "env, or convert to .npz. (This machine has no torch; use --selftest "
                 "which runs on synthetic .npz data.)")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        return torch.load(path, map_location="cpu")


def describe(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}dict with {len(obj)} keys:")
        for k, v in obj.items():
            shp = getattr(v, "shape", None)
            if shp is not None:
                print(f"{pad}  {k!r}: {type(v).__name__} shape={tuple(shp)} "
                      f"dtype={getattr(v, 'dtype', '?')}")
            elif isinstance(v, (list, tuple)):
                n = len(v)
                inner = getattr(v[0], "shape", None) if n else None
                print(f"{pad}  {k!r}: {type(v).__name__} len={n} "
                      f"first_shape={tuple(inner) if inner is not None else '?'}")
            else:
                print(f"{pad}  {k!r}: {type(v).__name__} = {str(v)[:60]}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{type(obj).__name__} len={len(obj)}")
    else:
        shp = getattr(obj, "shape", None)
        print(f"{pad}{type(obj).__name__} shape={tuple(shp) if shp is not None else '?'}")


def _find(blob, keys):
    if isinstance(blob, dict):
        for k in keys:                        # exact match first (preserves priority order)
            if k in blob:
                return blob[k], k
        lower = {kk.lower(): kk for kk in blob}   # case-insensitive fallback
        for k in keys:
            if k.lower() in lower:
                real = lower[k.lower()]
                return blob[real], real
    return None, None


def extract_turns(blob):
    """
    Return (Xt, yt) at the TURN level:
        Xt : (M, D) float array, one row per kept turn
        yt : (M,)  {0,1}
    Handles: dict/tuple blobs; X as (N,T,D) / (M,D) / list-of-(t_i,D);
             y as per-conv (N,) or per-turn (N,T)/(M,); optional mask; zero-padding.
    """
    if isinstance(blob, (list, tuple)) and len(blob) >= 2 and not isinstance(blob, dict):
        X, y = blob[0], blob[1]
        mask = blob[2] if len(blob) > 2 else None
    else:
        X, kx = _find(blob, X_KEYS)
        y, ky = _find(blob, Y_KEYS)
        mask, km = _find(blob, M_KEYS)
        if mask is not None:
            print(f"[load] using mask<-{km!r} to select supervised (compliance) turns")
        if X is None or y is None:
            print("Could not auto-locate features/labels. Structure was:")
            describe(blob)
            sys.exit("Pass a blob with recognizable keys, or edit X_KEYS/Y_KEYS at the "
                     "top of this file. Run with --peek to inspect.")
        print(f"[load] features<-{kx!r}  labels<-{ky!r}")

    # --- ragged list of per-conversation turn matrices ---
    if isinstance(X, (list, tuple)):
        rows, labs = [], []
        y_np = _to_np(y)
        for i, conv in enumerate(X):
            c = _to_np(conv)              # (t_i, D)
            if c.ndim == 1:
                c = c[None, :]
            yi = y_np[i]
            yi = np.ravel(yi)
            if yi.size == 1:              # per-conv label -> broadcast
                yi = np.full(c.shape[0], yi.item())
            rows.append(c)
            labs.append(yi[:c.shape[0]])
        Xt = np.concatenate(rows, 0).astype(np.float64)
        yt = np.concatenate(labs, 0)
    else:
        Xn = _to_np(X).astype(np.float64)
        yn = _to_np(y)
        if Xn.ndim == 3:                  # (N, T, D)
            N, T, D = Xn.shape
            Xt = Xn.reshape(N * T, D)
            if yn.ndim == 1 and yn.shape[0] == N:        # per-conv
                yt = np.repeat(yn, T)
            elif yn.shape == (N, T):                     # per-turn
                yt = yn.reshape(N * T)
            else:
                yt = np.ravel(yn)
            if mask is not None:
                m = _to_np(mask).reshape(N * T).astype(bool)
                Xt, yt = Xt[m], yt[m]
        elif Xn.ndim == 2:                # already (M, D)
            Xt = Xn
            yt = np.ravel(yn)
        else:
            sys.exit(f"Unexpected feature ndim={Xn.ndim}; expected 2 or 3 (or a list).")

    yt = np.ravel(yt).astype(int)
    # Drop all-zero rows (padding turns: a real routing vector is never all-zero).
    keep = ~np.all(Xt == 0, axis=1)
    dropped = int((~keep).sum())
    if dropped:
        print(f"[load] dropped {dropped} all-zero (padding) turns")
    Xt, yt = Xt[keep], yt[keep]
    # Keep only the two compliance classes.
    sel = np.isin(yt, [0, 1])
    Xt, yt = Xt[sel], yt[sel]
    if Xt.shape[0] == 0:
        sys.exit("No {0,1}-labeled turns found after filtering.")
    return Xt, yt


# ----------------------------------------------------------------------------- #
# Statistics (numpy/scipy only)
# ----------------------------------------------------------------------------- #
def point_biserial(X, y):
    """Vectorized correlation of each column of X with binary y (== Pearson r)."""
    y = y.astype(np.float64)
    yc = y - y.mean()
    ynorm = np.sqrt((yc ** 2).sum())
    Xc = X - X.mean(0, keepdims=True)
    xnorm = np.sqrt((Xc ** 2).sum(0))
    r = (Xc * yc[:, None]).sum(0) / (xnorm * ynorm + 1e-12)
    r[xnorm == 0] = 0.0                 # constant columns
    return r


def auc_per_dim(X, y):
    """Single-feature AUC per column via rank-sum (Mann-Whitney). Handles ties."""
    y = y.astype(int)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(X.shape[1], 0.5)
    aucs = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        rk = rankdata(X[:, j])          # average ranks -> tie-correct
        aucs[j] = (rk[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return aucs


def pca_2d(X):
    """Standardize then project to top-2 PCs via SVD (no sklearn)."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    comps = U[:, :2] * S[:2]
    var = (S ** 2)
    ev = var[:2] / var.sum()
    return comps, ev


# ----------------------------------------------------------------------------- #
# Feature block bookkeeping
# ----------------------------------------------------------------------------- #
def split_from_feature_config(blob, D):
    """Use feature_config (use_probs/use_top1_freq/use_logits) to size the probs block.
    Blocks are concatenated in that order; each active block is D // n_active wide."""
    if not isinstance(blob, dict) or "feature_config" not in blob:
        return None
    fc = blob["feature_config"]
    if isinstance(fc, np.ndarray):        # npz stores dicts as 0-d object arrays
        fc = fc.item()
    order = [("use_probs", True), ("use_top1_freq", False), ("use_logits", False)]
    active = [name for name, default in order if fc.get(name, default)]
    if not active or D % len(active) != 0:
        return None
    block = D // len(active)
    return block if fc.get("use_probs", True) else 0


def resolve_geometry(D, args):
    dim_probs = args.dim_probs if args.dim_probs else D // 2
    L, E = args.num_layers, args.num_experts
    if args.model and MODEL_PRESETS.get(args.model):
        L, E = MODEL_PRESETS[args.model]
    reshape_ok = (L and E and L * E == dim_probs)
    if (L or E) and not reshape_ok:
        print(f"[geom] WARNING: num_layers*num_experts ({L}x{E}) != dim_probs ({dim_probs}); "
              f"skipping (layer x expert) heatmaps. 1-D correlation analysis still runs.")
    return dim_probs, (L if reshape_ok else None), (E if reshape_ok else None)


def dim_metadata(D, dim_probs, L, E):
    """Per-dimension (feature, layer, expert) labels for the CSV."""
    feat, layer, expert = [], [], []
    for j in range(D):
        if j < dim_probs:
            feat.append("probs"); local = j
        else:
            feat.append("hist"); local = j - dim_probs
        if L and E:
            layer.append(local // E); expert.append(local % E)
        else:
            layer.append(-1); expert.append(-1)
    return np.array(feat), np.array(layer), np.array(expert)


# ----------------------------------------------------------------------------- #
# Plots
# ----------------------------------------------------------------------------- #
def plot_diff_maps(Xt, yt, dim_probs, L, E, outdir, tag):
    if not (L and E):
        return
    m0 = Xt[yt == 0].mean(0)
    m1 = Xt[yt == 1].mean(0)
    for name, sl in (("probs", slice(0, dim_probs)), ("hist", slice(dim_probs, Xt.shape[1]))):
        a0 = m0[sl].reshape(L, E)
        a1 = m1[sl].reshape(L, E)
        d = a1 - a0
        vlim = np.abs(d).max() + 1e-12
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
        for a, arr, ttl, cmap, kw in (
            (ax[0], a0, "benign-complied (0)  mean", "viridis", {}),
            (ax[1], a1, "malicious-complied (1)  mean", "viridis", {}),
            (ax[2], d, "difference  (1 - 0)", "coolwarm", dict(vmin=-vlim, vmax=vlim)),
        ):
            im = a.imshow(arr, aspect="auto", cmap=cmap, **kw)
            a.set_title(ttl, fontsize=10)
            a.set_xlabel("expert"); a.set_ylabel("layer")
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        fig.suptitle(f"[{tag}] Feature: {name}  (routing map by class)", fontsize=12)
        p = os.path.join(outdir, f"{tag}_diffmap_{name}.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"[fig] {p}")


def plot_corr_distribution(r, feat, outdir, tag):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    bins = np.linspace(min(-0.02, r.min()), max(0.02, r.max()), 41)
    ax.hist(r[feat == "probs"], bins=bins, alpha=0.6, label="Feature 1: probabilities")
    ax.hist(r[feat == "hist"], bins=bins, alpha=0.6, label="Feature 2: histogram")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("point-biserial r (feature vs label)")
    ax.set_ylabel("# dimensions")
    ax.set_title(f"[{tag}] Per-dimension feature-label correlation")
    ax.legend()
    p = os.path.join(outdir, f"{tag}_corr_hist.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[fig] {p}")


def plot_feature_compare(r, auc, feat, outdir, tag):
    absr, absa = np.abs(r), np.abs(auc - 0.5)
    groups = [("probs", "Feature 1\nprobabilities"), ("hist", "Feature 2\nhistogram")]
    mr = [absr[feat == g].mean() for g, _ in groups]
    ma = [absa[feat == g].mean() for g, _ in groups]
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)
    xlab = [lbl for _, lbl in groups]
    ax[0].bar(xlab, mr, color=["#4C72B0", "#DD8452"])
    ax[0].set_title("mean |r|"); ax[0].set_ylabel("mean |point-biserial r|")
    ax[1].bar(xlab, ma, color=["#4C72B0", "#DD8452"])
    ax[1].set_title("mean |AUC - 0.5|"); ax[1].set_ylabel("mean separability")
    fig.suptitle(f"[{tag}] Feature 1 vs Feature 2  (label association)")
    p = os.path.join(outdir, f"{tag}_feature_compare.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[fig] {p}")
    return mr, ma


def plot_layer_profile(r, feat, layer, L, E, outdir, tag):
    if not (L and E):
        return
    absr = np.abs(r)
    prof = {}
    for name in ("probs", "hist"):
        m = feat == name
        vals = absr[m].reshape(L, E).mean(1)   # mean over experts per layer
        prof[name] = vals
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax.plot(range(L), prof["probs"], marker="o", ms=3, label="Feature 1: probabilities")
    ax.plot(range(L), prof["hist"], marker="s", ms=3, label="Feature 2: histogram")
    ax.set_xlabel("layer index"); ax.set_ylabel("mean |r| over experts")
    ax.set_title(f"[{tag}] Where the signal lives (layer-wise)")
    ax.legend()
    p = os.path.join(outdir, f"{tag}_layer_profile.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[fig] {p}")


def plot_pca(Xt, yt, outdir, tag, cap=4000):
    idx = np.arange(Xt.shape[0])
    if idx.size > cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, cap, replace=False)
    comps, ev = pca_2d(Xt[idx])
    y = yt[idx]
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    ax.scatter(comps[y == 0, 0], comps[y == 0, 1], s=6, alpha=0.4, label="benign (0)")
    ax.scatter(comps[y == 1, 0], comps[y == 1, 1], s=6, alpha=0.4, label="malicious (1)")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title(f"[{tag}] Routing space, PCA (unsupervised)")
    ax.legend()
    p = os.path.join(outdir, f"{tag}_pca.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[fig] {p}")


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def run(args):
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or (args.model if args.model else "model")

    blob = load_blob(args.data)
    if args.peek:
        describe(blob)
        return

    Xt, yt = extract_turns(blob)
    M, D = Xt.shape
    n1, n0 = int((yt == 1).sum()), int((yt == 0).sum())
    print(f"[data] {M} turns  |  malicious-complied={n1}  benign-complied={n0}  |  dim={D}")

    if args.dim_probs is None:
        auto = split_from_feature_config(blob, D)
        if auto:
            args.dim_probs = auto
            print(f"[geom] probs-block size {auto} from feature_config")
    dim_probs, L, E = resolve_geometry(D, args)
    feat, layer, expert = dim_metadata(D, dim_probs, L, E)

    r = point_biserial(Xt, yt)
    auc = np.full(D, 0.5) if args.no_auc else auc_per_dim(Xt, yt)
    auc_abs = np.abs(auc - 0.5)
    m0 = Xt[yt == 0].mean(0); m1 = Xt[yt == 1].mean(0)

    # ---- per-dimension CSV ----
    order = np.argsort(-auc_abs)
    csv_path = os.path.join(args.out, f"{tag}_eda_correlations.csv")
    with open(csv_path, "w") as f:
        f.write("rank,dim,feature,layer,expert,class0_mean,class1_mean,diff,pb_r,auc,auc_abs\n")
        for rk, j in enumerate(order):
            f.write(f"{rk},{j},{feat[j]},{layer[j]},{expert[j]},"
                    f"{m0[j]:.6g},{m1[j]:.6g},{m1[j]-m0[j]:.6g},"
                    f"{r[j]:.5f},{auc[j]:.5f},{auc_abs[j]:.5f}\n")
    print(f"[csv] {csv_path}")

    # ---- figures ----
    plot_diff_maps(Xt, yt, dim_probs, L, E, args.out, tag)
    plot_corr_distribution(r, feat, args.out, tag)
    mr, ma = plot_feature_compare(r, auc, feat, args.out, tag)
    plot_layer_profile(r, feat, layer, L, E, args.out, tag)
    if not args.no_pca:
        plot_pca(Xt, yt, args.out, tag)

    # ---- summary ----
    top = order[:args.topk]
    best = order[0]
    summary = {
        "tag": tag, "n_turns": M, "n_malicious_complied": n1, "n_benign_complied": n0,
        "dim": D, "dim_probs": dim_probs, "layers": L, "experts": E,
        "mean_abs_r_probs": float(np.abs(r[feat == "probs"]).mean()),
        "mean_abs_r_hist": float(np.abs(r[feat == "hist"]).mean()),
        "mean_sep_probs": float(np.abs(auc[feat == "probs"] - 0.5).mean()),
        "mean_sep_hist": float(np.abs(auc[feat == "hist"] - 0.5).mean()),
        "best_single_dim_auc": float(max(auc[best], 1 - auc[best])),
        "best_single_dim_feature": str(feat[best]),
        "best_single_dim_layer": int(layer[best]),
        "best_single_dim_expert": int(expert[best]),
    }
    with open(os.path.join(args.out, f"{tag}_eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    md = os.path.join(args.out, f"{tag}_eda_summary.md")
    with open(md, "w") as f:
        f.write(f"# EDA summary -- {tag}\n\n")
        f.write(f"- Turns: **{M}**  (malicious-complied={n1}, benign-complied={n0}); dim={D}\n")
        f.write(f"- Mean |r| -- Feature 1 (probs): **{summary['mean_abs_r_probs']:.4f}**  |  "
                f"Feature 2 (hist): **{summary['mean_abs_r_hist']:.4f}**\n")
        f.write(f"- Mean |AUC-0.5| -- probs: **{summary['mean_sep_probs']:.4f}**  |  "
                f"hist: **{summary['mean_sep_hist']:.4f}**\n")
        f.write(f"- Best *single* dimension AUC: **{summary['best_single_dim_auc']:.3f}** "
                f"({summary['best_single_dim_feature']}, layer {summary['best_single_dim_layer']}, "
                f"expert {summary['best_single_dim_expert']})\n\n")
        f.write("### Top discriminative dimensions\n\n")
        f.write("| rank | feature | layer | expert | r | AUC |\n|---|---|---|---|---|---|\n")
        for rk, j in enumerate(top):
            f.write(f"| {rk} | {feat[j]} | {layer[j]} | {expert[j]} | "
                    f"{r[j]:+.3f} | {auc[j]:.3f} |\n")
        f.write("\n> Caveats: correlations/AUCs treat turns as independent (they are grouped "
                "by conversation) and are **univariate**. The GRU uses multivariate + temporal "
                "structure, so these numbers understate the signal it can reach. Small per-dim "
                "r is expected and does not contradict a strong detector.\n")
    print(f"[md]  {md}")

    print("\n==== quick read ====")
    print(f"mean|r|   probs={summary['mean_abs_r_probs']:.4f}   hist={summary['mean_abs_r_hist']:.4f}")
    print(f"mean sep  probs={summary['mean_sep_probs']:.4f}   hist={summary['mean_sep_hist']:.4f}")
    print(f"best single-dim AUC = {summary['best_single_dim_auc']:.3f}  "
          f"({summary['best_single_dim_feature']} L{summary['best_single_dim_layer']} "
          f"E{summary['best_single_dim_expert']})")
    if summary["best_single_dim_auc"] < 0.75:
        print("-> no single expert separates the classes: signal is distributed "
              "(consistent with 'no single unsafe expert'); motivates a learned detector.")


# ----------------------------------------------------------------------------- #
# Synthetic self-test (no torch / no real data needed)
# ----------------------------------------------------------------------------- #
def selftest():
    rng = np.random.default_rng(0)
    L, E, T = 8, 6, 3
    D = 2 * L * E
    N = 300
    y_conv = rng.integers(0, 2, N)
    X = rng.normal(0, 1, (N, T, D)).astype(np.float32)
    # plant a distributed class-1 shift in a handful of (layer,expert) cells, both blocks
    cells = [(2, 1), (5, 3), (6, 0)]
    for (l, e) in cells:
        jp = l * E + e            # probs block
        jh = L * E + l * E + e    # hist block
        X[y_conv == 1, :, jp] += 0.6
        X[y_conv == 1, :, jh] += 0.4
    # a couple of padding turns (all-zero) to exercise the drop path
    X[0, 2, :] = 0.0
    tmp = "/tmp/eda_selftest"
    os.makedirs(tmp, exist_ok=True)
    np.savez(os.path.join(tmp, "dev.npz"), X=X, y=y_conv)
    args = argparse.Namespace(
        data=os.path.join(tmp, "dev.npz"), out=tmp, tag="selftest",
        model=None, num_layers=L, num_experts=E, dim_probs=None,
        topk=10, no_auc=False, no_pca=False, peek=False,
    )
    run(args)
    produced = sorted(f for f in os.listdir(tmp) if f.startswith("selftest"))
    print("\n[selftest] produced:", produced)
    assert any(f.endswith("diffmap_probs.png") for f in produced)
    assert any(f.endswith("correlations.csv") for f in produced)
    assert any(f.endswith("feature_compare.png") for f in produced)
    print("[selftest] OK")


def build_parser():
    p = argparse.ArgumentParser(description="RoutingGuard EDA: routing difference + feature-label correlation")
    p.add_argument("--data", default="dev.pt", help="dev split (.pt or .npz). Do NOT use test.pt.")
    p.add_argument("--out", default="eda_out", help="output directory for figures/CSV")
    p.add_argument("--tag", default=None, help="label for filenames/titles (defaults to --model)")
    p.add_argument("--model", choices=list(MODEL_PRESETS), default=None,
                   help="preset (num_layers,num_experts): zaya/lfm2/phi")
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_experts", type=int, default=None)
    p.add_argument("--dim_probs", type=int, default=None,
                   help="size of Feature-1 block (defaults to D//2)")
    p.add_argument("--topk", type=int, default=15, help="rows in the top-dimensions table")
    p.add_argument("--no_auc", action="store_true", help="skip per-dim AUC (faster)")
    p.add_argument("--no_pca", action="store_true", help="skip PCA scatter")
    p.add_argument("--peek", action="store_true", help="print .pt structure and exit")
    p.add_argument("--selftest", action="store_true", help="run on synthetic data (no torch needed)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
    else:
        run(args)