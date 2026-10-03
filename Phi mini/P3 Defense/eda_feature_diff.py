"""
eda_feature_diff.py  (v2 — model-agnostic geometry)
---------------------------------------------------
Three figures for the two engineered features (integer axes, any geometry):

  <tag>_diffmap_probs.png   [ benign mean | malicious mean | difference (1-0) ]  Feature 1
  <tag>_diffmap_hist.png    [ benign mean | malicious mean | difference (1-0) ]  Feature 2
  <tag>_feature_diff_compare.png
        [ Feature 1 difference | Feature 2 difference | difference-of-the-two ]

The third panel of the comparison figure z-scores each difference map first,
then shows z(F1) - z(F2): grey where the two features agree, and coloured where
one feature shifts toward malicious more than the other (in std units).
Standardising is necessary because probs and hist live on different scales.

The console also prints whether the two features are DIFFERENT (cell-by-cell
Pearson / Spearman correlation + cosine similarity of the two difference maps).

GEOMETRY (v2): the feature vector is [ Feature 1 | Feature 2 ], each block L*E,
so D = 2*L*E and dim_probs = D/2. The three studied models have distinct
probs-block sizes, so the (L, E) geometry is detected from the data itself:

    zaya : 40 x 17  (dim_probs 680, D 1360)   # 17 = extraction width; see 16-vs-17 note
    lfm2 : 22 x 32  (dim_probs 704, D 1408)
    phi  : 32 x 16  (dim_probs 512, D 1024)

--model is now only a HINT. If it contradicts the data it is overridden with a
warning and the correct geometry is auto-detected. --num_layers/--num_experts
still force a manual override for any new model.

Reuses the loaders in eda_routing.py so the feature split matches the main EDA.
Keep this file named something OTHER than eda_routing.py.

Run (model optional — all of these work):
  python eda_feature_diff.py --data dev.pt --out out
  python eda_feature_diff.py --data dev.pt --model lfm2 --out out
  python eda_feature_diff.py --data dev.pt --model phi  --out out
  python eda_feature_diff.py --data dev.pt --num_layers 22 --num_experts 32 --out out
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

# reuse the exact loading path from the main EDA (loaders are unchanged).
# resolve_geometry is intentionally NOT imported: v2 resolves geometry locally.
from eda_routing import load_blob, extract_turns
try:
    from eda_routing import split_from_feature_config
except Exception:                       # optional; we fall back to D//2
    split_from_feature_config = None

DPI = 300

# (layers, experts) per model. F1 and F2 blocks are each L*E, so D = 2*L*E.
# Sizes are distinct across models, so dim_probs alone identifies the geometry.
MODEL_GEOMETRY = {
    "zaya": (40, 17),   # dim_probs 680  (17 = saved router width; 16-vs-17 open in methods)
    "lfm2": (22, 32),   # dim_probs 704
    "phi":  (32, 16),   # dim_probs 512
}

# forgiving aliases so --model phimoe / lfm2.5 / zaya-1 all work
MODEL_ALIASES = {
    "phimoe": "phi", "phi-mini": "phi", "phi_mini_moe": "phi", "phimini": "phi",
    "lfm": "lfm2", "lfm2.5": "lfm2", "lfm25": "lfm2", "lfm-2.5": "lfm2",
    "zaya1": "zaya", "zaya-1": "zaya", "zaya1-8b": "zaya",
}


def _norm_model(name):
    if name is None:
        return None
    key = name.strip().lower()
    return MODEL_ALIASES.get(key, key)


def resolve_geometry_local(D, blob, args):
    """Return (dim_probs, L, E) robustly, independent of --model correctness.

    Priority:
      1. dim_probs from the feature-config split if available, else D // 2.
      2. explicit --num_layers/--num_experts  (manual override, any model)
      3. --model, but ONLY if it matches dim_probs (else warn + auto-detect)
      4. auto-detect: the MODEL_GEOMETRY entry whose L*E == dim_probs
    """
    # --- 1. size of the probs block -----------------------------------------
    dim_probs = None
    if split_from_feature_config is not None:
        try:
            dim_probs = split_from_feature_config(blob, D)
        except Exception:
            dim_probs = None
    if not dim_probs:
        dim_probs = D // 2          # [F1 | F2] with equal blocks -> half of D

    # --- 2. explicit manual override ----------------------------------------
    if args.num_layers and args.num_experts:
        return dim_probs, args.num_layers, args.num_experts

    model = _norm_model(args.model)

    # --- 3. named model, only if consistent with the data -------------------
    if model in MODEL_GEOMETRY:
        l, e = MODEL_GEOMETRY[model]
        if l * e == dim_probs:
            return dim_probs, l, e
        print(f"[geom] --model {args.model} => {l}x{e}={l * e}, but this data has "
              f"dim_probs={dim_probs}. Ignoring --model and auto-detecting.")
    elif model is not None:
        print(f"[geom] unknown --model {args.model!r}; auto-detecting from data.")

    # --- 4. auto-detect from the table --------------------------------------
    for name, (l, e) in MODEL_GEOMETRY.items():
        if l * e == dim_probs:
            print(f"[geom] auto-detected geometry: {name} ({l}x{e}, dim_probs={dim_probs})")
            return dim_probs, l, e

    # nothing matched
    print(f"[geom] no known model matches dim_probs={dim_probs}. "
          f"Pass --num_layers and --num_experts explicitly.")
    return dim_probs, None, None


def _int_ticks(ax, L, E):
    xstep = max(1, int(np.ceil(E / 8)))
    ystep = max(1, int(np.ceil(L / 8)))
    ax.set_xticks(np.arange(0, E, xstep))
    ax.set_yticks(np.arange(0, L, ystep))


def _box_top(ax, M, E):
    idx = int(np.argmax(np.abs(M)))
    r0, c0 = idx // E, idx % E
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), 1, 1, fill=False,
                           edgecolor="black", linewidth=2))
    return r0, c0, float(M[r0, c0])


def plot_class_maps(m0, m1, L, E, feat_name, outdir, tag):
    """[ benign mean | malicious mean | difference (1-0) ] for one feature block."""
    a0 = m0.reshape(L, E)
    a1 = m1.reshape(L, E)
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
        _int_ticks(a, L, E)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(f"[{tag}] Feature: {feat_name}  (routing map by class)", fontsize=12)
    p = os.path.join(outdir, f"{tag}_diffmap_{feat_name}.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print(f"[fig] {p}")
    return p


def _z(M):
    return (M - M.mean()) / (M.std() + 1e-12)


def plot_feature_compare(probs_diff, hist_diff, L, E, outdir, tag):
    """[ F1 difference | F2 difference | z(F1) - z(F2) ] + correlation stats."""
    pf, hf = probs_diff.reshape(-1), hist_diff.reshape(-1)
    pearson = float(np.corrcoef(pf, hf)[0, 1])
    spear = float(spearmanr(pf, hf).correlation)
    cosine = float(np.dot(pf, hf) / (np.linalg.norm(pf) * np.linalg.norm(hf) + 1e-12))

    zdiff = _z(probs_diff) - _z(hist_diff)     # standardised disagreement map

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)

    vlim_p = np.abs(probs_diff).max() + 1e-12
    im0 = ax[0].imshow(probs_diff, aspect="auto", cmap="coolwarm", vmin=-vlim_p, vmax=vlim_p)
    lp, ep, vp = _box_top(ax[0], probs_diff, E)
    ax[0].set_title(f"Feature 1 (probs) difference\ntop L{lp}E{ep} = {vp:+.3f}", fontsize=10)
    ax[0].set_xlabel("expert"); ax[0].set_ylabel("layer"); _int_ticks(ax[0], L, E)
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    vlim_h = np.abs(hist_diff).max() + 1e-12
    im1 = ax[1].imshow(hist_diff, aspect="auto", cmap="coolwarm", vmin=-vlim_h, vmax=vlim_h)
    lh, eh, vh = _box_top(ax[1], hist_diff, E)
    ax[1].set_title(f"Feature 2 (hist) difference\ntop L{lh}E{eh} = {vh:+.3f}", fontsize=10)
    ax[1].set_xlabel("expert"); ax[1].set_ylabel("layer"); _int_ticks(ax[1], L, E)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    vlim_z = np.abs(zdiff).max() + 1e-12
    im2 = ax[2].imshow(zdiff, aspect="auto", cmap="PuOr", vmin=-vlim_z, vmax=vlim_z)
    ax[2].set_title("difference of the two features\nz(F1) - z(F2)  (grey = agree)", fontsize=10)
    ax[2].set_xlabel("expert"); ax[2].set_ylabel("layer"); _int_ticks(ax[2], L, E)
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    same_top = (lp, ep) == (lh, eh)
    fig.suptitle(f"[{tag}] Feature-difference comparison   "
                 f"Pearson r={pearson:.2f}  Spearman={spear:.2f}  cos={cosine:.2f}   "
                 f"(lead cell {'SAME' if same_top else 'different'})", fontsize=12)
    p = os.path.join(outdir, f"{tag}_feature_diff_compare.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print(f"[fig] {p}")
    return dict(pearson=pearson, spearman=spear, cosine=cosine,
                probs_top=(lp, ep, vp), hist_top=(lh, eh, vh), same_top=same_top, fig=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dev.pt (do NOT use test.pt)")
    ap.add_argument("--model", default=None,
                    help="optional hint: zaya | lfm2 | phi (aliases ok). "
                         "Overridden if it contradicts the data.")
    ap.add_argument("--num_layers", type=int, default=0,
                    help="manual geometry override (use with --num_experts)")
    ap.add_argument("--num_experts", type=int, default=0,
                    help="manual geometry override (use with --num_layers)")
    ap.add_argument("--dim_probs", type=int, default=None,
                    help="manual probs-block size (rarely needed; default D/2)")
    ap.add_argument("--out", default="feature_diff")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or (_norm_model(args.model) or "model")

    blob = load_blob(args.data)
    Xt, yt = extract_turns(blob)
    D = Xt.shape[1]

    # geometry: robust, model-agnostic
    if args.dim_probs is not None:
        dim_probs = args.dim_probs
        if args.num_layers and args.num_experts:
            L, E = args.num_layers, args.num_experts
        else:
            L = E = None
            for name, (l, e) in MODEL_GEOMETRY.items():
                if l * e == dim_probs:
                    L, E = l, e
                    break
    else:
        dim_probs, L, E = resolve_geometry_local(D, blob, args)

    if not (L and E and dim_probs == L * E and D - dim_probs == L * E):
        raise SystemExit(
            f"Cannot reshape to (L,E): D={D} dim_probs={dim_probs} L={L} E={E}.\n"
            f"  Expected D = 2*L*E (equal probs/hist blocks). Known models: "
            f"{', '.join(f'{k}={l}x{e}' for k,(l,e) in MODEL_GEOMETRY.items())}.\n"
            f"  If this is a new model, pass --num_layers and --num_experts.")

    m0 = Xt[yt == 0].mean(0)
    m1 = Xt[yt == 1].mean(0)

    # image 1 + image 2: per-feature class maps (benign | malicious | difference)
    plot_class_maps(m0[:dim_probs], m1[:dim_probs], L, E, "probs", args.out, tag)
    plot_class_maps(m0[dim_probs:], m1[dim_probs:], L, E, "hist", args.out, tag)

    # image 3: the two differences + their standardised difference
    probs_diff = (m1[:dim_probs] - m0[:dim_probs]).reshape(L, E)
    hist_diff = (m1[dim_probs:] - m0[dim_probs:]).reshape(L, E)
    s = plot_feature_compare(probs_diff, hist_diff, L, E, args.out, tag)

    print(f"\n[data] dim={D}  probs-block={dim_probs}  geometry={L}x{E}")
    print("=== Are the two features different? ===")
    print(f"  Feature 1 top diff cell : L{s['probs_top'][0]}E{s['probs_top'][1]} ({s['probs_top'][2]:+.3f})")
    print(f"  Feature 2 top diff cell : L{s['hist_top'][0]}E{s['hist_top'][1]} ({s['hist_top'][2]:+.3f})")
    print(f"  same lead cell?         : {s['same_top']}")
    print(f"  Pearson  r (per cell)   : {s['pearson']:.3f}")
    print(f"  Spearman r (per cell)   : {s['spearman']:.3f}")
    print(f"  cosine similarity       : {s['cosine']:.3f}")
    print("  high corr -> same experts (redundant);  low corr -> different experts (complementary)")


if __name__ == "__main__":
    main()