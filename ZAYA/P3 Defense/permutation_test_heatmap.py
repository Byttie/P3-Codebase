"""
permutation_test_heatmap.py
---------------------------
Tests whether the complied-vs-refused routing DIFFERENCE is bigger than chance.

The difference heatmap showed cells with |Delta| up to ~0.2. But with only ~18-33
refused examples, some difference is expected from sampling noise alone. This test
builds the NULL distribution: shuffle the complied/refused labels many times,
recompute the difference map each time, and see how extreme a difference chance
produces. Observed cells that exceed the null are statistically significant.

Outputs:
  * a per-cell p-value map (fraction of shuffles whose |Delta| >= observed |Delta|)
  * the count of cells significant at p<0.05 (and how many survive FDR correction)
  * a GLOBAL p-value: is the observed map's total |Delta| beyond the null total?
  * a masked heatmap PNG: real Delta with non-significant cells greyed out

Uses the peak-token routing map (topk_token_probs[:, 0, :]) — the same feature the
heatmap was built from. Run in the P3 Defense root.

Usage:
  python permutation_test_heatmap.py --family multi-turn --pool topk --n_perm 1000
  python permutation_test_heatmap.py --family m2s-pythonize --pool topk --n_perm 1000
"""

import argparse
import glob
import os
import re
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

N_LAYERS, N_EXPERTS = 40, 17

POOL_DIRNAME = {"mean": "mean_pool_tensors", "topk": "topk_pool_tensors"}
REFUSAL_FILES = {
    "m2s-pythonize": ["m2s_pythonize_refusals.json"],
    "multi-turn":    ["multi_turn_refusals.json"],
}


def peak_map(payload):
    if "topk_token_probs" in payload:
        t = payload["topk_token_probs"]
        t = t.float() if torch.is_tensor(t) else torch.as_tensor(t).float()
        return t[:, 0, :].numpy()
    p = payload["pooled_probs"]
    p = p.float() if torch.is_tensor(p) else torch.as_tensor(p).float()
    return p.numpy()


def load_refused(refusal_dir, family):
    for name in REFUSAL_FILES[family]:
        p = os.path.join(refusal_dir, name)
        if os.path.exists(p):
            rows = json.load(open(p, encoding="utf-8"))
            if family == "m2s-pythonize":
                return {("m2s", int(r["prompt_id"])) for r in rows}
            return {("mt", int(r["conversation_id"])) for r in rows}
    return set()


def load_maps(root, family, pool, refusal_dir):
    tdir = os.path.join(root, POOL_DIRNAME[pool], family, "moe_routing_tensors")
    refused = load_refused(refusal_dir, family)
    comp, ref = [], []
    if family == "m2s-pythonize":
        for fp in sorted(glob.glob(os.path.join(tdir, "prompt_*.pt"))):
            pid = int(re.search(r"prompt_(\d+)", fp).group(1))
            m = _fix(peak_map(torch.load(fp, map_location="cpu", weights_only=False)))
            (ref if ("m2s", pid) in refused else comp).append(m)
    else:
        # one map per conversation = mean of its turns
        conv = {}
        for fp in sorted(glob.glob(os.path.join(tdir, "conv_*_turn_*.pt"))):
            cid = int(re.search(r"conv_(\d+)", fp).group(1))
            m = _fix(peak_map(torch.load(fp, map_location="cpu", weights_only=False)))
            conv.setdefault(cid, []).append(m)
        for cid, maps in conv.items():
            avg = np.mean(maps, axis=0)
            (ref if ("mt", cid) in refused else comp).append(avg)
    return np.array(comp), np.array(ref)


def _fix(m):
    if m.shape[0] < N_LAYERS or m.shape[1] < N_EXPERTS:
        pad = np.zeros((N_LAYERS, N_EXPERTS)); pad[:m.shape[0], :m.shape[1]] = m; return pad
    return m[:N_LAYERS, :N_EXPERTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--refusal_dir", default="refusals")
    ap.add_argument("--family", required=True, choices=["m2s-pythonize", "multi-turn"])
    ap.add_argument("--pool", default="topk", choices=["mean", "topk"])
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    comp, ref = load_maps(a.root, a.family, a.pool, a.refusal_dir)
    n_c, n_r = len(comp), len(ref)
    if n_c == 0 or n_r == 0:
        raise SystemExit(f"Need both classes; got complied={n_c} refused={n_r}")
    print(f"{a.family} / {a.pool}:  complied={n_c}  refused={n_r}")

    allm = np.concatenate([comp, ref], axis=0)      # (N, L, E)
    N = allm.shape[0]
    labels = np.array([1] * n_c + [0] * n_r)         # 1=complied

    observed = comp.mean(0) - ref.mean(0)            # (L, E)
    obs_abs = np.abs(observed)

    rng = np.random.default_rng(a.seed)
    ge_count = np.zeros_like(obs_abs)                # cells: null |Δ| >= observed
    null_totals = np.empty(a.n_perm)
    obs_total = obs_abs.sum()

    for i in range(a.n_perm):
        perm = rng.permutation(N)
        pl = labels[perm]
        d = allm[pl == 1].mean(0) - allm[pl == 0].mean(0)
        da = np.abs(d)
        ge_count += (da >= obs_abs)
        null_totals[i] = da.sum()

    pmap = (ge_count + 1) / (a.n_perm + 1)           # per-cell p-value
    sig = pmap < a.alpha
    n_sig = int(sig.sum())

    # Benjamini-Hochberg FDR correction across all cells
    flat = pmap.flatten()
    order = np.argsort(flat)
    ranked = flat[order]
    m = len(flat)
    thresh = a.alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    fdr_sig = np.zeros(m, dtype=bool)
    if k > 0:
        fdr_sig[order[:k]] = True
    fdr_sig = fdr_sig.reshape(pmap.shape)
    n_fdr = int(fdr_sig.sum())

    global_p = (np.sum(null_totals >= obs_total) + 1) / (a.n_perm + 1)

    print(f"\n=== Permutation test ({a.n_perm} shuffles) ===")
    print(f"  cells significant p<{a.alpha} (uncorrected): {n_sig} / {N_LAYERS*N_EXPERTS}")
    print(f"  cells significant after FDR correction     : {n_fdr}")
    print(f"  GLOBAL p-value (total |Δ| vs null)         : {global_p:.4f}")
    print()
    if n_fdr > 0 and global_p < a.alpha:
        print("  SIGNIFICANT. The routing difference exceeds chance. The surviving")
        print("  cells localise where complied vs refused routing genuinely diverges.")
        ys, xs = np.where(fdr_sig)
        print("  Surviving (layer, expert, Δ):")
        for l, e in sorted(zip(ys, xs), key=lambda t: -abs(observed[t]))[:12]:
            print(f"      L{l:>2} E{e:>2}   Δ={observed[l, e]:+.3f}  p={pmap[l, e]:.4f}")
    elif global_p < a.alpha:
        print("  GLOBAL significance but NO cells survive FDR. The difference is real")
        print("  in aggregate but too diffuse to localise to specific experts.")
    else:
        print("  NOT SIGNIFICANT. The observed heatmap difference is within what random")
        print("  labelling produces. With this sample size the routing-distribution")
        print("  difference cannot be distinguished from noise. (This is a valid,")
        print("  reportable finding — it means H2 is not supported at this n.)")

    # masked heatmap: real Δ with non-significant cells greyed
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    masked = np.where(fdr_sig, observed, np.nan)
    ax.imshow(np.zeros_like(observed).T, cmap="Greys", alpha=0.12,
              origin="lower", extent=[0, N_LAYERS, -0.5, N_EXPERTS - 0.5], aspect="auto")
    vmax = max(abs(observed.min()), abs(observed.max())) or 1.0
    im = ax.imshow(masked.T, cmap="RdBu_r", origin="lower",
                   extent=[0, N_LAYERS, -0.5, N_EXPERTS - 0.5], aspect="auto",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
                   interpolation="nearest")
    ax.set_title(f"Significant routing differences (FDR p<{a.alpha}) — {a.family}/{a.pool}\n"
                 f"{n_fdr} cells survive · global p={global_p:.4f} · grey = not significant",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("MoE Layer"); ax.set_ylabel("Expert ID")
    ax.set_yticks(range(N_EXPERTS))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ prob. (complied - refused)")
    plt.tight_layout()
    out = os.path.join(a.root, "visualizations",
                       f"{a.family}_{a.pool}_permutation_significance.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"\n  masked heatmap -> {out}")


if __name__ == "__main__":
    main()