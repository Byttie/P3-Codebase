"""
eda_topk_heatmap.py
-------------------
Hypothesis-2 test on the TOP-K tensors: do complied (jailbroken) prompts route
to different experts than refused (safe) prompts, once we look at the peak spike
token instead of the diluted mean?

For each prompt tensor, topk_token_probs has shape (L, KMAX, E). Index [:, 0, :]
is the PEAK token per layer — the single highest-routing token's full 17-expert
distribution at each of the 40 layers. That's the "top-1" (peak-token) reading.

We average that (L, E) peak map across all prompts in each class, giving:
    complied_map  (L, E)
    refused_map   (L, E)
    diff_map = complied - refused   (L, E)

Three heatmaps are written. The DIFFERENCE map is the experiment: bright/dark
cells are (layer, expert) pairs where jailbroken vs refused routing diverges. A
near-blank diff map means top-1 routing does NOT separate the classes (consistent
with the bubble plots); structured cells localise where it does.

Run in the ZAYA root (same folder as eda.py). Uses the *_topk tensor dirs.
"""

import os
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_EXPERTS = 17
N_LAYERS = 40

# family -> (tensor-dir globs to try, refusals-json candidates, id kind)
FAMILIES = {
    "m2s-pythonize_topk": dict(kind="m2s",
        refus=["m2s_pythonize_refusals.json", "single_prompt_refusals.json"]),
    "multi-turn_topk": dict(kind="multi",
        refus=["multi_turn_refusals.json", "flagged_refusals.json"]),
}


def _resolve_family_dir(base, cat):
    direct = os.path.join(base, cat)
    if os.path.isdir(direct):
        return direct
    prefix = cat.replace("_topk", "").replace("_mean", "")
    nested = os.path.join(base, prefix, cat)
    return nested if os.path.isdir(nested) else direct


def _tensor_dir(base, cat):
    fam = _resolve_family_dir(base, cat)
    for sub in ("moe_routing_tensors_topk", "moe_routing_tensors"):
        c = os.path.join(fam, sub)
        if os.path.isdir(c):
            return c
    return fam


def _find_refusals(base, cat, candidates):
    fam = _resolve_family_dir(base, cat)
    for name in candidates:
        for c in (os.path.join(fam, name), os.path.join(base, name),
                  os.path.join(base, cat, name)):
            if os.path.exists(c):
                return c
    # last resort: any *refus*.json in the family dir
    for hit in glob.glob(os.path.join(fam, "*refus*.json")):
        return hit
    return None


def _peak_map(payload):
    """(L, E) peak-token routing map = topk_token_probs[:, 0, :].
    Falls back to pooled_probs (mean) if top-K is absent, with a flag."""
    if "topk_token_probs" in payload:
        t = payload["topk_token_probs"]
        t = t.float() if torch.is_tensor(t) else torch.as_tensor(t).float()
        return t[:, 0, :].numpy(), True          # (L, E), peak token
    # fallback so the script still runs on mean-only tensors
    p = payload["pooled_probs"]
    p = p.float() if torch.is_tensor(p) else torch.as_tensor(p).float()
    return p.numpy(), False


def _load_refused_ids(path, kind):
    if not path:
        return set()
    rows = json.load(open(path, encoding="utf-8"))
    if kind == "m2s":
        return {int(r["prompt_id"]) if isinstance(r, dict) else int(r) for r in rows}
    return {(int(r["conversation_id"]), int(r["turn_id"])) for r in rows}


def _iter_tensors(tdir, kind):
    if kind == "m2s":
        for fp in sorted(glob.glob(os.path.join(tdir, "prompt_*.pt"))):
            pid = int(os.path.basename(fp).split("_")[1].split(".")[0])
            yield pid, fp
    else:
        for fp in sorted(glob.glob(os.path.join(tdir, "conv_*_turn_*.pt"))):
            b = os.path.basename(fp)
            c = int(b.split("_")[1]); t = int(b.split("_")[3].split(".")[0])
            yield (c, t), fp


def build_maps(base, cat, cfg):
    tdir = _tensor_dir(base, cat)
    refused_ids = _load_refused_ids(_find_refusals(base, cat, cfg["refus"]), cfg["kind"])

    comp_sum = np.zeros((N_LAYERS, N_EXPERTS))
    ref_sum = np.zeros((N_LAYERS, N_EXPERTS))
    n_comp = n_ref = 0
    used_topk = True

    for key, fp in _iter_tensors(tdir, cfg["kind"]):
        payload = torch.load(fp, map_location="cpu", weights_only=False)
        m, is_topk = _peak_map(payload)
        used_topk &= is_topk
        if m.shape[0] < N_LAYERS or m.shape[1] < N_EXPERTS:
            pad = np.zeros((N_LAYERS, N_EXPERTS)); pad[:m.shape[0], :m.shape[1]] = m; m = pad
        if key in refused_ids:
            ref_sum += m; n_ref += 1
        else:
            comp_sum += m; n_comp += 1

    if n_comp == 0 and n_ref == 0:
        print(f"  [{cat}] no tensors found under {tdir}")
        return None
    comp = comp_sum / max(1, n_comp)
    ref = ref_sum / max(1, n_ref)
    print(f"  [{cat}] complied={n_comp} refused={n_ref} "
          f"({'top-K peak' if used_topk else 'MEAN fallback (no topk_token_probs!)'})")
    return dict(comp=comp, ref=ref, n_comp=n_comp, n_ref=n_ref, topk=used_topk)


def plot_maps(cat, maps, save_dir):
    comp, ref = maps["comp"], maps["ref"]
    diff = comp - ref

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=180)
    ext = [0, N_LAYERS, -0.5, N_EXPERTS - 0.5]

    for ax, data, title, cmap, norm in [
        (axes[0], comp, f"Complied (n={maps['n_comp']})", "viridis", None),
        (axes[1], ref, f"Refused (n={maps['n_ref']})", "viridis", None),
        (axes[2], diff, "Difference (Complied - Refused)", "RdBu_r",
         TwoSlopeNorm(vcenter=0.0, vmin=diff.min(), vmax=diff.max())),
    ]:
        im = ax.imshow(data.T, aspect="auto", origin="lower", extent=ext,
                       cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xlabel("MoE Layer")
        ax.set_ylabel("Expert ID")
        ax.set_yticks(range(N_EXPERTS))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="peak-token routing prob." if norm is None else "Δ prob.")

    fig.suptitle(f"Top-K peak-token routing — {cat}   "
                 f"(layer x expert, class-averaged)",
                 fontweight="bold", fontsize=15)
    plt.tight_layout()
    out = os.path.join(save_dir, f"{cat}_topk_peak_heatmap.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  [{cat}] -> {out}")

    # quick quantitative read-out of the difference map
    flat = np.abs(diff)
    top = np.dstack(np.unravel_index(np.argsort(flat, axis=None)[-6:], diff.shape))[0]
    print(f"  [{cat}] largest complied-refused gaps (layer, expert, Δ):")
    for l, e in reversed(top):
        print(f"      L{l:>2} E{e:>2}  Δ={diff[l, e]:+.3f}")
    print(f"  [{cat}] mean |Δ| over all cells = {flat.mean():.4f}  "
          f"(near 0 => top-1 routing does NOT separate the classes)")


def main():
    save_dir = os.path.join(BASE_DIR, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    print("=== Top-K peak-token routing heatmaps ===")
    for cat, cfg in FAMILIES.items():
        maps = build_maps(BASE_DIR, cat, cfg)
        if maps:
            plot_maps(cat, maps, save_dir)
    print(f"\nDone. See {save_dir}")


if __name__ == "__main__":
    main()