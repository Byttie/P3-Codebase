"""
ZAYA MoE Routing — EDA v3
=========================

For EACH of the four groups  ->  m2s-hyphenize, m2s-numberize, m2s-pythonize, multiturn
produce the SAME four plots:

  1. pca_clusters          PCA of routing tensors, refused vs compliant
  2. clean_score_max_proof per-layer routing probabilities sum to 1.0 (sanity)
  3. refusal_bubbles       rows = refused prompt/turn IDs, cols = all 40 layers,
                           one bubble per (row, layer) = the top-1 expert
                           (colour = expert ID, size = its routing probability)
  4. low_variance_baseline per-layer dominant expert across ALL 40 layers:
                           x = layer, y = mean global-max routing probability
                           (mean +/- std over refusals), point coloured/labelled
                           with the modal top-1 expert at that layer

and for MULTITURN additionally:

  5. context_drift_sankey  conversations flowing Compliant/Refused across turns 1->2->3
                           (the safe -> harmful context drift)

Payload schema (per .pt, dict):
  pooled_probs (L,E) rows sum to 1 | pooled_logits (L,E) | valid_mask (L,E) bool | ...
ZAYA-8B: L=40 layers, E=17 experts, top-1 routing.

NOTE: the *_refusals.json files list the refused items (single-turn -> prompt_id ;
multi-turn -> (conversation_id, turn_id)). The loaders load ALL available tensors
and label those ids refused=True; every other tensor found on disk is Compliant.
  - PCA clusters  -> uses BOTH classes (separate compliant / refused clusters)
  - Context-drift Sankey -> uses BOTH classes (Compliant -> Refused intent drift)
  - Per-layer baseline & refusal bubbles -> filter to refusals only
"""

import os
import glob
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.colors import BoundaryNorm
from sklearn.decomposition import PCA


# ==========================================================================
# CONFIG
# ==========================================================================
BASE_DIR = "."

SINGLE_TURN = {
    "m2s-hyphenize": "m2s_hyphenize_refusals.json",
    "m2s-numberize": "m2s_numberize_refusals.json",
    "m2s-pythonize": "m2s_pythonize_refusals.json",
}
MULTI_TURN_DIR = "multi-turn"
MULTI_TURN_REFUSALS = "multi_turn_refusals.json"

NUM_PROMPTS = 100
NUM_CONVS = 100
NUM_TURNS = 3

TARGET_LAYER = 20        # (unused by the per-layer baseline; kept for reference)
TOP_K = 6                # (unused by the per-layer baseline; kept for reference)
N_EXPERTS = 17           # ZAYA expert count (for consistent colour scale)


# ==========================================================================
# LOADING
# ==========================================================================
def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def probs_and_mask(payload):
    if isinstance(payload, dict):
        t = payload.get("pooled_probs", payload.get("pooled_logits"))
        mask = payload.get("valid_mask", None)
    else:
        t, mask = payload, None
    t = torch.as_tensor(t).float()
    if t.dim() == 3:
        t = t.mean(dim=1)
    elif t.dim() == 1:
        t = t.unsqueeze(0)
    probs = t.numpy()
    if mask is not None:
        mask = torch.as_tensor(mask).bool().numpy()
    return probs, mask


def dominant_expert_per_layer(probs, mask=None):
    m = probs.astype(float).copy()
    if mask is not None:
        m[~mask] = -np.inf
    return np.argmax(m, axis=1)


def _resolve_tensor_dir(base_dir, sub):
    nested = os.path.join(base_dir, sub, "moe_routing_tensors")
    return nested if os.path.isdir(nested) else os.path.join(base_dir, sub)


def _find_json(base_dir, cat, filename):
    for c in [os.path.join(base_dir, filename),
              os.path.join(base_dir, cat, filename),
              os.path.join(base_dir, cat, "single_prompt_refusals.json"),
              os.path.join(base_dir, MULTI_TURN_DIR, filename)]:
        if os.path.exists(c):
            return c
    return None


def load_single_turn_refusals(base_dir):
    out = {}
    for cat, fname in SINGLE_TURN.items():
        path = _find_json(base_dir, cat, fname)
        ids = []
        if path:
            for row in json.load(open(path, encoding="utf-8")):
                if isinstance(row, dict) and "prompt_id" in row:
                    ids.append(int(row["prompt_id"]))
                elif isinstance(row, (int, str)):
                    ids.append(int(row))
        out[cat] = sorted(set(ids))
        print(f"  {cat:<14} refused prompts: {len(out[cat])}")
    return out


def _find_multi_turn_json(base_dir):
    explicit = _find_json(base_dir, MULTI_TURN_DIR, MULTI_TURN_REFUSALS)
    if explicit:
        return explicit
    for d in (base_dir, os.path.join(base_dir, MULTI_TURN_DIR)):
        for pat in ["*multi*turn*refus*.json", "*multiturn*refus*.json",
                    "*flagged*refus*.json", "*refus*.json"]:
            for hit in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    data = json.load(open(hit, encoding="utf-8"))
                    if data and isinstance(data[0], dict) and \
                       "conversation_id" in data[0] and "turn_id" in data[0]:
                        return hit
                except Exception:
                    continue
    return None


def load_multi_turn_refusals(base_dir):
    path = _find_multi_turn_json(base_dir)
    pairs = set()
    if path:
        for r in json.load(open(path, encoding="utf-8")):
            pairs.add((int(r["conversation_id"]), int(r["turn_id"])))
        print(f"  multi-turn refusals file: {path}")
    else:
        print(f"  [warn] multi-turn refusals JSON not found under '{base_dir}'.")
    print(f"  multi-turn refused (conv,turn) pairs: {len(pairs)}")
    return pairs


# ---- sample builders: each sample = {label, probs, mask, refused, conv?, turn?} ----
def load_single_turn_samples(base_dir, cat, refused_ids):
    """Load ALL available prompt tensors; label refusals from the JSON ids.

    refused=True comes from *_refusals.json; every other tensor found on disk is
    Compliant. PCA uses both classes (separate clusters); the per-layer baseline
    and refusal bubbles filter to refusals only.
    """
    tdir = _resolve_tensor_dir(base_dir, cat)
    refused_set = set(refused_ids)
    samples = []
    for i in range(NUM_PROMPTS):
        fp = os.path.join(tdir, f"prompt_{i:04d}.pt")
        if not os.path.exists(fp):
            continue
        probs, mask = probs_and_mask(load_payload(fp))
        samples.append({"label": i, "probs": probs, "mask": mask,
                        "refused": i in refused_set})
    n_ref = sum(s["refused"] for s in samples)
    print(f"  [{cat}] loaded {len(samples)} tensors "
          f"({n_ref} refused, {len(samples) - n_ref} compliant)")
    return samples


def load_multi_turn_samples(base_dir, refusals):
    """Load ALL available (conv,turn) tensors; label refusals from the JSON pairs.

    refused=True comes from multi_turn_refusals.json; every other turn tensor
    found on disk is Compliant. The Sankey needs both states to show the
    Compliant -> Refused context drift across turns.
    """
    tdir = _resolve_tensor_dir(base_dir, MULTI_TURN_DIR)
    samples = []
    for c in range(NUM_CONVS):
        for t in range(1, NUM_TURNS + 1):
            fp = os.path.join(tdir, f"conv_{c:04d}_turn_{t:02d}.pt")
            if not os.path.exists(fp):
                continue
            probs, mask = probs_and_mask(load_payload(fp))
            samples.append({"label": f"c{c}t{t}", "probs": probs, "mask": mask,
                            "refused": (c, t) in refusals, "conv": c, "turn": t})
    n_ref = sum(s["refused"] for s in samples)
    print(f"  [multiturn] loaded {len(samples)} tensors "
          f"({n_ref} refused, {len(samples) - n_ref} compliant)")
    return samples


# ==========================================================================
# 1. PCA CLUSTERS  (refused vs compliant)
# ==========================================================================
def _flatten_pad(samples):
    flat = [s["probs"].flatten() for s in samples]
    L = max(len(f) for f in flat)
    return np.array([np.pad(f, (0, L - len(f))) for f in flat])


def plot_pca_clusters(samples, name, save_dir):
    if len(samples) < 2:
        print(f"  [{name}] pca skipped (too few samples)"); return
    X = _flatten_pad(samples)
    XY = PCA(n_components=2).fit_transform(X)
    ref = np.array([s["refused"] for s in samples])

    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(XY[~ref, 0], XY[~ref, 1], s=28, c="blue", alpha=0.7, label="Compliant")
    plt.scatter(XY[ref, 0], XY[ref, 1], s=28, c="red", alpha=0.7, label="Refused")
    plt.title(f"PCA of ZAYA MoE Routing Tensors — {name}", fontsize=13)
    plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, f"{name}_pca_clusters.png")
    plt.savefig(out); plt.close()
    print(f"  [{name}] -> {out}")


# ==========================================================================
# 2. CLEAN SCORE MAX PROOF  (per-layer sum -> 1.0)
# ==========================================================================
def plot_clean_score_max_proof(samples, name, save_dir):
    chosen = next((s for s in samples if s["refused"]), samples[0] if samples else None)
    if chosen is None:
        print(f"  [{name}] sum-proof skipped (no samples)"); return
    layer_sums = chosen["probs"].sum(axis=1)
    n = len(layer_sums)

    plt.figure(figsize=(12, 4), dpi=200)
    plt.plot(range(n), layer_sums, marker="o", ms=7, color="#006666", lw=2.5, zorder=3)
    plt.axhline(1.0, color="gray", ls="--", lw=1, zorder=2)
    plt.title(f"Mathematical Proof: Layer Routing Decisions Sum to 1.0 — {name}",
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel(f"MoE Layer (0 to {n - 1})", fontweight="bold")
    plt.ylabel("Sum of Probabilities", fontweight="bold")
    plt.ylim(0.95, 1.05)
    plt.grid(True, ls="-", alpha=0.3, zorder=1)
    plt.xticks(range(0, n, max(1, n // 10)))
    plt.tight_layout()
    out = os.path.join(save_dir, f"{name}_clean_score_max_proof.png")
    plt.savefig(out); plt.close()
    print(f"  [{name}] -> {out}")


# ==========================================================================
# 3. REFUSAL BUBBLES  (rows = refusals, cols = layers, bubble = top-1 expert)
# ==========================================================================
def plot_refusal_bubbles(samples, name, save_dir):
    refused = [s for s in samples if s["refused"]]
    if not refused:
        print(f"  [{name}] bubbles skipped (no refusals)"); return

    rows_expert, rows_prob, ylabels = [], [], []
    for s in refused:
        de = dominant_expert_per_layer(s["probs"], s["mask"])       # (L,)
        prob = s["probs"][np.arange(len(de)), de]                   # its probability
        rows_expert.append(de); rows_prob.append(prob); ylabels.append(s["label"])

    E = np.vstack(rows_expert)                                       # (P, L)
    Pr = np.vstack(rows_prob)
    P, L = E.shape
    n_experts = max(int(E.max()) + 1, N_EXPERTS)

    cmap = plt.get_cmap("tab20", n_experts)
    norm = BoundaryNorm(np.arange(-0.5, n_experts + 0.5), cmap.N)

    xs = np.tile(np.arange(L), P)
    ys = np.repeat(np.arange(P), L)
    cols = E.flatten()
    sizes = 20 + 260 * Pr.flatten()          # bubble size = top-1 routing probability

    fig, ax = plt.subplots(figsize=(15, max(5, 0.30 * P + 2)), dpi=200)
    sc = ax.scatter(xs, ys, c=cols, s=sizes, cmap=cmap, norm=norm,
                    edgecolors="black", linewidth=0.3, alpha=0.9)
    ax.set_title(f"Refusal Bubbles — {name}  (top-1 expert per layer; "
                 f"size = routing prob.)  [{P} refusals x {L} layers]",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("MoE Layer (0 .. {})".format(L - 1), fontsize=11)
    ax.set_ylabel("Refused Prompt / Turn ID", fontsize=11)
    ax.set_xticks(range(0, L, 2))
    step = max(1, P // 35)
    ax.set_yticks(range(0, P, step))
    ax.set_yticklabels([ylabels[i] for i in range(0, P, step)], fontsize=7)
    ax.set_ylim(-1, P)
    ax.grid(True, ls=":", alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, ticks=range(n_experts), pad=0.01)
    cb.set_label("Top-1 Expert ID", fontsize=10)
    plt.tight_layout()
    out = os.path.join(save_dir, f"{name}_refusal_bubbles.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()

    # report the modal expert per layer for quick reading
    modal = [np.bincount(E[:, l]).argmax() for l in range(L)]
    print(f"  [{name}] -> {out}")
    print(f"      modal top-1 expert by layer: "
          f"{{{', '.join(f'L{l}:E{modal[l]}' for l in range(0, L, 5))}}}")


# ==========================================================================
# 4. LOW-VARIANCE BASELINE  (per-layer dominant expert across ALL layers)
# ==========================================================================
def plot_low_variance_baseline(samples, name, save_dir, annotate_agreement=False):
    """
    Per-layer refusal baseline spanning every MoE layer.

      x-axis : layer index (0 .. L-1)
      y-axis : mean of the global-max (top-1) routing probability across refused
               prompts, with +/- std error bars
      colour + label of each point : the MODAL top-1 expert at that layer
                                     (the expert most refusals routed to there)

    The std captures prompt-to-prompt spread in the top-1 probability, which
    widens at layers where refusals disagree about the dominant expert.

    Set annotate_agreement=True to also print, under each expert label, the
    percentage of refusals that actually routed to that modal expert.
    """
    refused = [s for s in samples if s["refused"]]
    if not refused:
        print(f"  [{name}] baseline skipped (no refusals)"); return

    n_layers = min(s["probs"].shape[0] for s in refused)
    n_prompts = len(refused)

    # per-prompt, per-layer: top-1 expert id and its (max) routing probability
    top_expert = np.zeros((n_prompts, n_layers), dtype=int)    # (P, L)
    top_prob = np.zeros((n_prompts, n_layers), dtype=float)    # (P, L)
    for i, s in enumerate(refused):
        p = s["probs"][:n_layers]                              # (L, E)
        msk = s["mask"][:n_layers] if s.get("mask") is not None else None
        de = dominant_expert_per_layer(p, msk)                 # (L,)
        top_expert[i] = de
        top_prob[i] = p[np.arange(n_layers), de]

    mean_prob = top_prob.mean(axis=0)                          # (L,)
    std_prob = top_prob.std(axis=0)                            # (L,)

    # modal (most frequent) top-1 expert per layer + agreement share
    modal_expert = np.zeros(n_layers, dtype=int)
    modal_share = np.zeros(n_layers, dtype=float)
    for l in range(n_layers):
        counts = np.bincount(top_expert[:, l], minlength=N_EXPERTS)
        modal_expert[l] = counts.argmax()
        modal_share[l] = counts.max() / n_prompts

    # --- OPTIONAL: make std measure disagreement about WHICH expert instead ---
    # Uncomment to plot the mean/std of the modal expert's probability across
    # ALL prompts (deviating prompts contribute their low prob -> wider std):
    # for l in range(n_layers):
    #     col = np.array([s["probs"][l, modal_expert[l]] for s in refused])
    #     mean_prob[l], std_prob[l] = col.mean(), col.std()

    # ---- plot ----
    n_experts = max(int(top_expert.max()) + 1, N_EXPERTS)
    cmap = plt.get_cmap("tab20", n_experts)
    norm = BoundaryNorm(np.arange(-0.5, n_experts + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(18, 7), dpi=200)
    x = np.arange(n_layers)

    # error bars first (behind the coloured points)
    ax.errorbar(x, mean_prob, yerr=std_prob, fmt="none",
                ecolor="black", elinewidth=1.3, capsize=3, capthick=1.3,
                alpha=0.65, zorder=2)
    # points coloured by modal expert
    sc = ax.scatter(x, mean_prob, c=modal_expert, cmap=cmap, norm=norm,
                    s=130, edgecolors="black", linewidth=0.6, zorder=3)

    # label each point with its dominant expert id (and optional agreement %)
    for l in range(n_layers):
        lbl = f"E{modal_expert[l]}"
        if annotate_agreement:
            lbl += f"\n{modal_share[l]*100:.0f}%"
        ax.annotate(lbl, (x[l], mean_prob[l] + std_prob[l]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color=cmap(norm(modal_expert[l])))

    ax.set_title(
        f"Per-Layer Refusal Baseline — {name}\n"
        f"Dominant expert & global-max routing probability "
        f"(mean \u00b1 std over {n_prompts} refusals, {n_layers} layers)",
        fontweight="bold", fontsize=14)
    ax.set_xlabel("MoE Layer", fontsize=12)
    ax.set_ylabel("Global Max Routing Probability", fontsize=12)
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_ylim(0, min(1.05, float((mean_prob + std_prob).max()) * 1.30))
    ax.grid(True, ls=":", alpha=0.5)

    cb = fig.colorbar(sc, ax=ax, ticks=range(n_experts), pad=0.01)
    cb.set_label("Modal Top-1 Expert ID", fontsize=10)

    plt.tight_layout()
    out = os.path.join(save_dir, f"{name}_low_variance_baseline.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  [{name}] -> {out}")


# ==========================================================================
# 5. CONTEXT-DRIFT SANKEY  (multiturn only)
# ==========================================================================
def _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.55):
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor=color,
                                    edgecolor="none", alpha=alpha))


def plot_context_drift_sankey(samples, save_dir):
    """Flow of conversations through Compliant/Refused states across turns 1->2->3."""
    conv_state = {}
    for s in samples:
        conv_state.setdefault(s["conv"], {})[s["turn"]] = "R" if s["refused"] else "C"
    if not conv_state:
        print("  sankey skipped (no multi-turn samples)"); return

    turns = list(range(1, NUM_TURNS + 1))
    states = ["C", "R"]                              # C at bottom, R on top
    node_color = {"C": "#2ca02c", "R": "#d62728"}
    link_color = {("C", "C"): "#a7d8a0", ("C", "R"): "#ff8c42",   # C->R = drift into harmful
                  ("R", "R"): "#b03030", ("R", "C"): "#8fb3d9"}   # R->C = recovered

    def stage_counts(t):
        c = {"C": 0, "R": 0}
        for cs in conv_state.values():
            if t in cs:
                c[cs[t]] += 1
        return c

    def transitions(ta, tb):
        d = {}
        for cs in conv_state.values():
            if ta in cs and tb in cs:
                k = (cs[ta], cs[tb]); d[k] = d.get(k, 0) + 1
        return d

    counts = [stage_counts(t) for t in turns]
    max_total = max(sum(c.values()) for c in counts) or 1
    scale = 0.82 / max_total
    gap = 0.08
    node_w = 0.05
    xstage = [0.0, 1.0, 2.0]

    span = {}                                        # (stage_idx, state) -> (y0, y1)
    for i, c in enumerate(counts):
        y = 0.0
        for st in states:
            h = c[st] * scale
            span[(i, st)] = (y, y + h)
            y += h + gap

    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)

    # nodes
    for i in range(len(turns)):
        for st in states:
            y0, y1 = span[(i, st)]
            if y1 - y0 <= 0:
                continue
            ax.add_patch(mpatches.Rectangle((xstage[i], y0), node_w, y1 - y0,
                                            facecolor=node_color[st], edgecolor="black", lw=0.5))
            ax.text(xstage[i] + node_w / 2, y1 + 0.012,
                    f"{'Refused' if st=='R' else 'Compliant'}: {counts[i][st]}",
                    ha="center", va="bottom", fontsize=8,
                    color=node_color[st], fontweight="bold")

    # links
    for i in range(len(turns) - 1):
        tr = transitions(turns[i], turns[i + 1])
        out_off = {st: span[(i, st)][0] for st in states}
        in_off = {st: span[(i + 1, st)][0] for st in states}
        for src in states:
            for dst in states:
                cnt = tr.get((src, dst), 0)
                if cnt <= 0:
                    continue
                h = cnt * scale
                y0a = out_off[src]; y0b = y0a + h; out_off[src] = y0b
                y1a = in_off[dst]; y1b = y1a + h; in_off[dst] = y1b
                _ribbon(ax, xstage[i] + node_w, xstage[i + 1],
                        y0a, y0b, y1a, y1b, link_color[(src, dst)])

    for i, t in enumerate(turns):
        ax.text(xstage[i] + node_w / 2, -0.06, f"Turn {t}", ha="center",
                va="top", fontsize=12, fontweight="bold")

    legend = [mpatches.Patch(color="#ff8c42", label="Compliant → Refused (drift into harmful)"),
              mpatches.Patch(color="#b03030", label="Refused → Refused"),
              mpatches.Patch(color="#a7d8a0", label="Compliant → Compliant"),
              mpatches.Patch(color="#8fb3d9", label="Refused → Compliant (recovered)")]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, fontsize=9, frameon=False)

    ax.set_xlim(-0.15, 2.20); ax.set_ylim(-0.12, 1.02)
    ax.axis("off")
    ax.set_title("Multi-Turn Context Drift (Safe → Harmful): "
                 "Conversation Flow Across Turns", fontweight="bold", fontsize=13)
    plt.tight_layout()
    out = os.path.join(save_dir, "multiturn_context_drift_sankey.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


# ==========================================================================
# DRIVER
# ==========================================================================
def run_group(samples, name, save_dir, sankey=False):
    if not samples:
        print(f"  [{name}] no tensors found; skipped")
        return
    print(f"\n--- {name}  ({len(samples)} tensors, "
          f"{sum(s['refused'] for s in samples)} refused) ---")
    plot_pca_clusters(samples, name, save_dir)
    plot_clean_score_max_proof(samples, name, save_dir)
    plot_refusal_bubbles(samples, name, save_dir)
    plot_low_variance_baseline(samples, name, save_dir)
    if sankey:
        plot_context_drift_sankey(samples, save_dir)


def main(base_dir=BASE_DIR):
    vis = os.path.join(base_dir, "visualizations")
    os.makedirs(vis, exist_ok=True)

    print("=== Refusal labels ===")
    single_ref = load_single_turn_refusals(base_dir)
    mt_pairs = load_multi_turn_refusals(base_dir)

    for cat, ids in single_ref.items():
        run_group(load_single_turn_samples(base_dir, cat, ids), cat, vis)

    run_group(load_multi_turn_samples(base_dir, mt_pairs), "multiturn", vis, sankey=True)

    print(f"\nDone. See {vis}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=BASE_DIR)
    args = ap.parse_args()
    main(args.base_dir)