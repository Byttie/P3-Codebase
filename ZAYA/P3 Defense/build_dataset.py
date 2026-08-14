"""
build_dataset.py  (v3 — layout-aware)
-------------------------------------
Assemble a unified training set from the tensors your ZAYA extraction wrote.

What changed in v3 (consistency with the real folder layout):
  * `--root` is now the PROJECT root (e.g. "E:/P3 Defense"), not a single
    tensor dir. Under it the code expects the layout you actually have:

        <root>/mean_pool_tensors/<family>/moe_routing_tensors/*.pt
        <root>/topk_pool_tensors/<family>/moe_routing_tensors/*.pt
        <root>/refusals/*.json

    Family folders use the on-disk names ("m2s-pythonize", "multi-turn").
  * The pooling MODE picks the root: `--local_pool mean` reads mean_pool_tensors,
    `--local_pool topk` (and max == topk K=1) reads topk_pool_tensors. A top-K
    extraction also stores pooled_probs, so if the preferred root is missing a
    family the resolver falls back to the other root and says so once.

Everything downstream (labels, padding, grouping) is unchanged. Still supports:
  * --families to restrict to a subset (e.g. pythonize + multiturn only)
  * local pooling choice (--local_pool mean|topk, --K) via FeatureConfig
  * exposes assemble(...) so a K-sweep can build in-process.
"""

import json, re, argparse
from pathlib import Path
import torch

from routing_features import FeatureConfig, payload_to_vector

M2S_VARIANTS = ["pythonize", "hyphenize", "numberize"]
BENIGN_GROUP_OFFSET = 100000

# ---------------------------------------------------------------------------
# On-disk layout.  Edit these three maps if your folder names differ.
# ---------------------------------------------------------------------------
POOL_DIRNAME = {"mean": "mean_pool_tensors", "topk": "topk_pool_tensors"}
FAMILY_DIRNAME = {
    "pythonize": "m2s-pythonize",
    "hyphenize": "m2s-hyphenize",
    "numberize": "m2s-numberize",
    "multiturn": "multi-turn",
    "benign":    "benign",
}
TENSOR_SUBDIR = "moe_routing_tensors"     # extractors write .pt here; "" also accepted

_FALLBACK_NOTED = set()


def _has_tensors(d):
    return d.is_dir() and (any(d.glob("prompt_*.pt")) or any(d.glob("conv_*_turn_*.pt")))


def resolve_tensor_dir(project_root, pool, family):
    """Return the directory that actually holds this family's .pt files.

    pool='mean' prefers mean_pool_tensors; 'topk' prefers topk_pool_tensors.
    The .pt may sit directly in the family folder or under moe_routing_tensors/.

    The fallback across roots is ASYMMETRIC on purpose:
      * 'mean' MAY borrow the top-K root — top-K files also carry pooled_probs,
        so reading mean from them is information-preserving.
      * 'topk' must NEVER borrow the mean root — mean files dropped the token
        axis, so there is nothing to top-K. We instead return the (empty)
        top-K path so the caller raises a clear "run the top-K extraction" error
        rather than silently degrading to mean and looking like a real top-K run.
    A one-time notice is printed whenever a fallback is taken.
    """
    project_root = Path(project_root)
    fam = FAMILY_DIRNAME.get(family, family)
    pref = "topk" if pool == "topk" else "mean"
    order = [pref] if pool == "topk" else [pref, "topk"]
    for p in order:
        for base in (project_root / POOL_DIRNAME[p] / fam / TENSOR_SUBDIR,
                     project_root / POOL_DIRNAME[p] / fam):
            if _has_tensors(base):
                if p != pref and (pool, family) not in _FALLBACK_NOTED:
                    print(f"[build] {family}: no tensors under {POOL_DIRNAME[pref]}/, "
                          f"using {POOL_DIRNAME[p]}/ instead.")
                    _FALLBACK_NOTED.add((pool, family))
                return base
    # nothing found -> hand back the preferred nested path so errors name it
    return project_root / POOL_DIRNAME[pref] / fam / TENSOR_SUBDIR


def load_m2s_refusals(p):
    return {int(r["prompt_id"]) for r in json.load(open(p))}


def load_multi_refusals(p):
    out = {}
    for r in json.load(open(p)):
        c, t = int(r["conversation_id"]), int(r["turn_id"])
        out[c] = min(t, out.get(c, 99))
    return out


def _vec(path, cfg):
    return payload_to_vector(torch.load(path, map_location="cpu", weights_only=False), cfg)


def _turn_label(mode, is_attack, complied, turn_idx, n_turns, refusal_turn):
    last = turn_idx == n_turns
    if not is_attack:
        return 0.0, True, False
    if mode == "T1":
        return 1.0, True, complied
    if mode == "T2":
        return (1.0 if complied else 0.0), last, False
    if mode == "T3":
        if refusal_turn is None:
            return 0.0, True, False
        return (1.0 if turn_idx >= refusal_turn else 0.0), True, False
    raise ValueError(mode)


def build_m2s(vdir, refused, name, mode, cfg, hw):
    ex = []
    for pt in sorted(Path(vdir).glob("prompt_*.pt")):
        S = int(re.search(r"prompt_(\d+)\.pt", pt.name).group(1))
        v, ok, _ = _vec(pt, cfg)
        if not ok:
            continue
        complied = S not in refused
        y, inl, hard = _turn_label(mode, True, complied, 1, 1, None)
        ex.append(dict(group=S, source=f"m2s_{name}", seq=[v], y=[y],
                       loss=[inl], w=[hw if hard else 1.0]))
    return ex


def build_multi(mdir, rturns, mode, cfg, hw, n_turns=3):
    ex = []
    convs = sorted({int(re.search(r"conv_(\d+)_", p.name).group(1))
                    for p in Path(mdir).glob("conv_*_turn_*.pt")})
    for S in convs:
        rt = rturns.get(S, None); complied = rt is None
        seq = []
        for t in range(1, n_turns + 1):
            pt = Path(mdir) / f"conv_{S:04d}_turn_{t:02d}.pt"
            if not pt.exists():
                break
            v, ok, _ = _vec(pt, cfg)
            if not ok:
                break
            seq.append(v)
        if not seq:
            continue
        ys, ls, ws = [], [], []
        for t in range(1, len(seq) + 1):
            y, inl, hard = _turn_label(mode, True, complied, t, len(seq), rt)
            ys.append(y); ls.append(inl); ws.append(hw if hard else 1.0)
        ex.append(dict(group=S, source="multi_turn", seq=seq, y=ys, loss=ls, w=ws))
    return ex


def build_benign(bdir, cfg):
    ex = []; d = Path(bdir)
    for pt in sorted(d.glob("prompt_*.pt")):
        S = int(re.search(r"prompt_(\d+)\.pt", pt.name).group(1))
        v, ok, _ = _vec(pt, cfg)
        if ok:
            ex.append(dict(group=BENIGN_GROUP_OFFSET + S, source="benign_st",
                           seq=[v], y=[0.0], loss=[True], w=[1.0]))
    convs = sorted({int(re.search(r"conv_(\d+)_", p.name).group(1))
                    for p in d.glob("conv_*_turn_*.pt")})
    for S in convs:
        seq = []
        for t in range(1, 6):
            pt = d / f"conv_{S:04d}_turn_{t:02d}.pt"
            if not pt.exists():
                break
            v, ok, _ = _vec(pt, cfg)
            if not ok:
                break
            seq.append(v)
        if seq:
            n = len(seq)
            ex.append(dict(group=BENIGN_GROUP_OFFSET + 10000 + S, source="benign_mt",
                           seq=seq, y=[0.0]*n, loss=[True]*n, w=[1.0]*n))
    return ex


def pad_and_pack(ex, max_len=None):
    dim = ex[0]["seq"][0].numel()
    max_len = max_len or max(len(e["seq"]) for e in ex)
    N = len(ex)
    X = torch.zeros(N, max_len, dim); Y = torch.zeros(N, max_len)
    LOSS = torch.zeros(N, max_len, dtype=torch.bool)
    W = torch.ones(N, max_len); PAD = torch.zeros(N, max_len, dtype=torch.bool)
    G = torch.zeros(N, dtype=torch.long); SRC = []
    for i, e in enumerate(ex):
        n = len(e["seq"])
        for t in range(n):
            X[i, t] = e["seq"][t]
        Y[i, :n] = torch.tensor(e["y"]); LOSS[i, :n] = torch.tensor(e["loss"])
        W[i, :n] = torch.tensor(e["w"]); PAD[i, :n] = True; G[i] = e["group"]
        SRC.append(e["source"])
    return dict(X=X, Y=Y, loss_mask=LOSS, weight=W, turn_mask=PAD, group=G,
                source=SRC, dim=dim, max_len=max_len)


def assemble(project_root, refusal_dir, mode="T2", cfg=None,
             families=("pythonize", "hyphenize", "numberize", "multiturn"),
             benign_dir=None, hard_weight=3.0):
    cfg = cfg or FeatureConfig()
    project_root, rdir = Path(project_root), Path(refusal_dir)
    pool = cfg.local_pool
    ex = []
    for v in M2S_VARIANTS:
        if v not in families:
            continue
        vdir = resolve_tensor_dir(project_root, pool, v)
        if any(vdir.glob("prompt_*.pt")):
            ex += build_m2s(vdir, load_m2s_refusals(rdir / f"m2s_{v}_refusals.json"),
                            v, mode, cfg, hard_weight)
    if "multiturn" in families:
        mdir = resolve_tensor_dir(project_root, pool, "multiturn")
        if any(mdir.glob("conv_*_turn_*.pt")):
            ex += build_multi(mdir,
                              load_multi_refusals(rdir / "multi_turn_refusals.json"),
                              mode, cfg, hard_weight)
    if mode == "T1":
        bdir = Path(benign_dir) if benign_dir else resolve_tensor_dir(project_root, pool, "benign")
        if not bdir.exists() or not any(bdir.iterdir()):
            raise SystemExit("T1 needs benign routing tensors (pass --benign_dir or put "
                             f"them under {POOL_DIRNAME.get(pool, 'mean_pool_tensors')}/benign/).")
        ex += build_benign(bdir, cfg)
    if not ex:
        want = POOL_DIRNAME.get(pool, "mean_pool_tensors")
        raise SystemExit(
            f"No examples found under {project_root}/{want}/ for families={families}. "
            "If you asked for top-K, run the top-K extraction first — topk_pool_tensors "
            "is empty until you re-extract with the patched zaya_*.py.")
    packed = pad_and_pack(ex)
    packed["mode"] = mode; packed["feature_config"] = cfg.__dict__
    return packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="PROJECT root that contains mean_pool_tensors/, "
                         "topk_pool_tensors/, refusals/ (e.g. 'E:/P3 Defense').")
    ap.add_argument("--refusal_dir", required=True)
    ap.add_argument("--benign_dir", default=None,
                    help="T1 only. Defaults to <root>/<pool>_pool_tensors/benign/.")
    ap.add_argument("--mode", default="T2", choices=["T1", "T2", "T3"])
    ap.add_argument("--families", nargs="+",
                    default=["pythonize", "multiturn"])
    ap.add_argument("--local_pool", default="mean", choices=["mean", "topk"])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--use_logits", action="store_true")
    ap.add_argument("--hard_weight", type=float, default=3.0)
    ap.add_argument("--out", default="dataset.pt")
    a = ap.parse_args()
    cfg = FeatureConfig(local_pool=a.local_pool, K=a.K, use_logits=a.use_logits)
    packed = assemble(a.root, a.refusal_dir, a.mode, cfg,
                      tuple(a.families), a.benign_dir, a.hard_weight)
    torch.save(packed, a.out)
    pos = (packed["Y"][packed["loss_mask"]] == 1).sum().item()
    neg = (packed["Y"][packed["loss_mask"]] == 0).sum().item()
    from collections import Counter
    print(f"[build] mode={a.mode} families={a.families} pool={a.local_pool} K={a.K} "
          f"dim={packed['dim']} pos/neg={pos}/{neg} sources={dict(Counter(packed['source']))}")
    print(f"[build] saved -> {a.out}")


if __name__ == "__main__":
    main()
