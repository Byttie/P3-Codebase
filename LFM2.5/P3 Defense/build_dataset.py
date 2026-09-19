"""
build_dataset.py  (v4 — Malicious/Benign layout, T1-compliance labeling)
------------------------------------------------------------------------
Assemble a T1 training set for harmful-compliance vs benign-compliance.

NEW on-disk layout (--root is the PROJECT root, e.g. "E:/P3 Defense"):

    <root>/Malicious_Tensors/multi-turn_mean/moe_routing_tensors/conv_*.pt
    <root>/Malicious_Tensors/multi-turn_topk/moe_routing_tensors/conv_*.pt
    <root>/Benign_Tensors/multi-turn_mean/moe_routing_tensors/conv_*.pt
    <root>/Benign_Tensors/multi-turn_topk/moe_routing_tensors/conv_*.pt
    <root>/refusals/multi_turn_refusals.json      (MALICIOUS refusals to DROP)

LABELS (T1-compliance, source-based):
    * malicious conversation NOT in the refusal JSON  -> label 1  (harmful compliance)
    * malicious conversation     in the refusal JSON  -> DROPPED  (regex handles refusals)
    * every benign conversation                       -> label 0  (benign compliance)

Pooling (--local_pool mean|topk, --K) selects the *_mean or *_topk subfolder.

Malicious and benign both use conv_XXXX ids starting at 0, so they would collide.
We keep them apart with a group offset: benign groups get +BENIGN_GROUP_OFFSET,
so grouped CV never mixes a malicious and benign conversation into one group.

exposes assemble(...) so the CV / experiment scripts can build in-process.
"""

import json, re, argparse
from pathlib import Path
import torch

from routing_features import FeatureConfig, payload_to_vector

BENIGN_GROUP_OFFSET = 1_000_000     # benign conv ids offset so they never collide with malicious

# ---------------------------------------------------------------------------
# On-disk layout.  Edit these if your folder names differ.
# ---------------------------------------------------------------------------
MALICIOUS_DIR = "Malicious_Tensors"
BENIGN_DIR = "Benign_Tensors"
POOL_SUBDIR = {"mean": "multi-turn_mean", "topk": "multi-turn_topk"}
TENSOR_SUBDIR = "moe_routing_tensors"
MALICIOUS_REFUSALS = "multi_turn_refusals.json"


def _tensor_dir(root, top_level, pool):
    """root/<Malicious|Benign>_Tensors/multi-turn_<pool>/moe_routing_tensors"""
    return Path(root) / top_level / POOL_SUBDIR[pool] / TENSOR_SUBDIR


def load_refused_ids(path):
    """Set of malicious conversation_ids to DROP (ZAYA refused these)."""
    if not Path(path).exists():
        print(f"[build] WARNING: refusal file not found: {path} (dropping nothing)")
        return set()
    return {int(r["conversation_id"]) for r in json.load(open(path, encoding="utf-8"))}


def _vec(path, cfg):
    return payload_to_vector(torch.load(path, map_location="cpu", weights_only=False), cfg)


def _conv_ids(tdir):
    return sorted({int(re.search(r"conv_(\d+)_", p.name).group(1))
                   for p in Path(tdir).glob("conv_*_turn_*.pt")})


def build_source(tdir, cfg, label, refused_ids=None, group_offset=0,
                 source_name="src", n_turns=3):
    """Build examples from one tensor folder.
    label: 0.0 or 1.0 for every conversation from this source.
    refused_ids: conversation_ids to SKIP (drop). None = keep all.
    group_offset: added to conv id so malicious/benign never share a group id.
    """
    refused_ids = refused_ids or set()
    ex = []
    for S in _conv_ids(tdir):
        if S in refused_ids:
            continue                                  # DROP refused conversations
        seq = []
        for t in range(1, n_turns + 1):
            pt = Path(tdir) / f"conv_{S:04d}_turn_{t:02d}.pt"
            if not pt.exists():
                break
            v, ok, _ = _vec(pt, cfg)
            if not ok:
                break
            seq.append(v)
        if not seq:
            continue
        n = len(seq)
        ex.append(dict(group=S + group_offset, source=source_name,
                       seq=seq, y=[label] * n,
                       loss=[True] * n, w=[1.0] * n))
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


def assemble(project_root, refusal_dir, mode="T1", cfg=None,
             families=None, benign_dir=None, hard_weight=1.0):
    """T1-compliance assembly. `mode`/`families`/`hard_weight` kept in the
    signature for compatibility with the CV/experiment scripts, but this v4
    always builds harmful-compliance (1) vs benign-compliance (0)."""
    cfg = cfg or FeatureConfig()
    root = Path(project_root)
    pool = cfg.local_pool

    mal_dir = _tensor_dir(root, MALICIOUS_DIR, pool)
    ben_dir = _tensor_dir(root, BENIGN_DIR, pool)
    refused = load_refused_ids(Path(refusal_dir) / MALICIOUS_REFUSALS)

    if not any(mal_dir.glob("conv_*_turn_*.pt")):
        raise SystemExit(f"No malicious tensors under {mal_dir}")
    if not any(ben_dir.glob("conv_*_turn_*.pt")):
        raise SystemExit(f"No benign tensors under {ben_dir}")

    ex = []
    # malicious: drop refusals, label 1
    ex += build_source(mal_dir, cfg, label=1.0, refused_ids=refused,
                       group_offset=0, source_name="malicious")
    n_mal = len(ex)
    # benign: keep all, label 0, offset groups so no collision
    ex += build_source(ben_dir, cfg, label=0.0, refused_ids=None,
                       group_offset=BENIGN_GROUP_OFFSET, source_name="benign")
    n_ben = len(ex) - n_mal

    if not ex:
        raise SystemExit("No examples assembled — check tensor folders and refusal file.")

    packed = pad_and_pack(ex)
    packed["mode"] = "T1"
    packed["feature_config"] = cfg.__dict__
    packed["n_malicious"] = n_mal
    packed["n_benign"] = n_ben
    return packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="PROJECT root containing Malicious_Tensors/, Benign_Tensors/, refusals/")
    ap.add_argument("--refusal_dir", required=True,
                    help="folder containing multi_turn_refusals.json (malicious refusals to drop)")
    ap.add_argument("--local_pool", default="mean", choices=["mean", "topk"])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--use_logits", action="store_true")
    ap.add_argument("--out", default="dataset_t1.pt")
    a = ap.parse_args()

    cfg = FeatureConfig(local_pool=a.local_pool, K=a.K, use_logits=a.use_logits)
    packed = assemble(a.root, a.refusal_dir, "T1", cfg)
    torch.save(packed, a.out)

    pos = int((packed["Y"][packed["loss_mask"]] == 1).sum())
    neg = int((packed["Y"][packed["loss_mask"]] == 0).sum())
    print(f"[build] T1-compliance  pool={a.local_pool} K={a.K} dim={packed['dim']}")
    print(f"[build] conversations: {packed['n_malicious']} harmful(1) + "
          f"{packed['n_benign']} benign(0) = {packed['n_malicious'] + packed['n_benign']}")
    print(f"[build] scored turns: pos(harmful)/neg(benign) = {pos}/{neg}")
    print(f"[build] saved -> {a.out}")


if __name__ == "__main__":
    main()