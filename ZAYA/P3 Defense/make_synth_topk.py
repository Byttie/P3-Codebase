"""
make_synth_topk.py  (v2 — writes the real project layout)
---------------------------------------------------------
Synthetic tensors WITH a token axis, so top-K pooling is meaningful. The class
signal lives ONLY in a few spike tokens (like an adversarial trigger): mean
pooling dilutes it, top-K pooling recovers it. A correct pipeline shows val F1
rising with K up to ~n_spike, then flattening/falling -> your sweet spot.

It now writes the SAME folder layout your real data uses, so the smoke test
exercises the exact paths build_dataset.py / sweep_k.py resolve:

    <root>/mean_pool_tensors/m2s-pythonize/moe_routing_tensors/prompt_*.pt   (mean only)
    <root>/mean_pool_tensors/multi-turn/moe_routing_tensors/conv_*.pt        (mean only)
    <root>/topk_pool_tensors/m2s-pythonize/moe_routing_tensors/prompt_*.pt   (+ topk_token_probs)
    <root>/topk_pool_tensors/multi-turn/moe_routing_tensors/conv_*.pt        (+ topk_token_probs)
    <root>/refusals/m2s_pythonize_refusals.json
    <root>/refusals/multi_turn_refusals.json

The mean root deliberately omits 'topk_token_probs' (like your legacy files), so
'--local_pool mean' works there and '--local_pool topk' needs the top-K root.
Focus families: pythonize + multiturn (T2: complied vs refused).
"""

import json, argparse
from pathlib import Path
import torch

L, E, KMAX, T = 40, 17, 10, 200
NEXP = [17] * L


def payload(spike_experts, is_spike, n_spike, seed):
    g = torch.Generator().manual_seed(seed)
    probs = torch.rand(L, T, E, generator=g) * 0.10           # small base
    # label-INDEPENDENT per-prompt baseline on the signal experts -> pollutes MEAN
    for e in spike_experts:
        base = torch.rand(1, generator=g).item() * 0.40
        probs[:, :, e] += base
    # label signal: a few EXTREME spike tokens, only for the spiking class
    if is_spike and n_spike:
        idx = torch.randperm(T, generator=g)[:n_spike]
        for e in spike_experts:
            probs[:, idx[:, None], e] = 0.95
    probs = probs.clamp(0, 1)
    mean = probs.mean(1)                                       # (L,E)
    topk = probs.topk(KMAX, dim=1).values                     # (L,KMAX,E)
    h = {i: torch.bincount(probs[i].argmax(-1), minlength=E).tolist() for i in range(L)}
    mask = torch.ones(L, E, dtype=torch.bool)
    full = dict(pooled_probs=mean, pooled_logits=mean, topk_token_probs=topk,
                valid_mask=mask, layer_indices=list(range(L)),
                num_experts_per_layer=NEXP, response_token_count=T,
                top1_expert_histogram=h)
    return full


def _save(payload_full, mean_path, topk_path):
    """Write a mean-only copy to the mean root and a full (top-K) copy to the topk root."""
    mean_only = {k: v for k, v in payload_full.items() if k != "topk_token_probs"}
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    topk_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_only, mean_path)
    torch.save(payload_full, topk_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="synth_proj")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    root = Path(a.root)
    mean_root = root / "mean_pool_tensors"
    topk_root = root / "topk_pool_tensors"
    ref = root / "refusals"; ref.mkdir(parents=True, exist_ok=True)
    seed = 0

    def fam_dir(base, fam):
        return base / fam / "moe_routing_tensors"

    # pythonize (single-turn), ASR 85% -> 15% refused
    refused = []
    for S in range(a.n):
        is_ref = (S % 100) < 15
        p = payload([0, 1], is_spike=is_ref, n_spike=4, seed=seed); seed += 1
        _save(p,
              fam_dir(mean_root, "m2s-pythonize") / f"prompt_{S:04d}.pt",
              fam_dir(topk_root, "m2s-pythonize") / f"prompt_{S:04d}.pt")
        if is_ref:
            refused.append(dict(prompt_id=S, user_prompt="x", model_response="refuse"))
    json.dump(refused, open(ref / "m2s_pythonize_refusals.json", "w"))

    # multiturn, ASR 69% -> 31% refused (refusal at a random turn)
    mref = []
    for S in range(a.n):
        is_ref = (S % 100) < 31
        rturn = (S % 3) + 1 if is_ref else None
        for t in range(1, 4):
            strong = is_ref and t >= (rturn or 99)
            p = payload([0, 1], is_spike=strong, n_spike=4, seed=seed); seed += 1
            _save(p,
                  fam_dir(mean_root, "multi-turn") / f"conv_{S:04d}_turn_{t:02d}.pt",
                  fam_dir(topk_root, "multi-turn") / f"conv_{S:04d}_turn_{t:02d}.pt")
        if is_ref:
            mref.append(dict(conversation_id=S, turn_id=rturn,
                             user_prompt="x", model_response="refuse"))
    json.dump(mref, open(ref / "multi_turn_refusals.json", "w"))
    print(f"[synth] wrote {root}/ (n={a.n}) — mean_pool_tensors/ + topk_pool_tensors/ + refusals/")


if __name__ == "__main__":
    main()
