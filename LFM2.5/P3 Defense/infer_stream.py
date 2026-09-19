"""
infer_stream.py
---------------
Stateful, real-time scoring: feed one turn at a time, carry the GRU hidden state,
print a risk score per turn, and flag the turn that first crosses threshold.

Uses everything the checkpoint saved by train.py carries:
    state_dict, in_dim, scaler_mu, scaler_sigma, append_zscore, feature_config.

So the same local pooling (mean / top-K) and the same global Z-score scaler used
in training are applied here — no drift between train and serve.

    python infer_stream.py --ckpt guard.pt \
        --conv_glob 'E:/P3 Defense/topk_pool_tensors/multi-turn/moe_routing_tensors/conv_0007_turn_*.pt'
"""

import argparse, glob, re
import torch

import zscore
from routing_features import FeatureConfig, payload_to_vector
from model import RoutingGuard


def _turn_key(path):
    m = re.search(r"turn_(\d+)", path)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv_glob", required=True,
                    help="Glob for one conversation's turn tensors, e.g. "
                         "'.../multi-turn/moe_routing_tensors/conv_0007_turn_*.pt'.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FeatureConfig(**ck["feature_config"]) if ck.get("feature_config") else FeatureConfig()
    mu, sigma = ck["scaler_mu"], ck["scaler_sigma"]
    append_z = ck.get("append_zscore", False)

    model = RoutingGuard(in_dim=ck["in_dim"]).to(a.device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    turns = sorted(glob.glob(a.conv_glob), key=_turn_key)
    if not turns:
        raise SystemExit(f"No turn tensors match {a.conv_glob!r}")

    h = None
    flagged_at = None
    print(f"{'turn':>4} | {'risk':>6} | state")
    print("-" * 34)
    for i, path in enumerate(turns, 1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        v, ok, _ = payload_to_vector(payload, cfg)
        if not ok:
            print(f"{i:>4} | {'--':>6} | empty response, skipped")
            continue
        x = zscore.transform(v, mu, sigma)
        if append_z:
            s = zscore.anomaly_score(v, mu, sigma).reshape(1)
            x = torch.cat([x, s])
        prob, h = model.step(x.to(a.device), h)
        r = float(prob.item())
        hit = r >= a.threshold
        if hit and flagged_at is None:
            flagged_at = i
        print(f"{i:>4} | {r:>6.3f} | {'FLAG' if hit else ''}")

    if flagged_at:
        print(f"\n[flagged] first crossed {a.threshold} at turn {flagged_at} "
              f"(turns-to-detection = {flagged_at}).")
    else:
        print(f"\n[clean] never crossed {a.threshold}.")


if __name__ == "__main__":
    main()
