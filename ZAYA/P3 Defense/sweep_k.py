"""
sweep_k.py
----------
The "find the sweet spot for K" ablation, as a train-time sweep (no model
re-runs — the top-K token values were saved once). For each K it rebuilds
features with top-K local pooling, trains RoutingGuard, and reports val F1.

The 'mean' row is the baseline your mean_pool_tensors/ can already produce; the
top1..topK rows read topk_pool_tensors/. If you haven't run the top-K extraction
yet those rows are skipped with an 'n/a' note instead of crashing, so you still
get the mean baseline.

--root is the PROJECT root (contains mean_pool_tensors/, topk_pool_tensors/,
refusals/) — same as build_dataset.py.
"""

import argparse
import torch
from routing_features import FeatureConfig
from build_dataset import assemble
from train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="PROJECT root (see build_dataset.py).")
    ap.add_argument("--refusal_dir", required=True)
    ap.add_argument("--mode", default="T2")
    ap.add_argument("--families", nargs="+", default=["pythonize", "multiturn"])
    ap.add_argument("--Kmax", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--benign_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    print(f"{'pool':>8} | {'F1':>5} {'recall':>6} {'prec':>5} {'MTTD':>5}")
    print("-" * 40)
    rows = ([("mean", FeatureConfig(local_pool="mean"))] +
            [(f"top{K}", FeatureConfig(local_pool="topk", K=K))
             for K in range(1, a.Kmax + 1)])
    for label, cfg in rows:
        try:
            packed = assemble(a.root, a.refusal_dir, a.mode, cfg,
                              tuple(a.families), a.benign_dir)
            res = run_training(packed, epochs=a.epochs, device=a.device, verbose=False)
            m = res["val_metrics"]
            print(f"{label:>8} | {m['F1']:>5} {m['recall']:>6} {m['precision']:>5} {str(m['MTTD']):>5}")
        except SystemExit as e:
            print(f"{label:>8} | n/a   ({e})")


if __name__ == "__main__":
    main()
