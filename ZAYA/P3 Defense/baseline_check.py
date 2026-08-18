"""
baseline_check.py
-----------------
Computes the trivial baselines your RoutingGuard F1 must beat to mean anything:

  * ALWAYS-ATTACK   — predict every turn is an attack (complied=1). On an
                      imbalanced set this alone can score high F1.
  * ALWAYS-SAFE     — predict every turn is refused (0).
  * MAJORITY        — whichever of the two is more common in the labels.
  * RANDOM (p=base) — coin-flip weighted by the base rate, for reference.

It uses build_dataset.assemble() so the labels are IDENTICAL to what training
sees (same refusals JSONs, same loss_mask). Run it for the SAME families and
pool you trained on, then compare its F1 to your sweep_k.py numbers.

If your detector's F1 does not clearly exceed ALWAYS-ATTACK, the routing signal
is adding little over guessing the majority class.

Usage (from your P3 Defense root):
  python baseline_check.py --root . --refusal_dir refusals --families multiturn
  python baseline_check.py --root . --refusal_dir refusals --families pythonize
  python baseline_check.py --root . --refusal_dir refusals --families pythonize multiturn
"""

import argparse
import numpy as np
import torch

from build_dataset import assemble, FeatureConfig


def prf1(y_true, y_pred):
    """Binary P/R/F1 for the positive class (1 = attack/complied)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--refusal_dir", required=True)
    ap.add_argument("--families", nargs="+", default=["pythonize", "multiturn"])
    ap.add_argument("--mode", default="T2")
    ap.add_argument("--local_pool", default="mean", choices=["mean", "topk"])
    ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()

    cfg = FeatureConfig(local_pool=a.local_pool, K=a.K)
    packed = assemble(a.root, a.refusal_dir, a.mode, cfg, tuple(a.families))

    Y = packed["Y"]
    mask = packed["loss_mask"]
    y = Y[mask].cpu().numpy().astype(int)      # only scored positions

    n = len(y)
    pos = int((y == 1).sum())          # attacks / complied
    neg = int((y == 0).sum())          # refused / safe
    base_rate = pos / n if n else 0.0

    print(f"\n=== Label distribution (families={a.families}, pool={a.local_pool}) ===")
    print(f"  scored turns : {n}")
    print(f"  attack (1)   : {pos}  ({base_rate:.1%})")
    print(f"  refused (0)  : {neg}  ({neg/n:.1%})")

    print(f"\n=== Trivial baselines (F1 for the positive/attack class) ===")
    print(f"  {'baseline':<16} {'prec':>6} {'recall':>7} {'F1':>7}")
    print("  " + "-" * 40)

    # always attack
    p, r, f = prf1(y, np.ones_like(y))
    print(f"  {'ALWAYS-ATTACK':<16} {p:>6.3f} {r:>7.3f} {f:>7.3f}")

    # always safe
    p, r, f = prf1(y, np.zeros_like(y))
    print(f"  {'ALWAYS-SAFE':<16} {p:>6.3f} {r:>7.3f} {f:>7.3f}")

    # majority
    maj = 1 if pos >= neg else 0
    p, r, f = prf1(y, np.full_like(y, maj))
    print(f"  {'MAJORITY(=' + str(maj) + ')':<16} {p:>6.3f} {r:>7.3f} {f:>7.3f}")

    # random weighted by base rate (averaged over a few seeds for stability)
    fs = []
    for s in range(20):
        rng = np.random.default_rng(s)
        pred = (rng.random(n) < base_rate).astype(int)
        fs.append(prf1(y, pred)[2])
    print(f"  {'RANDOM(p=base)':<16} {'':>6} {'':>7} {np.mean(fs):>7.3f}")

    print(f"\n=== How to read this ===")
    print(f"  Your detector's F1 must clearly beat ALWAYS-ATTACK "
          f"({prf1(y, np.ones_like(y))[2]:.3f}) to be adding real signal.")
    print(f"  If your sweep F1 ~= that number, the model is mostly exploiting")
    print(f"  class imbalance, not the routing tensors.\n")


if __name__ == "__main__":
    main()