"""
ablation_shuffle_labels.py
--------------------------
The decisive "is the signal real?" test.

Trains RoutingGuard TWICE on the same feature tensors:
  (1) REAL labels        — your normal training
  (2) SHUFFLED labels    — Y randomly permuted, X untouched

If the detector is using genuine routing signal, (2) should COLLAPSE toward the
trivial baseline: with labels scrambled, there is nothing real to learn, so F1
drops to roughly the always-predict-majority score. If (2) stays HIGH, the model
is exploiting an artifact in X (sequence length, padding, group structure) rather
than the routing content — a red flag.

We shuffle at the CONVERSATION level (the group), not the turn level, so the
shuffle respects the grouped train/val split exactly the way real training does.
We average over several shuffle seeds because a single shuffle can get lucky.

Usage (from P3 Defense root):
  python ablation_shuffle_labels.py --data ds_multiturn.pt --epochs 40 --shuffles 5
"""

import argparse
import copy
import numpy as np
import torch

from train import run_training


def majority_f1(Y, loss_mask):
    """F1 of always-predicting the majority class over scored turns."""
    y = Y[loss_mask].cpu().numpy().astype(int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    maj = 1 if pos >= neg else 0
    pred = np.full_like(y, maj)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def shuffle_labels(packed, seed):
    """Return a copy of packed with Y permuted at the CONVERSATION (row) level.
    Each conversation keeps its own turn-label pattern, but that whole pattern is
    reassigned to a different conversation — breaking the feature<->label link
    while preserving label balance and turn structure."""
    p = copy.copy(packed)
    Y = packed["Y"].clone()
    n = Y.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    p["Y"] = Y[perm]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="e.g. ds_multiturn.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--shuffles", type=int, default=5, help="number of shuffle seeds to average")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    packed = torch.load(a.data, weights_only=False)
    trivial = majority_f1(packed["Y"], packed["loss_mask"])

    print("=" * 56)
    print("LABEL-SHUFFLE ABLATION")
    print("=" * 56)
    print(f"  data              : {a.data}")
    print(f"  trivial baseline  : F1 {trivial:.3f}  (always-majority)")
    print()

    # (1) real labels
    print("--- Training on REAL labels ---")
    real = run_training(packed, epochs=a.epochs, device=a.device, verbose=False)
    real_f1 = real["val_metrics"]["F1"]
    print(f"  REAL labels       : F1 {real_f1:.3f}  "
          f"(R {real['val_metrics']['recall']:.3f} "
          f"P {real['val_metrics']['precision']:.3f})")
    print()

    # (2) shuffled labels, averaged over seeds
    print(f"--- Training on SHUFFLED labels ({a.shuffles} seeds) ---")
    shuf_f1s = []
    for s in range(a.shuffles):
        p = shuffle_labels(packed, seed=1000 + s)
        res = run_training(p, epochs=a.epochs, seed=1234, device=a.device, verbose=False)
        f1 = res["val_metrics"]["F1"]
        shuf_f1s.append(f1)
        print(f"  shuffle seed {1000+s}: F1 {f1:.3f}")
    shuf_mean = float(np.mean(shuf_f1s))
    shuf_std = float(np.std(shuf_f1s))

    print()
    print("=" * 56)
    print("RESULT")
    print("=" * 56)
    print(f"  trivial baseline      : {trivial:.3f}")
    print(f"  REAL labels           : {real_f1:.3f}")
    print(f"  SHUFFLED labels (mean): {shuf_mean:.3f} +/- {shuf_std:.3f}")
    print()

    gain_real = real_f1 - trivial
    gain_shuf = shuf_mean - trivial
    print(f"  real gain over trivial    : {gain_real:+.3f}")
    print(f"  shuffled gain over trivial: {gain_shuf:+.3f}")
    print()
    print("=== Interpretation ===")
    if shuf_mean <= trivial + 0.03 and real_f1 >= shuf_mean + 0.05:
        print("  PASS. Shuffled labels collapse to the trivial baseline while real")
        print("  labels score well above it. The detector is learning GENUINE routing")
        print("  signal, not an artifact of the feature tensors.")
    elif shuf_mean >= real_f1 - 0.05:
        print("  FAIL / WARNING. Shuffled labels score nearly as high as real labels.")
        print("  The model is exploiting structure in X unrelated to the routing")
        print("  content (padding, sequence length, or group leakage). Investigate")
        print("  before trusting the real F1.")
    else:
        print("  PARTIAL. Shuffled sits between trivial and real. Some genuine signal,")
        print("  but the features may also carry a non-routing artifact. Report both")
        print("  numbers and the gap.")
    print()


if __name__ == "__main__":
    main()