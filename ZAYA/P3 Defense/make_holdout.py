"""
make_holdout.py
---------------
Carve a stratified, grouped held-out TEST set off the full T1 dataset, ONCE.

  * dev.pt  (default 90%) -> all development: CV, sweeps, final model training
  * test.pt (default 10%) -> locked away; touched ONCE to evaluate the deployed
                             checkpoint on truly unseen conversations.

Split properties:
  * GROUPED   — a conversation (all its turns) goes wholly into dev OR test.
  * STRATIFIED — the harmful/benign ratio is preserved in both dev and test.
  * ONE-TIME  — fixed by --seed; the test set never rotates and is never used
                during CV, tuning, or the sweeps.

Requires scikit-learn.

Usage:
  python make_holdout.py --data dataset_full.pt --test_frac 0.10 \\
      --dev_out dev.pt --test_out test.pt --seed 1234
"""

import argparse
import numpy as np
import torch

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    raise SystemExit("Needs scikit-learn:  pip install scikit-learn")


def conv_labels(Y, loss_mask):
    out = []
    for i in range(Y.shape[0]):
        m = loss_mask[i]
        out.append(int((Y[i][m] == 1).any()) if m.sum() > 0 else 0)
    return np.array(out)


def subset(packed, idx):
    """Return a new packed dict containing only rows in idx (preserves all keys)."""
    idx = np.asarray(idx)
    out = dict(packed)  # copy scalars / metadata
    for key in ("X", "Y", "loss_mask", "weight", "turn_mask", "group"):
        out[key] = packed[key][torch.as_tensor(idx)]
    if "source" in packed:
        out["source"] = [packed["source"][i] for i in idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="full T1 dataset (from build_dataset.py)")
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--dev_out", default="dev.pt")
    ap.add_argument("--test_out", default="test.pt")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    packed = torch.load(a.data, weights_only=False)
    Y, LOSS, G = packed["Y"], packed["loss_mask"], packed["group"]
    labels = conv_labels(Y, LOSS)
    groups = G.numpy()
    N = Y.shape[0]

    # StratifiedGroupKFold with k = round(1/test_frac) folds; take ONE fold as test.
    k = max(2, round(1.0 / a.test_frac))     # e.g. 0.10 -> 10 folds -> one is ~10%
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=a.seed)
    dev_idx, test_idx = next(iter(sgkf.split(np.zeros(N), labels, groups)))

    dev = subset(packed, dev_idx)
    test = subset(packed, test_idx)

    torch.save(dev, a.dev_out)
    torch.save(test, a.test_out)

    # Also save the conversation GROUP ids for each split, so the K-sweep
    # experiment (which re-pools per K from tensors) can filter to dev-only
    # without leaking the held-out test conversations.
    import json
    dev_groups = sorted(int(g) for g in dev["group"].tolist())
    test_groups = sorted(int(g) for g in test["group"].tolist())
    json.dump({"dev_groups": dev_groups, "test_groups": test_groups,
               "seed": a.seed, "test_frac": a.test_frac},
              open("holdout_groups.json", "w"), indent=2)
    print("  saved group ids -> holdout_groups.json "
          "(used by run_full_experiment.py to CV on dev only)")

    def summarize(name, idx):
        lab = labels[idx]
        n = len(idx); h = int((lab == 1).sum()); b = int((lab == 0).sum())
        print(f"  {name:5s}: {n:4d} conversations  |  harmful(1)={h}  benign(0)={b}  "
              f"({h/n:.1%} harmful)")

    print("=" * 60)
    print(f"HELD-OUT SPLIT  (test_frac={a.test_frac}, k={k} folds, seed={a.seed})")
    print("=" * 60)
    print(f"  full : {N} conversations  ({int((labels==1).sum())} harmful, "
          f"{int((labels==0).sum())} benign)")
    summarize("dev", dev_idx)
    summarize("test", test_idx)
    print()
    print(f"  saved dev  -> {a.dev_out}   (use for CV, sweeps, final training)")
    print(f"  saved test -> {a.test_out}  (LOCK AWAY; evaluate once at the end)")
    print()
    print("  Verify: harmful% in dev and test should both match the full set")
    print("  (stratification held). Groups are whole conversations (no leakage).")


if __name__ == "__main__":
    main()
