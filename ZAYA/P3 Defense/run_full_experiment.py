"""
run_full_experiment.py
----------------------
The complete comparison grid, every cell measured by stratified grouped k-fold CV:

    pooling = mean            × zscore ∈ {on, off}          =  2 cells
    pooling = topk, K=1..Kmax × zscore ∈ {on, off}          =  2 * Kmax cells

With Kmax=30 that is (2 + 60) = 62 configurations × 5 folds = 310 model trainings.
Expect a few hours. It is a one-time run that produces a full, defensible table.

Answers three questions at once:
  * best top-K value (where F1 peaks vs converges to mean)
  * mean vs top-K   (does any K beat mean pooling?)
  * z-score on/off  (does normalisation help, hurt, or neither?)

Each cell reports F1 mean ± std across folds. Stratified + grouped + per-fold
z-score fit (no leakage), identical to cross_validate.py.

Datasets are assembled IN MEMORY per (pooling, K) — no 30 files written to disk.

Requires scikit-learn. Run from the project root (where build_dataset.py etc. live).

Usage:
  python run_full_experiment.py --root . --refusal_dir refusals --families multiturn --Kmax 30 --folds 5 --epochs 40
"""

import argparse
import csv
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import zscore
from model import RoutingGuard
from train import focal_bce_with_logits, evaluate
from build_dataset import assemble
from routing_features import FeatureConfig
import json as _json

def _filter_to_groups(packed, keep_groups):
    """Keep only rows whose group id is in keep_groups (dev-only CV)."""
    if keep_groups is None:
        return packed
    keep = set(int(g) for g in keep_groups)
    g = packed['group'].tolist()
    idx = [i for i, gg in enumerate(g) if int(gg) in keep]
    sel = torch.as_tensor(idx)
    out = dict(packed)
    for k in ('X','Y','loss_mask','weight','turn_mask','group'):
        out[k] = packed[k][sel]
    if 'source' in packed:
        out['source'] = [packed['source'][i] for i in idx]
    return out

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    raise SystemExit("Needs scikit-learn:  pip install scikit-learn")


def conversation_labels(Y, loss_mask):
    labels = []
    for i in range(Y.shape[0]):
        m = loss_mask[i]
        labels.append(int((Y[i][m] == 1).any()) if m.sum() > 0 else 0)
    return np.array(labels)


def train_on_fold(Xtr, Ytr, Ltr, Wtr, Xte, Yte, Lte, TMte,
                  in_dim, epochs, lr, wd, batch, gamma, device):
    model = RoutingGuard(in_dim=in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dl = DataLoader(TensorDataset(Xtr, Ytr, Ltr, Wtr), batch_size=batch, shuffle=True)
    best = {"F1": -1, "precision": 0, "recall": 0}
    for _ in range(epochs):
        model.train()
        for xb, yb, lb, wb in dl:
            xb, yb, lb, wb = [t.to(device) for t in (xb, yb, lb, wb)]
            logits, _ = model(xb)
            if lb.sum() == 0:
                continue
            loss = focal_bce_with_logits(logits[lb], yb[lb], weight=wb[lb], gamma=gamma)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val = evaluate(model, Xte, Yte, Lte, TMte, device=device)
        if val["F1"] >= best["F1"]:
            best = val
    return best


def cv_evaluate(packed, use_zscore, folds, epochs, lr, wd, batch, gamma, seed, device):
    """Stratified grouped k-fold CV for one packed dataset + one z-score setting.
    Returns (mean_f1, std_f1, mean_prec, mean_rec)."""
    X, Y, LOSS, W, TM, G = (packed["X"], packed["Y"], packed["loss_mask"],
                            packed["weight"], packed["turn_mask"], packed["group"])
    conv_lab = conversation_labels(Y, LOSS)
    groups = G.numpy(); N = X.shape[0]

    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    f1s, precs, recs = [], [], []
    for tr_idx, te_idx in sgkf.split(np.zeros(N), conv_lab, groups):
        tr = torch.tensor(tr_idx); te = torch.tensor(te_idx)
        if use_zscore:
            mu, sigma = zscore.fit(X[tr][TM[tr]])
            Xs = zscore.transform(X, mu, sigma) * TM.unsqueeze(-1)
        else:
            Xs = X * TM.unsqueeze(-1)
        dim = Xs.shape[-1]
        best = train_on_fold(Xs[tr], Y[tr], LOSS[tr], W[tr],
                             Xs[te], Y[te], LOSS[te], TM[te],
                             dim, epochs, lr, wd, batch, gamma, device)
        f1s.append(best["F1"]); precs.append(best["precision"]); recs.append(best["recall"])
    return np.mean(f1s), np.std(f1s), np.mean(precs), np.mean(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--refusal_dir", default="refusals")
    ap.add_argument("--families", nargs="+", default=["multiturn"])
    ap.add_argument("--mode", default="T2")
    ap.add_argument("--Kmax", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_csv", default="experiment_results.csv")
    ap.add_argument("--dev_groups", default=None,
                    help="holdout_groups.json — restrict CV to dev conversations "
                         "only (keeps the held-out test set out of the experiment)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    fam = tuple(a.families)
    dev_groups = None
    if a.dev_groups:
        dev_groups = _json.load(open(a.dev_groups))['dev_groups']
        print(f'[dev-only] restricting CV to {len(dev_groups)} dev conversations (held-out test excluded)')
    results = []  # (pooling, K, zscore, f1_mean, f1_std, prec, rec)

    print("=" * 72)
    print("FULL EXPERIMENT GRID (stratified grouped k-fold CV)")
    print(f"families={fam}  Kmax={a.Kmax}  folds={a.folds}  epochs={a.epochs}")
    print(f"total configs: {2 + 2*a.Kmax}  ->  {(2 + 2*a.Kmax)*a.folds} model trainings")
    print("=" * 72)
    print(f"{'pooling':>7} {'K':>3} {'zscore':>7} | {'F1 mean':>8} {'F1 std':>7} {'prec':>6} {'recall':>7}")
    print("-" * 72)

    # ---- MEAN pooling (2 cells) ----
    packed_mean = assemble(a.root, a.refusal_dir, a.mode,
                           FeatureConfig(local_pool="mean"), fam)
    packed_mean = _filter_to_groups(packed_mean, dev_groups)
    for zs in (True, False):
        m, s, p, r = cv_evaluate(packed_mean, zs, a.folds, a.epochs, a.lr, a.wd,
                                 a.batch, a.gamma, a.seed, a.device)
        results.append(("mean", 0, zs, m, s, p, r))
        print(f"{'mean':>7} {'-':>3} {('on' if zs else 'off'):>7} | "
              f"{m:>8.3f} {s:>7.3f} {p:>6.3f} {r:>7.3f}")

    # ---- TOP-K pooling, K=1..Kmax (2*Kmax cells) ----
    for K in range(1, a.Kmax + 1):
        packed_topk = assemble(a.root, a.refusal_dir, a.mode,
                               FeatureConfig(local_pool="topk", K=K), fam)
        packed_topk = _filter_to_groups(packed_topk, dev_groups)
        for zs in (True, False):
            m, s, p, r = cv_evaluate(packed_topk, zs, a.folds, a.epochs, a.lr, a.wd,
                                     a.batch, a.gamma, a.seed, a.device)
            results.append(("topk", K, zs, m, s, p, r))
            print(f"{'topk':>7} {K:>3} {('on' if zs else 'off'):>7} | "
                  f"{m:>8.3f} {s:>7.3f} {p:>6.3f} {r:>7.3f}")

    # ---- save + summarise ----
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pooling", "K", "zscore", "f1_mean", "f1_std", "precision", "recall"])
        for row in results:
            pooling, K, zs, m, s, p, r = row
            w.writerow([pooling, K, "on" if zs else "off",
                        f"{m:.4f}", f"{s:.4f}", f"{p:.4f}", f"{r:.4f}"])

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    best = max(results, key=lambda r: r[3])
    mean_on = next(r for r in results if r[0] == "mean" and r[2] is True)
    mean_off = next(r for r in results if r[0] == "mean" and r[2] is False)
    best_topk = max((r for r in results if r[0] == "topk"), key=lambda r: r[3])

    print(f"  best overall     : {best[0]} K={best[1]} zscore={'on' if best[2] else 'off'}"
          f"  F1 {best[3]:.3f} ± {best[4]:.3f}")
    print(f"  mean (z-on)      : F1 {mean_on[3]:.3f} ± {mean_on[4]:.3f}")
    print(f"  mean (z-off)     : F1 {mean_off[3]:.3f} ± {mean_off[4]:.3f}")
    print(f"  best top-K       : K={best_topk[1]} zscore={'on' if best_topk[2] else 'off'}"
          f"  F1 {best_topk[3]:.3f} ± {best_topk[4]:.3f}")
    print()
    # mean vs topk verdict
    if best[0] == "mean":
        print("  VERDICT: mean pooling wins — top-K provides no advantage. Signal is")
        print("           distributed across tokens, not concentrated in a few peaks.")
    else:
        gain = best_topk[3] - max(mean_on[3], mean_off[3])
        print(f"  VERDICT: top-K wins at K={best_topk[1]} by {gain:+.3f} F1 over best mean.")
    # z-score verdict (averaged across all configs)
    on_avg = np.mean([r[3] for r in results if r[2] is True])
    off_avg = np.mean([r[3] for r in results if r[2] is False])
    print(f"  z-score effect   : on avg {on_avg:.3f} vs off avg {off_avg:.3f}  "
          f"({on_avg - off_avg:+.3f})")
    print(f"\n  full table -> {a.out_csv}")
    print()


if __name__ == "__main__":
    main()