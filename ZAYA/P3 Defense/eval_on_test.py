"""
eval_on_test.py
---------------
The FINAL, one-time evaluation: score a trained RoutingGuard checkpoint on the
held-out test set (test.pt from make_holdout.py) that was NEVER used during CV,
tuning, or the sweeps.

This is the number that validates the deployed model on truly unseen data. Run
it ONCE, at the very end. Reports F1 / precision / recall + confusion matrix.

The checkpoint carries its own scaler (scaler_mu/sigma) and feature config, so
the test features are scaled EXACTLY as in training — no train/serve drift.

Usage:
  python eval_on_test.py --ckpt guard_final.pt --data test.pt
"""

import argparse
import numpy as np
import torch

import zscore
from model import RoutingGuard


@torch.no_grad()
def evaluate_full(model, X, Y, loss_mask, turn_mask, thr, device):
    model.eval()
    logits, _ = model(X.to(device))
    probs = torch.sigmoid(logits).cpu()
    m = loss_mask
    y = Y[m].numpy(); p = (probs[m].numpy() >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # MTTD: mean turn at which a harmful conversation first crosses threshold
    mttd, n = 0.0, 0
    for i in range(X.shape[0]):
        nt = int(turn_mask[i].sum())
        if nt <= 0 or not ((Y[i, :nt] == 1) & loss_mask[i, :nt]).any():
            continue
        cr = (probs[i, :nt] >= thr).nonzero().flatten()
        if cr.numel():
            mttd += float(cr[0].item() + 1); n += 1
    return dict(F1=f1, precision=prec, recall=rec, tp=tp, fp=fp, fn=fn, tn=tn,
                MTTD=(mttd / n if n else float("nan")))


def save_cm(mets, out="confusion_matrix_test.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[mets["tn"], mets["fp"]], [mets["fn"], mets["tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    labels = ["Benign (0)", "Harmful (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted", fontweight="bold"); ax.set_ylabel("Actual", fontweight="bold")
    tot = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n({v/tot*100:.1f}%)", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if v > cm.max() * 0.5 else "black")
    ax.set_title(f"HELD-OUT TEST Confusion Matrix\nF1={mets['F1']:.3f}  "
                 f"P={mets['precision']:.3f}  R={mets['recall']:.3f}",
                 fontweight="bold", pad=12)
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    print(f"  confusion matrix -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained checkpoint, e.g. guard_final.pt")
    ap.add_argument("--data", required=True, help="held-out test set, e.g. test.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    packed = torch.load(a.data, weights_only=False)
    X, Y, LOSS, TM = packed["X"], packed["Y"], packed["loss_mask"], packed["turn_mask"]

    # apply the SAME scaling the checkpoint was trained with
    append_z = ck.get("append_zscore", False)
    mu, sigma = ck["scaler_mu"], ck["scaler_sigma"]
    # detect whether the model was trained with z-score:
    # if the checkpoint's in_dim matches raw feature dim, it's z-off; if it matches
    # scaled dim it's the same shape either way — we replicate train.py's logic by
    # checking a flag we store. Fall back to applying z-score (the common case).
    use_zscore = ck.get("use_zscore", True)
    if use_zscore:
        Xs = zscore.transform(X, mu, sigma) * TM.unsqueeze(-1)
    else:
        Xs = X * TM.unsqueeze(-1)
    if append_z:
        score = zscore.anomaly_score(X, mu, sigma).unsqueeze(-1)
        Xs = torch.cat([Xs, score * TM.unsqueeze(-1)], dim=-1)

    model = RoutingGuard(in_dim=ck["in_dim"]).to(a.device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    # class balance of the test set (sanity)
    def conv_lab(Y, L):
        return np.array([int((Y[i][L[i]] == 1).any()) if L[i].sum() > 0 else 0
                         for i in range(Y.shape[0])])
    lab = conv_lab(Y, LOSS)
    n_h = int((lab == 1).sum()); n_b = int((lab == 0).sum())

    mets = evaluate_full(model, Xs, Y, LOSS, TM, a.threshold, a.device)

    print("=" * 56)
    print("HELD-OUT TEST EVALUATION  (one-time, unseen data)")
    print("=" * 56)
    print(f"  checkpoint : {a.ckpt}")
    print(f"  test set   : {a.data}  ({X.shape[0]} conversations: "
          f"{n_h} harmful, {n_b} benign)")
    print(f"  threshold  : {a.threshold}")
    print("-" * 56)
    print(f"  F1        : {mets['F1']:.3f}")
    print(f"  precision : {mets['precision']:.3f}")
    print(f"  recall    : {mets['recall']:.3f}")
    print(f"  MTTD      : {mets['MTTD']:.2f} turns")
    print("-" * 56)
    print(f"  confusion (turn-level):")
    print(f"    TP={mets['tp']}  FP={mets['fp']}  FN={mets['fn']}  TN={mets['tn']}")
    # benign false-positive rate = FP / (FP+TN)
    fpr = mets['fp'] / (mets['fp'] + mets['tn']) if (mets['fp'] + mets['tn']) else 0.0
    print(f"    benign false-positive rate: {fpr:.1%}")
    print("=" * 56)
    save_cm(mets)
    print()
    print("  This is the FINAL held-out number. Report it alongside the CV result:")
    print("  CV estimates expected performance; this confirms it on unseen data.")


if __name__ == "__main__":
    main()
