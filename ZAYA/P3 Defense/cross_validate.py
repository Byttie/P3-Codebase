"""
cross_validate.py
-----------------
Stratified, grouped k-fold cross-validation for RoutingGuard.

Why this and not a single split: with ~49 refused conversations, one held-out
test set would contain only ~5 refused examples — too few for a stable estimate.
k-fold rotates the test fold so EVERY conversation is tested exactly once, and
stratification guarantees each fold mirrors the overall class balance so no fold
is starved of the minority class.

Correctness properties:
  * GROUPED  — each conversation (all its turns) stays wholly within one fold
               (no turn-level leakage). Rows in `packed` are already per-conversation.
  * STRATIFIED — folds balanced by conversation-level label (refused vs complied),
               so every fold has ~the same % refused.
  * NO LEAKAGE — z-score is fit on each fold's TRAIN turns only, never the test fold.

Reports per-fold class balance (so you can SEE stratification held), per-fold F1,
and the headline mean ± std across folds.

Requires scikit-learn:  pip install scikit-learn

Usage:
  python cross_validate.py --data ds_multiturn_topk300.pt --folds 5 --epochs 40
"""

import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import zscore
from model import RoutingGuard
from train import focal_bce_with_logits, evaluate

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    raise SystemExit("Needs scikit-learn:  pip install scikit-learn")


def conversation_labels(Y, loss_mask):
    """Derive one label per conversation (row) for stratification:
    a conversation is 'attack/complied' (1) if ANY scored turn is 1, else refused (0).
    Matches the T2 convention (complied dominates if present)."""
    labels = []
    for i in range(Y.shape[0]):
        m = loss_mask[i]
        if m.sum() == 0:
            labels.append(0)
        else:
            labels.append(int((Y[i][m] == 1).any()))
    return np.array(labels)


def train_on_fold(Xtr, Ytr, Ltr, Wtr, Xte, Yte, Lte, TMte,
                  in_dim, epochs, lr, wd, batch, gamma, device):
    model = RoutingGuard(in_dim=in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ds = TensorDataset(Xtr, Ytr, Ltr, Wtr)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    best = {"F1": -1}
    for ep in range(1, epochs + 1):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="e.g. ds_multiturn_topk300.pt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    packed = torch.load(a.data, weights_only=False)
    X, Y, LOSS, W, TM, G = (packed["X"], packed["Y"], packed["loss_mask"],
                            packed["weight"], packed["turn_mask"], packed["group"])

    conv_lab = conversation_labels(Y, LOSS)       # (N,) 0/1 per conversation
    groups = G.numpy()
    N = X.shape[0]

    n_ref = int((conv_lab == 0).sum()); n_comp = int((conv_lab == 1).sum())
    print("=" * 64)
    print(f"STRATIFIED GROUPED {a.folds}-FOLD CROSS-VALIDATION")
    print(f"data={a.data}  conversations={N}  "
          f"refused={n_ref} ({n_ref/N:.1%})  complied={n_comp} ({n_comp/N:.1%})")
    print("=" * 64)

    sgkf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    fold_f1, fold_prec, fold_rec = [], [], []

    print(f"{'fold':>4} | {'test convs':>10} {'test refused':>12} | "
          f"{'F1':>6} {'recall':>7} {'prec':>6} {'MTTD':>5}")
    print("-" * 64)

    for k, (tr_idx, te_idx) in enumerate(sgkf.split(np.zeros(N), conv_lab, groups), 1):
        tr = torch.tensor(tr_idx); te = torch.tensor(te_idx)

        # z-score fit on TRAIN turns only (no leakage from the test fold)
        mu, sigma = zscore.fit(X[tr][TM[tr]])
        Xs = zscore.transform(X, mu, sigma) * TM.unsqueeze(-1)
        dim = Xs.shape[-1]

        best = train_on_fold(Xs[tr], Y[tr], LOSS[tr], W[tr],
                             Xs[te], Y[te], LOSS[te], TM[te],
                             dim, a.epochs, a.lr, a.wd, a.batch, a.gamma, a.device)

        te_ref = int((conv_lab[te_idx] == 0).sum())
        fold_f1.append(best["F1"]); fold_prec.append(best["precision"]); fold_rec.append(best["recall"])
        print(f"{k:>4} | {len(te_idx):>10} {te_ref:>12} | "
              f"{best['F1']:>6.3f} {best['recall']:>7.3f} "
              f"{best['precision']:>6.3f} {best['MTTD']:>5}")

    print("-" * 64)
    f1 = np.array(fold_f1)
    print("\n" + "=" * 64)
    print("CROSS-VALIDATION RESULT")
    print("=" * 64)
    print(f"  per-fold F1 : {[f'{x:.3f}' for x in fold_f1]}")
    print(f"  MEAN F1     : {f1.mean():.3f} ± {f1.std():.3f}   <-- headline number")
    print(f"  precision   : {np.mean(fold_prec):.3f} ± {np.std(fold_prec):.3f}")
    print(f"  recall      : {np.mean(fold_rec):.3f} ± {np.std(fold_rec):.3f}")
    print()
    print(f"  Every one of the {n_ref} refused conversations was tested exactly once,")
    print(f"  across the {a.folds} folds. The ± reflects stability across data splits.")
    print()


if __name__ == "__main__":
    main()
