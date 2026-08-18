"""
train.py  (v2)
--------------
Train RoutingGuard. Z-score global aggregation is the input scaler (zscore.py),
fit on TRAIN turns only. Optional --append_zscore adds mean(|Z|) as one extra
feature (your "mean of the Z matrix" anomaly signal) alongside the learned ones.
Exposes run_training(packed, ...) so sweep_k.py can call it in-process.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import zscore
from model import RoutingGuard, focal_bce_with_logits


def grouped_split(groups, val_frac=0.2, seed=1234):
    g = groups.numpy(); uniq = np.unique(g)
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    vg = set(uniq[:n_val].tolist())
    va = np.array([i for i, x in enumerate(g) if x in vg])
    tr = np.array([i for i, x in enumerate(g) if x not in vg])
    return torch.tensor(tr), torch.tensor(va)


@torch.no_grad()
def evaluate(model, X, Y, loss_mask, turn_mask, thr=0.5, device="cpu"):
    model.eval()
    logits, _ = model(X.to(device))
    probs = torch.sigmoid(logits).cpu()
    m = loss_mask
    y = Y[m].numpy(); p = (probs[m].numpy() >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mttd, n = 0.0, 0
    for i in range(X.shape[0]):
        nt = int(turn_mask[i].sum())
        if nt <= 1:
            continue
        if not ((Y[i, :nt] == 1) & loss_mask[i, :nt]).any():
            continue
        cr = (probs[i, :nt] >= thr).nonzero().flatten()
        if cr.numel():
            mttd += float(cr[0].item() + 1); n += 1
    return dict(F1=round(f1, 3), precision=round(prec, 3), recall=round(rec, 3),
                tp=tp, fp=fp, fn=fn, tn=tn, MTTD=(round(mttd / n, 2) if n else float("nan")))


def save_confusion_matrix(metrics, out_path="confusion_matrix_val.png"):
    """Draw a 2x2 confusion matrix from the best-epoch validation counts.
    Rows = actual, Cols = predicted. Positive class = 1 = attack/complied."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics.get("tn", 0)
    # layout: [[TN, FP], [FN, TP]] with actual on rows (0=refused/safe,1=attack)
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Refused / Safe (0)", "Complied / Attack (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9); ax.set_yticklabels(labels, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Predicted", fontweight="bold"); ax.set_ylabel("Actual", fontweight="bold")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            pct = (v / total * 100) if total else 0
            ax.text(j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if v > cm.max() * 0.5 else "black")
    f1 = metrics.get("F1", 0); prec = metrics.get("precision", 0); rec = metrics.get("recall", 0)
    ax.set_title(f"Validation Confusion Matrix\nF1={f1}  Precision={prec}  Recall={rec}",
                 fontweight="bold", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150); plt.close()
    print(f"  [confusion matrix] TN={tn} FP={fp} FN={fn} TP={tp} -> {out_path}")


def run_training(packed, epochs=40, lr=1e-3, wd=1e-4, batch=32, gamma=2.0,
                 val_frac=0.2, seed=1234, append_zscore=False,
                 device="cuda" if torch.cuda.is_available() else "cpu", verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y, LOSS, W, TM, G = (packed["X"], packed["Y"], packed["loss_mask"],
                            packed["weight"], packed["turn_mask"], packed["group"])
    tr, va = grouped_split(G, val_frac, seed)

    # GLOBAL Z-SCORE: fit on train turns only
    mu, sigma = zscore.fit(X[tr][TM[tr]])
    Xs = zscore.transform(X, mu, sigma) * TM.unsqueeze(-1)

    if append_zscore:
        score = zscore.anomaly_score(X, mu, sigma).unsqueeze(-1)   # (N,T,1)
        Xs = torch.cat([Xs, score * TM.unsqueeze(-1)], dim=-1)
    dim = Xs.shape[-1]

    ds = TensorDataset(Xs[tr], Y[tr], LOSS[tr], W[tr])
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = RoutingGuard(in_dim=dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_f1, best_state, best_metrics = -1.0, None, None
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
        val = evaluate(model, Xs[va], Y[va], LOSS[va], TM[va], device=device)
        if val["F1"] >= best_f1:
            best_f1 = val["F1"]; best_metrics = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  ep {ep:3d}  val F1 {val['F1']:.3f} R {val['recall']:.3f} "
                  f"P {val['precision']:.3f} MTTD {val['MTTD']}")
    if best_metrics is not None:
        save_confusion_matrix(best_metrics)
    return dict(state_dict=best_state, in_dim=dim, scaler_mu=mu, scaler_sigma=sigma,
                append_zscore=append_zscore, mode=packed.get("mode"),
                feature_config=packed.get("feature_config"), val_metrics=best_metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--append_zscore", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="routing_guard.pt")
    a = ap.parse_args()
    packed = torch.load(a.data, weights_only=False)
    res = run_training(packed, epochs=a.epochs, append_zscore=a.append_zscore, device=a.device)
    print("[best val]", res["val_metrics"])
    torch.save(res, a.out); print(f"[train] saved -> {a.out}")


if __name__ == "__main__":
    main()