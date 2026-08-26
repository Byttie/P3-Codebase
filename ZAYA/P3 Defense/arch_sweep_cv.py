"""
arch_sweep_cv.py
----------------
Experiment: does a bigger GRU (more layers / wider hidden, DeepContext-style)
improve detection, or is one small layer enough?

Sweeps GRU architecture:  num_layers ∈ {1,2,3}  ×  gru_hidden ∈ {128,512,2048}
= 9 configs, each evaluated by stratified grouped k-fold CV (consistent with the
main experiment). Reports F1 mean ± std and parameter count per config.

Runs on ONE pooling config — the winner from the main experiment. Pass it via
--local_pool / --K / --zscore so the architecture is tuned on the config you will
actually report.

Requires scikit-learn. Run from the project root.

Usage (example: winning config was top-K K=4, z-score on):
  python arch_sweep_cv.py --root . --refusal_dir refusals --families multiturn \\
      --local_pool topk --K 4 --zscore on --folds 5 --epochs 40

Or if mean pooling won:
  python arch_sweep_cv.py --root . --refusal_dir refusals --families multiturn \\
      --local_pool mean --zscore on --folds 5 --epochs 40
"""

import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import zscore
from model import RoutingGuard
from train import focal_bce_with_logits, evaluate
from build_dataset import assemble
from routing_features import FeatureConfig

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    raise SystemExit("Needs scikit-learn:  pip install scikit-learn")

LAYERS = [1, 2, 3]
HIDDENS = [128, 512, 2048]


def conversation_labels(Y, loss_mask):
    out = []
    for i in range(Y.shape[0]):
        m = loss_mask[i]
        out.append(int((Y[i][m] == 1).any()) if m.sum() > 0 else 0)
    return np.array(out)


def cv_one_arch(packed, gru_hidden, num_layers, use_zscore,
                folds, epochs, lr, wd, batch, gamma, seed, device):
    X, Y, LOSS, W, TM, G = (packed["X"], packed["Y"], packed["loss_mask"],
                            packed["weight"], packed["turn_mask"], packed["group"])
    conv_lab = conversation_labels(Y, LOSS); groups = G.numpy(); N = X.shape[0]
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    f1s = []
    nparams = None
    for tr_idx, te_idx in sgkf.split(np.zeros(N), conv_lab, groups):
        tr = torch.tensor(tr_idx); te = torch.tensor(te_idx)
        if use_zscore:
            mu, sigma = zscore.fit(X[tr][TM[tr]])
            Xs = zscore.transform(X, mu, sigma) * TM.unsqueeze(-1)
        else:
            Xs = X * TM.unsqueeze(-1)
        dim = Xs.shape[-1]

        model = RoutingGuard(in_dim=dim, gru_hidden=gru_hidden,
                             num_layers=num_layers).to(device)
        if nparams is None:
            nparams = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        dl = DataLoader(TensorDataset(Xs[tr], Y[tr], LOSS[tr], W[tr]),
                        batch_size=batch, shuffle=True)
        best = -1
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
            v = evaluate(model, Xs[te], Y[te], LOSS[te], TM[te], device=device)
            if v["F1"] > best:
                best = v["F1"]
        f1s.append(best)
    return np.mean(f1s), np.std(f1s), nparams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--refusal_dir", default="refusals")
    ap.add_argument("--families", nargs="+", default=["multiturn"])
    ap.add_argument("--mode", default="T2")
    ap.add_argument("--local_pool", default="mean", choices=["mean", "topk"])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--zscore", default="on", choices=["on", "off"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    use_z = (a.zscore == "on")
    cfg = FeatureConfig(local_pool=a.local_pool,
                        **({"K": a.K} if a.local_pool == "topk" else {}))
    packed = assemble(a.root, a.refusal_dir, a.mode, cfg, tuple(a.families))

    pool_desc = f"{a.local_pool}" + (f" K={a.K}" if a.local_pool == "topk" else "")
    print("=" * 66)
    print("ARCHITECTURE SWEEP under stratified k-fold CV")
    print(f"pooling={pool_desc}  zscore={a.zscore}  folds={a.folds}  epochs={a.epochs}")
    print(f"grid: {LAYERS} layers x {HIDDENS} hidden = {len(LAYERS)*len(HIDDENS)} configs")
    print("=" * 66)
    print(f"{'layers':>6} {'hidden':>7} {'params':>11} | {'F1 mean':>8} {'F1 std':>7}")
    print("-" * 66)

    results = []
    for nl in LAYERS:
        for h in HIDDENS:
            m, s, npar = cv_one_arch(packed, h, nl, use_z, a.folds, a.epochs,
                                     a.lr, a.wd, a.batch, a.gamma, a.seed, a.device)
            results.append((nl, h, npar, m, s))
            print(f"{nl:>6} {h:>7} {npar:>11,} | {m:>8.3f} {s:>7.3f}")

    baseline = next(r for r in results if r[0] == 1 and r[1] == 128)
    best = max(results, key=lambda r: r[3])
    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(f"  smallest (1 layer, 128): F1 {baseline[3]:.3f} ± {baseline[4]:.3f}  "
          f"({baseline[2]:,} params)")
    print(f"  best ({best[0]} layer, {best[1]}):     F1 {best[3]:.3f} ± {best[4]:.3f}  "
          f"({best[2]:,} params)")
    gain = best[3] - baseline[3]
    print(f"  gain from scaling up: {gain:+.3f} F1 for {best[2]/baseline[2]:.0f}x params")
    print()
    if gain <= 0.01:
        print("  VERDICT: scaling the GRU does NOT help — one small layer suffices.")
        print("  The routing signal is simple; a bigger recurrent model just adds")
        print("  parameters. Supports a lean-model design (report the small model).")
    elif gain <= 0.03:
        print("  VERDICT: marginal gain from scaling; parameter cost likely not worth it.")
    else:
        print("  VERDICT: scaling helps meaningfully; the larger GRU captures signal")
        print("  the small one misses.")
    print()


if __name__ == "__main__":
    main()
