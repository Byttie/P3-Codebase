# EDA summary -- zaya

- Turns: **5106**  (malicious-complied=2028, benign-complied=3078); dim=1360
- Mean |r| -- Feature 1 (probs): **0.0920**  |  Feature 2 (hist): **0.1075**
- Mean |AUC-0.5| -- probs: **0.0598**  |  hist: **0.0657**
- Best *single* dimension AUC: **0.876** (hist, layer 16, expert 4)

### Top discriminative dimensions

| rank | feature | layer | expert | r | AUC |
|---|---|---|---|---|---|
| 0 | hist | 16 | 4 | +0.624 | 0.876 |
| 1 | probs | 16 | 4 | +0.625 | 0.865 |
| 2 | hist | 33 | 13 | +0.553 | 0.816 |
| 3 | hist | 18 | 11 | +0.497 | 0.791 |
| 4 | probs | 18 | 11 | +0.436 | 0.774 |
| 5 | hist | 31 | 0 | +0.436 | 0.763 |
| 6 | hist | 33 | 14 | -0.432 | 0.239 |
| 7 | probs | 19 | 5 | +0.444 | 0.753 |
| 8 | hist | 19 | 12 | +0.429 | 0.753 |
| 9 | hist | 34 | 2 | +0.447 | 0.751 |
| 10 | probs | 34 | 2 | +0.403 | 0.737 |
| 11 | probs | 9 | 13 | +0.411 | 0.734 |
| 12 | hist | 7 | 7 | +0.373 | 0.732 |
| 13 | hist | 34 | 15 | +0.372 | 0.729 |
| 14 | hist | 15 | 13 | +0.362 | 0.728 |

> Caveats: correlations/AUCs treat turns as independent (they are grouped by conversation) and are **univariate**. The GRU uses multivariate + temporal structure, so these numbers understate the signal it can reach. Small per-dim r is expected and does not contradict a strong detector.
