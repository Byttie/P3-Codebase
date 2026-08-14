# RoutingGuard — a MoE routing-signature jailbreak defense

A stateful, per-turn jailbreak detector for **ZAYA-1 8B** that runs on the
model's own **expert-routing fingerprint** instead of text embeddings. Where
DeepContext feeds a fine-tuned BERT vector per turn, RoutingGuard feeds the
pooled MoE router distribution your extraction scripts already produce. A small
GRU tracks how that signature drifts across turns and flags the conversation.

The pipeline has two pooling stages, matching your notes:

* **Local aggregation** (token axis, within one prompt) -> a `(L, E)` routing
  signature. Modes: `mean`, `max` (= top-K with K=1), `top-K mean`.
* **Global aggregation** (prompt axis, across the dataset) -> per-feature
  **Z-score** standardization, which doubles as the classifier's input scaler.
  **Clustering / Mahalanobis are deliberately NOT used** (see Design notes).

---

## Files

| File | Role |
|------|------|
| `zaya_m2s.py` / `zaya_multi_turn.py` | ZAYA extraction. **Now also save `topk_token_probs`** (top-10 token weights per expert) and auto-find the family CSV. Run these to populate `topk_pool_tensors/`. |
| `topk_extraction_patch.py` | Reference snippet the two extractors above were patched from (`pool_layer_topk`). |
| `routing_features.py` | Payload -> feature vector. Chooses local pooling (`mean`/`topk`, `K`). |
| `zscore.py` | Global aggregation: fit `mu`,`sigma` on train, `Z=(X-mu)/(sigma+eps)`; also a standalone `mean|Z|` anomaly score. |
| `build_dataset.py` | Walks the tensor families + refusal JSONs -> labeled, padded sequences. **Layout-aware** (mean vs top-K root, hyphenated family dirs, `moe_routing_tensors/` nesting). |
| `model.py` | `RoutingGuard`: TurnEncoder -> GRU -> hybrid short-circuit head; focal BCE. |
| `train.py` | Grouped split, Z-score scaling, focal-loss training, metrics (F1 / recall / precision / MTTD). |
| `sweep_k.py` | The "sweet spot for K" ablation: `mean`, `top1..topK` in one table. |
| `infer_stream.py` | Stateful real-time scoring — one turn at a time, carries the GRU hidden state. |
| `make_synth_topk.py` | Synthetic tensors (with a token axis) that write the **real** layout, to smoke-test the whole thing without ZAYA. |

---

## Expected directory layout

`build_dataset.py` / `sweep_k.py` take `--root <PROJECT>` (the folder that holds
everything below). This mirrors your disk exactly:

```
<PROJECT>/                         e.g.  E:/P3 Defense
  mean_pool_tensors/               (already extracted — mean-pooled, no token axis)
    m2s-pythonize/
      moe_routing_tensors/  prompt_0000.pt ... prompt_0099.pt
      curated_dataset_pythonize.csv
    multi-turn/
      moe_routing_tensors/  conv_0000_turn_01.pt ... conv_0099_turn_03.pt
      curated_dataset_multi_turn.csv
  topk_pool_tensors/               (to extract — same tensors + topk_token_probs)
    m2s-pythonize/  moe_routing_tensors/  ...      <- empty until Step 0
    multi-turn/     moe_routing_tensors/  ...      <- empty until Step 0
  refusals/
    m2s_pythonize_refusals.json
    m2s_hyphenize_refusals.json      (optional families)
    m2s_numberize_refusals.json
    multi_turn_refusals.json
```

* **Pooling mode picks the root.** `--local_pool mean` reads `mean_pool_tensors/`;
  `--local_pool topk` (and `max`, which is `top1`) reads `topk_pool_tensors/`.
* `prompt_id` (m2s) and `conversation_id` (multi-turn) are the SAME scenario
  index `S` (0..99), so a scenario's m2s and multi-turn representations are
  grouped together and never straddle the train/val split.
* Folder names live at the top of `build_dataset.py` (`POOL_DIRNAME`,
  `FAMILY_DIRNAME`, `TENSOR_SUBDIR`) — edit there if yours differ.

---

## Step 0 - (once) re-extract to enable max / top-K pooling

Your `mean_pool_tensors/` files were mean-pooled over tokens at extraction time,
so the token axis is gone. `mean` works on them as-is; **`max` and `top-K` do not**.
The patched `zaya_m2s.py` / `zaya_multi_turn.py` now ADD one field per file,
`topk_token_probs` of shape `(L, KMAX=10, E)`, from which any `K <= 10` (plus
`max` = K=1) is derived at feature-build time — no re-running the model per K.

To populate `topk_pool_tensors/`, drop the patched extractor into each family
folder there and run it (it auto-detects `curated_dataset_pythonize.csv` /
`curated_dataset_multi_turn.csv` next to it and writes `moe_routing_tensors/`).

> **Keep labels and tensors from the SAME generation pass.** The refusal JSONs
> you already have came from the mean-pass generations (sampled: `do_sample=True`,
> `temperature=0.7`, `seed=1234`). A fresh top-K pass generates again and may
> refuse/comply differently on some prompts, so the old labels can silently
> mismatch the new tensors. Two clean options:
>
> 1. **Recommended — one pass, not two.** Because the patched extractor saves
>    *both* `pooled_probs` (mean) and `topk_token_probs`, run it once into
>    `topk_pool_tensors/` and re-derive the refusal JSONs from that run's
>    `benchmark_generation_outputs.json`. Then `--local_pool mean` and `topk`
>    both read one root (the resolver lets `mean` borrow the top-K root), and
>    there is zero label drift. Keep `mean_pool_tensors/` as your PCA artifact.
> 2. **Keep the mean labels authoritative.** Re-extract top-K deterministically
>    (`do_sample=False`) so generations reproduce, and still re-derive labels
>    from the new run to be safe.

If you skip Step 0 entirely, use `--local_pool mean` everywhere.

---

## Step 1 - pick a target

| Mode | Positive | Negative | Needs benign? | Use for |
|------|----------|----------|---------------|---------|
| **T1** attack-vs-benign | every attack turn | benign turns | **yes** | deployable guardrail |
| **T2** jailbreak-success | complied attack | refused attack | no | start now / analysis |
| **T3** refusal-replica | refused turn | complied | no | *avoid* (labels successful jailbreaks "safe") |

You focused on **pythonize + multiturn** (highest ASR: 85% / 69%). Those two
also overlap most in your PCA, so **T2 is hardest exactly there**; **T1** (with
benign routing tensors) is the easier, more useful target and the one where
top-K pooling pays off most (it recovers the diluted trigger spike that makes a
pythonized attack look like ordinary code).

---

## Step 2 - build the dataset

```bash
# T2, focus families, top-K local pooling with K=5  (reads topk_pool_tensors/)
python build_dataset.py \
    --root "E:/P3 Defense" --refusal_dir "E:/P3 Defense/refusals" \
    --mode T2 --families pythonize multiturn \
    --local_pool topk --K 5 \
    --out ds.pt

# max pooling instead:            --local_pool topk --K 1
# mean (reads mean_pool_tensors): --local_pool mean
# T1 (needs benign):  --mode T1 --benign_dir "E:/P3 Defense/<pool>_pool_tensors/benign"
```

## Step 3 - train

```bash
python train.py --data ds.pt --epochs 40 --out guard.pt
# add the mean|Z| anomaly feature alongside learned ones:
python train.py --data ds.pt --append_zscore --out guard.pt
```

## Step 4 - find the K sweet spot (ablation)

```bash
python sweep_k.py --root "E:/P3 Defense" --refusal_dir "E:/P3 Defense/refusals" \
    --mode T2 --families pythonize multiturn --Kmax 10
```
Prints `mean` (baseline, from `mean_pool_tensors/`) and `top1..top10` (from
`topk_pool_tensors/`) with F1 / recall / precision / MTTD. Rows that need the
top-K root print `n/a` until Step 0 is done. Expect F1 to rise to a peak around
`K ~= (#trigger tokens)` then fall as non-trigger tokens dilute it — that peak
is your empirical K.

## Step 5 - stateful real-time inference

```bash
python infer_stream.py --ckpt guard.pt \
    --conv_glob "E:/P3 Defense/topk_pool_tensors/multi-turn/moe_routing_tensors/conv_0007_turn_*.pt"
```
Feeds one turn at a time, carries the GRU state (and the training scaler +
feature config from the checkpoint), prints a risk score per turn, and flags the
turn it crosses threshold.

---

## Smoke test (no real data)

```bash
python make_synth_topk.py --root synth_proj --n 100
python sweep_k.py --root synth_proj --refusal_dir synth_proj/refusals \
    --mode T2 --families pythonize multiturn --Kmax 10
```
The generator writes the **same** two-root layout as production. The signal is
buried in a few spike tokens, so `mean` caps below `max`/`top-K` — confirming the
pooling path is wired correctly.

---

## Design notes

* **Two token segments.** `topk_token_probs` = which expert the model leans on
  *while answering*. (Add a prompt-segment block later if you want the trigger
  word's prefill spike; enable it via `FeatureConfig`.)
* **Global aggregation = Z-score, no clustering.** Your PCA shows refused/complied
  overlap for pythonize + multiturn, so unsupervised cluster "safe zones"
  (K-Means / GMM) won't separate them, and a full covariance/Mahalanobis matrix
  blows up with the expert count. Z-score standardization + a supervised head is
  the route; `mean|Z|` is available as an optional extra anomaly feature.
* **No leakage.** Split is grouped by scenario `S`, so a scenario's m2s variants
  and its multi-turn version stay on the same side.
```
