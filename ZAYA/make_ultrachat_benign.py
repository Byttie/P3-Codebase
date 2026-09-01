"""
make_ultrachat_benign.py
------------------------
Build a benign multi-turn CSV from UltraChat 200k, matching the EXACT format of
your curated_dataset.csv so it feeds the same ZAYA extractor.

Output format (one column "prompt", one row per conversation):
    [{"role": "user", "content": "..."},
     {"role": "user", "content": "..."},
     {"role": "user", "content": "..."}]

Design choices:
  * USER TURNS ONLY. Like your curated set, we keep only the user prompts and
    strip UltraChat's assistant turns — ZAYA generates its own responses during
    extraction, so the conversation history is all ZAYA (consistent with the 300
    curated + 1100 CoSafe conversations).
  * FIRST 3 USER TURNS. To match CoSafe's 3-turn structure. Conversations with
    fewer than 3 user turns are skipped (we need a full 3-turn conversation).
  * 1000 CONVERSATIONS by default (extra headroom so truncation drops still leave
    ~600 clean benign, as planned).
  * Benign by construction -> these will be label 0 (benign) in T1 training.

Requires: pip install datasets

Usage:
  python make_ultrachat_benign.py --n 1000 --out curated_dataset_benign.csv
  # reproducible sampling:
  python make_ultrachat_benign.py --n 1000 --seed 42 --out curated_dataset_benign.csv
"""

import argparse
import csv
import json
import random


def first_n_user_turns(messages, n=3):
    """Return the first n user-turn contents from an UltraChat 'messages' list.
    Returns None if there are fewer than n user turns."""
    users = [m["content"] for m in messages
             if isinstance(m, dict) and m.get("role") == "user" and m.get("content", "").strip()]
    if len(users) < n:
        return None
    return users[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000,
                    help="number of benign conversations to write (default 1000)")
    ap.add_argument("--turns", type=int, default=3,
                    help="user turns to keep per conversation (default 3, matches CoSafe)")
    ap.add_argument("--split", default="train_sft",
                    help="UltraChat 200k split (default train_sft)")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling seed for reproducibility")
    ap.add_argument("--min_chars", type=int, default=3,
                    help="skip user turns shorter than this many characters")
    ap.add_argument("--out", default="curated_dataset_benign.csv")
    a = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Needs the datasets library:  pip install datasets")

    print(f"Loading UltraChat 200k (split={a.split}) ...")
    # streaming=True avoids downloading the whole 200k set into memory
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=a.split, streaming=True)

    rng = random.Random(a.seed)
    kept = []
    scanned = 0
    skipped_short = 0

    # We scan sequentially and keep conversations that have >= `turns` user turns.
    # To avoid front-loading the same conversations every run, we over-scan a pool
    # and sample from it.
    POOL_TARGET = a.n * 3   # scan ~3x to sample from
    pool = []
    for ex in ds:
        scanned += 1
        msgs = ex.get("messages")
        if not msgs:
            continue
        turns = first_n_user_turns(msgs, a.turns)
        if turns is None:
            skipped_short += 1
            continue
        # length guard: all turns must have real content
        if any(len(t.strip()) < a.min_chars for t in turns):
            continue
        pool.append(turns)
        if len(pool) >= POOL_TARGET:
            break

    print(f"scanned {scanned} conversations, {len(pool)} had >= {a.turns} user turns "
          f"({skipped_short} skipped as too short)")

    if len(pool) < a.n:
        print(f"[warn] only {len(pool)} usable conversations found; writing all of them")
        chosen = pool
    else:
        chosen = rng.sample(pool, a.n)

    # write in the EXACT curated format: one column "prompt", JSON list of user dicts
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt"])
        for turns in chosen:
            cell = json.dumps(
                [{"role": "user", "content": t} for t in turns],
                ensure_ascii=False
            )
            w.writerow([cell])

    print(f"\n[done] wrote {len(chosen)} benign conversations -> {a.out}")
    print(f"       format: {a.turns} user turns each, matching curated_dataset.csv")
    print(f"       these are BENIGN (label 0) for T1 training")
    print(f"\nNext: extract with the combined extractor, then drop any refusals/truncations.")


if __name__ == "__main__":
    main()