"""
make_everyday_benign.py
-----------------------
Build a benign multi-turn CSV from HuggingFaceTB/everyday-conversations-llama3.1-2k,
matching the EXACT format of curated_dataset.csv so it feeds the same ZAYA extractor.

Why this dataset (vs UltraChat): its conversations are "3-4 short exchanges" with
"brief, clear answers" on everyday topics + elementary science. Whole-conversation
token_length is only 69-286, so ZAYA won't reason for thousands of tokens -> far
less truncation than UltraChat's essay-length prompts.

Output (one column "prompt", one row per conversation):
    [{"role": "user", "content": "..."}, {...}, {...}]   # user turns only

Turn handling:
  Every conversation starts with a greeting turn ("Hi" -> "Hello! How can I help
  you today?"). By default we SKIP that greeting and take the next 3 SUBSTANTIVE
  user turns, so all 3 turns carry real content (matching CoSafe/curated, which
  have no filler greeting). Use --keep_greeting to include it instead.

Requires: pip install datasets

Usage:
  python make_everyday_benign.py --n 1000 --out curated_dataset_benign.csv
  python make_everyday_benign.py --n 1000 --keep_greeting --out benign_with_greeting.csv
"""

import argparse
import csv
import json
import random


def user_turns(messages, skip_greeting=True):
    """Return the list of user-turn contents. If skip_greeting, drop the first
    user turn when it is a short greeting (Hi/Hello/Hey/Hi there)."""
    users = [m["content"] for m in messages
             if isinstance(m, dict) and m.get("role") == "user"
             and m.get("content", "").strip()]
    if skip_greeting and users:
        first = users[0].strip().lower().rstrip("!.,")
        if first in {"hi", "hello", "hey", "hi there", "hey there", "hello there"}:
            users = users[1:]
    return users


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000,
                    help="number of benign conversations to write (default 1000)")
    ap.add_argument("--turns", type=int, default=3,
                    help="user turns to keep per conversation (default 3)")
    ap.add_argument("--keep_greeting", action="store_true",
                    help="keep the opening greeting turn (default: skip it)")
    ap.add_argument("--split", default="train_sft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_chars", type=int, default=3)
    ap.add_argument("--out", default="curated_dataset_benign.csv")
    a = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Needs the datasets library:  pip install datasets")

    print(f"Loading everyday-conversations (split={a.split}) ...")
    # this dataset is small (~2.4k rows) so a normal (non-streaming) load is fine
    ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split=a.split)
    print(f"loaded {len(ds)} conversations")

    skip_greeting = not a.keep_greeting
    pool = []
    skipped_short = 0
    for ex in ds:
        msgs = ex.get("messages")
        if not msgs:
            continue
        us = user_turns(msgs, skip_greeting=skip_greeting)
        if len(us) < a.turns:
            skipped_short += 1
            continue
        turns = us[:a.turns]
        if any(len(t.strip()) < a.min_chars for t in turns):
            continue
        pool.append(turns)

    print(f"usable (>= {a.turns} substantive user turns): {len(pool)}  "
          f"(skipped {skipped_short} too-short)")

    rng = random.Random(a.seed)
    if len(pool) <= a.n:
        chosen = pool
        if len(pool) < a.n:
            print(f"[note] only {len(pool)} available; writing all of them "
                  f"(dataset has ~2.4k rows, request {a.n})")
    else:
        chosen = rng.sample(pool, a.n)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt"])
        for turns in chosen:
            cell = json.dumps([{"role": "user", "content": t} for t in turns],
                              ensure_ascii=False)
            w.writerow([cell])

    print(f"\n[done] wrote {len(chosen)} benign conversations -> {a.out}")
    print(f"       {a.turns} user turns each, greeting {'kept' if a.keep_greeting else 'skipped'}")
    print(f"       short everyday questions -> minimal truncation expected")


if __name__ == "__main__":
    main()
