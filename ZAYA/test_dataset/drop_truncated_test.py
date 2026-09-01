"""
drop_truncated_test.py
----------------------
Drops the 58 truncated conversations from the 100-prompt TEST set, leaving 42
clean conversations for held-out inference.

These 58 were generated at the old 512-token cap and hit truncation (no closing
</think>), so their refuse/comply verdict is unreliable.

Actions:
  1. Remove the 58 from benchmark_generation_outputs.json  (.bak backup)
  2. Remove the 58 from live_turns.jsonl                    (.bak backup)
  3. Delete their tensor files (you have backups elsewhere)
  -> then run regex.py on the cleaned live_turns.jsonl, then inference.

Example (run inside the TEST folder):
  python drop_truncated_test.py \\
      --benchmark benchmark_generation_outputs.json \\
      --live live_turns.jsonl \\
      --tensor_dirs moe_routing_tensors
"""

import argparse, json, shutil
from pathlib import Path

# 58 truncated conversations in the 100-prompt test set (0-indexed).
DROP_IDS = [3, 7, 12, 18, 20, 22, 23, 25, 26, 28, 31, 33, 35, 37, 41, 42, 45, 46,
            47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 61, 62, 63, 64, 66, 67, 68,
            69, 70, 71, 72, 73, 75, 76, 77, 78, 80, 82, 83, 85, 86, 89, 90, 92, 93,
            94, 95, 96, 97]
TURNS_PER_CONV = 3
TOTAL = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--live", default=None)
    ap.add_argument("--tensor_dirs", nargs="*", default=[])
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    drop = set(DROP_IDS)
    print(f"Dropping {len(drop)} truncated test conversations -> {TOTAL - len(drop)} clean remain\n")

    # 1. benchmark JSON
    if a.benchmark:
        bp = Path(a.benchmark)
        if bp.exists():
            data = json.load(open(bp, encoding="utf-8"))
            kept = [d for d in data if d.get("conversation_id") not in drop]
            if not a.dry_run:
                shutil.copy(str(bp), str(bp) + ".bak")
                json.dump(kept, open(bp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            print(f"  benchmark JSON: {len(data)} -> {len(kept)} conversations"
                  f"{'  [dry]' if a.dry_run else '  (.bak saved)'}")
        else:
            print(f"  [warn] not found: {bp}")

    # 2. live_turns.jsonl
    if a.live:
        lp = Path(a.live)
        if lp.exists():
            lines = [l for l in open(lp, encoding="utf-8") if l.strip()]
            kept = [l for l in lines if json.loads(l).get("conversation_id") not in drop]
            if not a.dry_run:
                shutil.copy(str(lp), str(lp) + ".bak")
                open(lp, "w", encoding="utf-8").writelines(kept)
            print(f"  live_turns.jsonl: {len(lines)} -> {len(kept)} turns"
                  f"{'  [dry]' if a.dry_run else '  (.bak saved)'}")
        else:
            print(f"  [warn] not found: {lp}")

    # 3. tensor files (deleted)
    for td in a.tensor_dirs:
        tdir = Path(td)
        if not tdir.exists():
            print(f"  [warn] tensor dir not found: {tdir}")
            continue
        deleted = missing = 0
        for cid in sorted(drop):
            for turn in range(1, TURNS_PER_CONV + 1):
                src = tdir / f"conv_{cid:04d}_turn_{turn:02d}.pt"
                if src.exists():
                    if not a.dry_run:
                        src.unlink()
                    deleted += 1
                else:
                    missing += 1
        remaining = len(list(tdir.glob("conv_*_turn_*.pt")))
        print(f"  {tdir}: {'would delete' if a.dry_run else 'DELETED'} {deleted} "
              f"(missing {missing}) -> {remaining} .pt remain")

    print(f"\n  CLEAN TEST SET: {TOTAL - len(drop)} conversations")
    print("  Next: run regex.py on the cleaned live_turns.jsonl, then inference.")
    if a.dry_run:
        print("  DRY RUN — nothing changed.")


if __name__ == "__main__":
    main()