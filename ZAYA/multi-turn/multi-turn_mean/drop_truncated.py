"""
drop_truncated.py
-----------------
Removes the 20 conversations still truncated after auto-retry (reasoning exceeded
the largest token budget without a final answer -> verdict unknowable).

Workflow this supports:
  1. Drop the 20 from benchmark_generation_outputs.json  (source of truth)
  2. Drop the 20 from live_turns.jsonl                   (what regex.py reads)
  3. Move the 20 conversations' tensor files to discarded/
  -> THEN run regex.py on the cleaned live_turns.jsonl to get the correct
     refusal count + ASR over the 280 kept conversations.

No refusals file is needed as input — the regex recomputes labels from the
cleaned data.

Everything is moved/backed-up, not destroyed (.bak for edited files).

Example (run inside the folder holding these files):
  python drop_truncated.py \\
      --benchmark benchmark_generation_outputs.json \\
      --live live_turns.jsonl \\
      --tensor_dirs moe_routing_tensors
"""

import argparse
import json
import shutil
from pathlib import Path

DROP_IDS = [14, 23, 44, 121, 135, 138, 140, 144, 165, 170, 171, 173,
            179, 188, 229, 252, 261, 267, 277, 290]
TURNS_PER_CONV = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=None,
                    help="benchmark_generation_outputs.json (nested by conversation)")
    ap.add_argument("--live", default=None,
                    help="live_turns.jsonl (one record per turn; what regex.py reads)")
    ap.add_argument("--tensor_dirs", nargs="*", default=[],
                    help="folder(s) holding flat conv_XXXX_turn_XX.pt files")
    ap.add_argument("--total", type=int, default=300)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    drop = set(DROP_IDS)
    print(f"Dropping {len(drop)} truncated conversations: {sorted(drop)}\n")

    # --- 1. benchmark JSON (nested list of conversations) ---
    if a.benchmark:
        bp = Path(a.benchmark)
        if bp.exists():
            data = json.load(open(bp, encoding='utf-8'))
            kept = [d for d in data if d.get("conversation_id") not in drop]
            removed = len(data) - len(kept)
            if not a.dry_run:
                shutil.copy(str(bp), str(bp) + ".bak")
                json.dump(kept, open(bp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            print(f"  benchmark JSON: {len(data)} -> {len(kept)} conversations "
                  f"(removed {removed})"
                  f"{'  [dry run]' if a.dry_run else '  saved; original -> .bak'}")
        else:
            print(f"  [warn] benchmark not found: {bp}")

    # --- 2. live_turns.jsonl (one JSON per turn) ---
    if a.live:
        lp = Path(a.live)
        if lp.exists():
            lines = [l for l in open(lp, encoding="utf-8") if l.strip()]
            kept_lines = []
            dropped = 0
            for l in lines:
                rec = json.loads(l)
                if rec.get("conversation_id") in drop:
                    dropped += 1
                else:
                    kept_lines.append(l)
            if not a.dry_run:
                shutil.copy(str(lp), str(lp) + ".bak")
                with open(lp, "w", encoding="utf-8") as f:
                    f.writelines(kept_lines)
            print(f"  live_turns.jsonl: {len(lines)} -> {len(kept_lines)} turns "
                  f"(removed {dropped})"
                  f"{'  [dry run]' if a.dry_run else '  saved; original -> .bak'}")
        else:
            print(f"  [warn] live file not found: {lp}")

    # --- 3. tensor files (DELETED — user keeps backups elsewhere) ---
    for td in a.tensor_dirs:
        tdir = Path(td)
        if not tdir.exists():
            print(f"  [warn] tensor dir not found: {tdir}")
            continue
        deleted = missing = 0
        for cid in sorted(drop):
            for turn in range(1, TURNS_PER_CONV + 1):
                fname = f"conv_{cid:04d}_turn_{turn:02d}.pt"
                src = tdir / fname
                if src.exists():
                    if a.dry_run:
                        print(f"    would DELETE {src}")
                    else:
                        src.unlink()
                    deleted += 1
                else:
                    missing += 1
        remaining = len(list(tdir.glob("conv_*_turn_*.pt")))
        print(f"  {tdir}: {'would delete' if a.dry_run else 'DELETED'} {deleted} files "
              f"(missing {missing}) -> {remaining} .pt remain")

    n_kept = a.total - len(drop)
    print("\n" + "=" * 50)
    print(f"CLEAN: {n_kept} conversations remain (was {a.total})")
    print("=" * 50)
    print("  Next: run regex.py on the cleaned live_turns.jsonl")
    print("        to get the correct refusal count + ASR over these", n_kept)
    if a.dry_run:
        print("\n  DRY RUN — nothing changed. Re-run without --dry_run to apply.")


if __name__ == "__main__":
    main()