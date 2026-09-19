"""
check_turns.py
--------------
Verify every conversation in the mean and top-K tensor folders has exactly 3 turns
(conv_XXXX_turn_01, _turn_02, _turn_03). Flags:
  * conversations with FEWER than 3 turns (partial — interrupted extraction)
  * conversations with MORE than 3 turns (unexpected)
  * turn NUMBERING gaps (e.g. turn_01 + turn_03 but no turn_02)
  * MISMATCH between the mean folder and the top-K folder (a conv in one but not the other)

Run from the model's root folder (where mean_pool_tensors/ and topk_pool_tensors/ live).

Usage:
  python check_turns.py
  python check_turns.py --mean_dir mean_pool_tensors/multi-turn/moe_routing_tensors \
                        --topk_dir topk_pool_tensors/multi-turn/moe_routing_tensors
"""
import argparse, glob, re, os
from collections import defaultdict

def scan(tdir):
    """conv_id -> sorted list of turn ids present."""
    convs = defaultdict(list)
    for f in glob.glob(os.path.join(tdir, "conv_*_turn_*.pt")):
        m = re.search(r"conv_(\d+)_turn_(\d+)", os.path.basename(f))
        if m:
            convs[int(m.group(1))].append(int(m.group(2)))
    for c in convs:
        convs[c] = sorted(convs[c])
    return convs

def report(name, convs, expected=3):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    total = len(convs)
    complete, partial, extra, gapped = [], [], [], []
    for c, turns in convs.items():
        if turns == list(range(1, expected + 1)):
            complete.append(c)
        elif len(turns) < expected:
            partial.append((c, turns))
        elif len(turns) > expected:
            extra.append((c, turns))
        else:  # right count but wrong numbering (gap)
            gapped.append((c, turns))
    print(f"  total conversations : {total}")
    print(f"  complete (exactly {expected} turns): {len(complete)}")
    print(f"  PARTIAL (<{expected} turns)        : {len(partial)}")
    print(f"  extra   (>{expected} turns)        : {len(extra)}")
    print(f"  gapped  (wrong numbering)          : {len(gapped)}")
    if partial:
        print(f"\n  --- PARTIAL conversations (these should be dropped) ---")
        for c, turns in sorted(partial)[:50]:
            print(f"    conv_{c:04d}: turns {turns}")
        if len(partial) > 50:
            print(f"    ... and {len(partial)-50} more")
    if extra:
        print(f"\n  --- EXTRA-turn conversations ---")
        for c, turns in sorted(extra)[:20]:
            print(f"    conv_{c:04d}: turns {turns}")
    if gapped:
        print(f"\n  --- GAPPED conversations ---")
        for c, turns in sorted(gapped)[:20]:
            print(f"    conv_{c:04d}: turns {turns}")
    return set(complete), set(c for c, _ in partial), set(convs.keys())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean_dir", default="mean_pool_tensors/multi-turn/moe_routing_tensors")
    ap.add_argument("--topk_dir", default="topk_pool_tensors/multi-turn/moe_routing_tensors")
    ap.add_argument("--expected", type=int, default=3)
    a = ap.parse_args()

    mean = scan(a.mean_dir)
    topk = scan(a.topk_dir)

    m_ok, m_partial, m_all = report(f"MEAN  ({a.mean_dir})", mean, a.expected)
    t_ok, t_partial, t_all = report(f"TOPK  ({a.topk_dir})", topk, a.expected)

    # cross-folder consistency
    print(f"\n{'='*60}\nMEAN vs TOPK consistency\n{'='*60}")
    only_mean = sorted(m_all - t_all)
    only_topk = sorted(t_all - m_all)
    print(f"  conversations in BOTH folders : {len(m_all & t_all)}")
    print(f"  in MEAN only (missing from topk): {len(only_mean)} {only_mean[:15]}")
    print(f"  in TOPK only (missing from mean): {len(only_topk)} {only_topk[:15]}")

    all_partial = sorted(m_partial | t_partial)
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  conversations to DROP (partial in either folder): {len(all_partial)}")
    if all_partial:
        print(f"    ids: {all_partial}")
        print(f"\n  -> remove these with drop_truncated-style cleanup, or re-extract them.")
    else:
        print(f"  [OK] every conversation has exactly {a.expected} turns in both folders.")

if __name__ == "__main__":
    main()
