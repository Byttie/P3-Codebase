"""
match_jsonl_tensors.py
----------------------
Cross-reference live_turns.jsonl against the tensor folders to find mismatches:

  * TENSORS WITHOUT JSONL : conversations that have .pt files but no entry in
        live_turns.jsonl (orphans — no label source). If benchmark JSON has their
        responses, rebuild the JSONL; otherwise drop these tensors.

  * JSONL WITHOUT TENSORS : conversations labeled in the JSONL but with no .pt
        files (shouldn't normally happen; would break build_dataset).

  * MEAN vs TOPK          : tensor conversations present in one pooling folder
        but not the other.

  * (optional) BENCHMARK  : if --benchmark is given, reports whether the orphans'
        responses exist there (so you know if a rebuild can recover them).

Run from the model root (where the folders + live_turns.jsonl live).

Usage:
  python match_jsonl_tensors.py
  python match_jsonl_tensors.py --benchmark benchmark_generation_outputs.json
"""
import argparse, glob, re, os, json
from collections import defaultdict


def tensor_convs(tdir):
    """conv_id -> sorted turn ids present in a tensor folder."""
    convs = defaultdict(list)
    for f in glob.glob(os.path.join(tdir, "conv_*_turn_*.pt")):
        m = re.search(r"conv_(\d+)_turn_(\d+)", os.path.basename(f))
        if m:
            convs[int(m.group(1))].append(int(m.group(2)))
    return {c: sorted(v) for c, v in convs.items()}


def jsonl_convs(path):
    """conv_id -> sorted turn ids present in live_turns.jsonl."""
    convs = defaultdict(list)
    if not os.path.exists(path):
        print(f"[warn] {path} not found")
        return {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            convs[int(r["conversation_id"])].append(int(r.get("turn_id", 0)))
    return {c: sorted(v) for c, v in convs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean_dir", default="mean_pool_tensors/multi-turn/moe_routing_tensors")
    ap.add_argument("--topk_dir", default="topk_pool_tensors/multi-turn/moe_routing_tensors")
    ap.add_argument("--live", default="live_turns.jsonl")
    ap.add_argument("--benchmark", default=None,
                    help="benchmark_generation_outputs.json — checks if orphans' "
                         "responses can be recovered")
    ap.add_argument("--dump", default=None,
                    help="write the orphan / missing id lists to this JSON file")
    a = ap.parse_args()

    mean = tensor_convs(a.mean_dir)
    topk = tensor_convs(a.topk_dir)
    live = jsonl_convs(a.live)

    mean_ids, topk_ids, live_ids = set(mean), set(topk), set(live)
    # tensor set = conversations present in BOTH pooling folders (the usable ones)
    tensor_ids = mean_ids & topk_ids

    print("=" * 64)
    print("COUNTS")
    print("=" * 64)
    print(f"  mean-folder conversations : {len(mean_ids)}")
    print(f"  topk-folder conversations : {len(topk_ids)}")
    print(f"  tensors in BOTH folders   : {len(tensor_ids)}")
    print(f"  live_turns.jsonl convs    : {len(live_ids)}")

    # --- mean vs topk ---
    only_mean = sorted(mean_ids - topk_ids)
    only_topk = sorted(topk_ids - mean_ids)
    print("\n" + "=" * 64)
    print("MEAN vs TOPK")
    print("=" * 64)
    print(f"  in MEAN only : {len(only_mean)} {only_mean[:20]}")
    print(f"  in TOPK only : {len(only_topk)} {only_topk[:20]}")

    # --- tensors without jsonl (orphans) ---
    orphans = sorted(tensor_ids - live_ids)
    print("\n" + "=" * 64)
    print("TENSORS WITHOUT JSONL  (orphans — have .pt but no label source)")
    print("=" * 64)
    print(f"  count: {len(orphans)}")
    print(f"  ids  : {orphans}")

    # --- jsonl without tensors ---
    ghost = sorted(live_ids - tensor_ids)
    print("\n" + "=" * 64)
    print("JSONL WITHOUT TENSORS  (labeled but no .pt — would break build)")
    print("=" * 64)
    print(f"  count: {len(ghost)}")
    print(f"  ids  : {ghost}")

    # --- benchmark recovery check ---
    bench_ids = set()
    if a.benchmark and os.path.exists(a.benchmark):
        bench = json.load(open(a.benchmark, encoding="utf-8"))
        bench_ids = set(d["conversation_id"] for d in bench)
        recoverable = [o for o in orphans if o in bench_ids]
        print("\n" + "=" * 64)
        print("BENCHMARK RECOVERY")
        print("=" * 64)
        print(f"  benchmark conversations       : {len(bench_ids)}")
        print(f"  orphans recoverable from it   : {len(recoverable)}/{len(orphans)}")
        if len(recoverable) == len(orphans) and orphans:
            print("  -> ALL orphans are in the benchmark: rebuild live_turns.jsonl")
            print("     from benchmark_generation_outputs.json, then re-run the regex.")
        elif orphans:
            lost = [o for o in orphans if o not in bench_ids]
            print(f"  -> {len(lost)} orphans NOT in benchmark either (responses lost):")
            print(f"     {lost}")
            print("     drop these tensors; rebuild JSONL for the rest.")

    # --- verdict ---
    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    if not orphans and not ghost and not only_mean and not only_topk:
        print("  [OK] tensors and JSONL match exactly. Ready to build.")
    else:
        print("  MISMATCH — reconcile before building:")
        if orphans:
            print(f"    - {len(orphans)} orphan tensors need labels (rebuild JSONL) or dropping")
        if ghost:
            print(f"    - {len(ghost)} jsonl entries have no tensors (drop from labels)")
        if only_mean or only_topk:
            print(f"    - mean/topk folders disagree on {len(only_mean)+len(only_topk)} convs")

    if a.dump:
        json.dump({"orphans_tensors_without_jsonl": orphans,
                   "ghost_jsonl_without_tensors": ghost,
                   "mean_only": only_mean, "topk_only": only_topk,
                   "recoverable_from_benchmark": sorted(set(orphans) & bench_ids)},
                  open(a.dump, "w"), indent=2)
        print(f"\n  id lists written -> {a.dump}")


if __name__ == "__main__":
    main()
