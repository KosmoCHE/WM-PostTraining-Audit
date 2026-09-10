#!/usr/bin/env python
"""Remove exact duplicate world-model samples.

Only duplicate ``(prompt, target_next_state)`` pairs are removed. Rows with the
same prompt and different targets are preserved because they may represent
partial observability rather than duplicated data.

Example:
  python dedup_wm_samples.py \
      --in  ../data/wm_rollouts/alfworld/wm_give_samples.jsonl \
      --out ../data/wm_rollouts/alfworld/wm_give_dedup.jsonl
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="world-model sample JSONL")
    ap.add_argument("--out", required=True, help="deduplicated output JSONL")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seen = set()
    n_in = n_out = 0
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            key = (row["prompt"], row["target_next_state"])
            if key in seen:
                continue
            seen.add(key)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    dropped = n_in - n_out
    ratio = dropped / n_in if n_in else 0.0
    print(f"[dedup] {n_in} -> {n_out} (removed {dropped} exact duplicates, {ratio:.1%})")
    print(f"        {args.out}")


if __name__ == "__main__":
    main()
