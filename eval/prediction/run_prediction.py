#!/usr/bin/env python
"""Generate next-observation predictions on held-out transition JSONL."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval.common.client import chat, make_client


NEXT_STATE_RE = re.compile(r"<next_state>(.*?)</next_state>", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
TASK_RE = re.compile(r"Task:\s*(.*?)\s*\n\nInteraction history:", re.S)
OBS_RE = re.compile(r"Current observation:\s*(.*?)\s*\nAction taken:", re.S)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="held-out intermediate JSONL")
    parser.add_argument("--base-url", required=True, help="model server root URL without /v1")
    parser.add_argument("--model", default="student")
    parser.add_argument("--tag", required=True, help="condition name stored in each output row")
    parser.add_argument("--limit-per-split", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    by_split = {}
    for path in args.input:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    split = row.get("meta", {}).get("split", "default")
                    by_split.setdefault(split, []).append(row)

    rng = random.Random(args.seed)
    samples = []
    for split, rows in sorted(by_split.items()):
        rng.shuffle(rows)
        for row in rows[: args.limit_per_split]:
            samples.append((len(samples), split, row))

    client = make_client(args.base_url)

    def predict(item):
        sample_id, split, row = item
        prompt = row["prompt"]
        try:
            content, _ = chat(
                client,
                args.model,
                [{"role": "user", "content": prompt}],
                args.temperature,
                args.max_tokens,
            )
        except Exception as error:
            content = f"__CALL_FAILED__ {error}"
        next_state = NEXT_STATE_RE.search(content)
        reasoning = THINK_RE.search(content)
        task_match = TASK_RE.search(prompt)
        observation_match = OBS_RE.search(prompt)
        return {
            "sample_id": sample_id,
            "tag": args.tag,
            "split": split,
            "task": task_match.group(1) if task_match else "",
            "current_obs": observation_match.group(1) if observation_match else "",
            "action": row.get("meta", {}).get("action"),
            "target_next_state": row["target_next_state"],
            "pred_next_state": next_state.group(1).strip() if next_state else None,
            "pred_think": reasoning.group(1).strip() if reasoning else None,
            "pred_full": content,
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(predict, item) for item in samples]
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 200 == 0:
                print(f"{count}/{len(samples)} predictions complete", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in sorted(results, key=lambda x: x["sample_id"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    parsed = sum(row["pred_next_state"] is not None for row in results)
    print(f"wrote {len(results)} predictions to {args.output}; parsed={parsed / max(len(results), 1):.2%}")


if __name__ == "__main__":
    main()
