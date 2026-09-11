#!/usr/bin/env python
"""Judge next-observation predictions with an OpenAI-compatible LLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval.common.client import chat, make_client
from eval.prediction.judge_prompts import JUDGE_PROMPTS


VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", re.I)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True, choices=sorted(JUDGE_PROMPTS))
    parser.add_argument("--input", required=True, help="prediction JSONL")
    parser.add_argument("--judge-url", nargs="+", required=True, help="judge server roots without /v1")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    clients = [make_client(url) for url in args.judge_url]
    prompt_template = JUDGE_PROMPTS[args.environment]

    def judge(item):
        index, row = item
        if not row.get("pred_next_state"):
            return {**row, "verdict": "INCORRECT", "judge_reason": "no next-state span"}
        prompt = prompt_template.format(
            task=row.get("task", ""),
            current_obs=row.get("current_obs", ""),
            action=row.get("action", ""),
            ground_truth=row["target_next_state"],
            prediction=row["pred_next_state"],
        )
        try:
            raw, _ = chat(
                clients[index % len(clients)],
                args.judge_model,
                [{"role": "user", "content": prompt}],
                0.0,
                512,
            )
        except Exception as error:
            return {**row, "verdict": "PARSE_FAIL", "judge_reason": repr(error)}
        matches = list(VERDICT_RE.finditer(raw))
        verdict = matches[-1].group(1).upper() if matches else "PARSE_FAIL"
        return {**row, "verdict": verdict, "judge_reason": raw[:500]}

    judged = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(judge, item) for item in enumerate(rows)]
        for count, future in enumerate(as_completed(futures), 1):
            judged.append(future.result())
            if count % 200 == 0:
                print(f"{count}/{len(rows)} predictions judged", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in sorted(judged, key=lambda x: x["sample_id"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_split = defaultdict(Counter)
    for row in judged:
        by_split[row["split"]][row["verdict"]] += 1
    for split, counts in sorted(by_split.items()):
        total = sum(counts.values())
        accuracy = counts["CORRECT"] / total if total else 0.0
        print(f"{split}: accuracy={accuracy:.2%} ({dict(counts)})")


if __name__ == "__main__":
    main()
