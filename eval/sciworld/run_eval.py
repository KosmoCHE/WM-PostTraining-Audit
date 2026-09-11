#!/usr/bin/env python
"""Run ScienceWorld task-success evaluation on a fixed item list."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_episode(job: tuple) -> dict:
    """Evaluate one ScienceWorld variation and sampling index."""
    (
        item_id,
        task,
        variation,
        sample_index,
        base_url,
        model,
        max_steps,
        temperature,
        max_tokens,
    ) = job

    from scienceworld import ScienceWorldEnv

    from eval.sciworld.agent import ReActAgent

    environment = None
    try:
        environment = ScienceWorldEnv("", envStepLimit=max_steps + 10)
        environment.load(task, variation, simplificationStr="")
        observation, info = environment.reset()
        agent = ReActAgent(base_url, model, temperature, max_tokens)
        agent.task = info["taskDesc"]

        final_score = 0
        done_reason = "max_steps"
        step_index = 0
        for step_index in range(1, max_steps + 1):
            _, action = agent.act(step_index, observation)
            executed_action = action or "look around"
            previous_observation = observation
            observation, _, done, info = environment.step(executed_action)
            final_score = int(info.get("score", 0))
            agent.record(previous_observation, action, observation)
            if done:
                done_reason = "done"
                break

        return {
            "environment": "sciworld",
            "task_id": item_id,
            "sample": sample_index,
            "success": final_score >= 100,
            "task": task,
            "variation": variation,
            "score": final_score,
            "steps": step_index,
            "done_reason": done_reason,
        }
    except Exception as error:
        return {
            "environment": "sciworld",
            "task_id": item_id,
            "sample": sample_index,
            "success": False,
            "task": task,
            "variation": variation,
            "score": 0,
            "steps": 0,
            "done_reason": "error",
            "error": repr(error),
        }
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-file", required=True, help="JSONL with item_id, task, and variation")
    parser.add_argument("--base-url", required=True, help="model server root URL without /v1")
    parser.add_argument("--runs", type=int, default=64, help="samples per item")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model", default="student")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.items_file, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    jobs = []
    for item in items:
        for sample_index in range(args.runs):
            jobs.append(
                (
                    item["item_id"],
                    item["task"],
                    item["variation"],
                    sample_index,
                    args.base_url,
                    args.model,
                    args.max_steps,
                    args.temperature,
                    args.max_tokens,
                )
            )

    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = [executor.submit(run_episode, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 25 == 0:
                print(f"{count}/{len(jobs)} episodes complete", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in sorted(results, key=lambda x: (x["task_id"], x["sample"])):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    valid = [row for row in results if row["done_reason"] != "error"]
    successes = sum(row["success"] for row in valid)
    print(
        f"wrote {len(results)} episodes to {args.output}; "
        f"valid={len(valid)}, errors={len(results) - len(valid)}, "
        f"success={successes / max(len(valid), 1):.2%}"
    )


if __name__ == "__main__":
    main()
