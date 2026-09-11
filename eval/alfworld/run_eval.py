#!/usr/bin/env python
"""Run ALFWorld task-success evaluation in list or no-list mode."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_episode(job: tuple) -> dict:
    """Evaluate one game and sampling index in an isolated process."""
    (
        gamefile,
        split,
        sample_index,
        base_url,
        variant,
        model,
        max_steps,
        temperature,
        max_tokens,
    ) = job

    from eval.alfworld.agent import ReActAgent
    from eval.alfworld.env import ALFWorldEnvironment

    environment = ALFWorldEnvironment(gamefile)
    agent = ReActAgent(base_url, variant, model, temperature, max_tokens)
    task, observation, admissible_actions = environment.reset()
    agent.task = task

    won = False
    done_reason = "max_steps"
    for step_index in range(1, max_steps + 1):
        try:
            _, action, _ = agent.act(step_index, observation, admissible_actions)
        except Exception as error:
            done_reason = f"model_error: {error}"
            break
        result = environment.step(action)
        agent.record(observation, action, result.observation)
        observation = result.observation
        admissible_actions = result.admissible_actions
        if result.done:
            won = result.won
            done_reason = "won" if won else "done"
            break

    return {
        "environment": "alfworld",
        "task_id": gamefile,
        "sample": sample_index,
        "success": won,
        "split": split,
        "task_type": environment.task_type,
        "steps": len(agent.transitions),
        "done_reason": done_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="model server root URL without /v1")
    parser.add_argument("--variant", choices=["give", "nolist"], required=True)
    parser.add_argument("--splits", nargs="+", default=["valid_seen", "valid_unseen"])
    parser.add_argument("--runs", type=int, default=64, help="samples per game")
    parser.add_argument("--limit", type=int, help="optional games per split")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--model", default="student")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from eval.alfworld.env import list_split_games

    jobs = []
    for split in args.splits:
        games = list_split_games(split, args.limit, args.seed)
        for gamefile in games:
            for sample_index in range(args.runs):
                jobs.append(
                    (
                        gamefile,
                        split,
                        sample_index,
                        args.base_url,
                        args.variant,
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
            if count % 50 == 0:
                print(f"{count}/{len(jobs)} episodes complete", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in sorted(results, key=lambda x: (x["task_id"], x["sample"])):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    successes = sum(row["success"] for row in results)
    print(f"wrote {len(results)} episodes to {args.output}; success={successes / max(len(results), 1):.2%}")


if __name__ == "__main__":
    main()
