#!/usr/bin/env python
"""Run the paper's tier-B VisualWebArena task evaluation."""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import multiprocessing
import os
import sys
import time
import traceback

from eval.common.client import chat, make_client
from eval.vwa.prompts import (
    SYSTEM_PROMPT,
    build_user_text,
    format_history,
    parse_response,
    tier_b_observation,
)


PARSE_FAILURE_LIMIT = 3
REPEAT_ACTION_LIMIT = 5
MAX_OBSERVATION_CHARS = 3840 * 4


def encode_image_array(array) -> str:
    """Encode a screenshot array as a base64 PNG."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def encode_image_file(path: str) -> str:
    """Encode a task input image as base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_model(client, model: str, user_text: str, screenshot: str, task_images: list[str], max_tokens: int):
    """Request one multimodal browser action."""
    content = [{"type": "text", "text": user_text}]
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
        for image in task_images
    )
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}}
    )
    return chat(
        client,
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=1.0,
        top_p=0.9,
        max_tokens=max_tokens,
    )


def run_episode(environment, client, args, task: dict) -> dict:
    """Evaluate one task and sampling index using the official environment."""
    from browser_env import create_id_based_action, create_stop_action
    from browser_env.actions import ActionParsingError
    from evaluation_harness import evaluator_router

    config_file = os.path.join(
        args.vwa_root,
        "config_files",
        "vwa",
        f"test_{task['site']}",
        f"{task['task_id']}.json",
    )
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    objective = config["intent"]

    task_images = []
    image_field = config.get("image")
    for path in image_field if isinstance(image_field, list) else ([image_field] if image_field else []):
        full_path = path if os.path.isabs(path) else os.path.join(args.vwa_root, path)
        if os.path.exists(full_path):
            task_images.append(encode_image_file(full_path))

    observation, info = environment.reset(options={"config_file": config_file})
    trajectory = [{"observation": observation, "info": info}]
    history = []
    parse_failures = 0
    repeated_actions = 0
    last_action = None
    stopped_by = "max_steps"

    for _ in range(args.max_steps):
        observation_text = tier_b_observation(observation["text"])
        observation_text = observation_text[:MAX_OBSERVATION_CHARS]
        user_text = build_user_text(
            objective,
            format_history(history),
            observation_text,
            environment.page.url,
        )
        raw, _ = call_model(
            client,
            args.model,
            user_text,
            encode_image_array(observation["image"]),
            task_images,
            args.max_tokens,
        )
        parsed = parse_response(raw)
        action = None
        if parsed["parse_ok"]:
            try:
                action = create_id_based_action(parsed["action"])
            except ActionParsingError:
                parsed["parse_ok"] = False

        history.append(parsed)
        if not parsed["parse_ok"]:
            parse_failures += 1
            if parse_failures >= PARSE_FAILURE_LIMIT:
                stopped_by = "parse_failures"
                break
            continue
        parse_failures = 0

        if parsed["action"] == last_action:
            repeated_actions += 1
            if repeated_actions >= REPEAT_ACTION_LIMIT:
                stopped_by = "repeating_actions"
                break
        else:
            repeated_actions = 0
        last_action = parsed["action"]

        trajectory.append(action)
        if action["action_type"] == 17:
            stopped_by = "stop"
            break
        observation, _, terminated, _, info = environment.step(action)
        trajectory.append({"observation": observation, "info": info})
        if terminated:
            stopped_by = "environment_terminated"
            break

    if stopped_by != "stop":
        trajectory.append(create_stop_action(""))
    evaluator = evaluator_router(config_file)
    score = evaluator(trajectory=trajectory, config_file=config_file, page=environment.page)
    score = float(score)
    return {
        "environment": "vwa",
        "task_id": f"{task['site']}/{task['task_id']}",
        "sample": task["_sample"],
        "success": score == 1.0,
        "score": score,
        "site": task["site"],
        "steps": len(history),
        "stopped_by": stopped_by,
    }


def worker(args, worker_id: int, tasks: list[dict]):
    """Reuse one browser environment across a worker's task shard."""
    os.chdir(args.vwa_root)
    sys.path.insert(0, args.vwa_root)
    os.environ.setdefault("OPENAI_API_KEY", "placeholder")
    time.sleep(worker_id * 0.25)

    from browser_env import ScriptBrowserEnv

    urls = [url.strip() for url in args.base_url.split(",") if url.strip()]
    client = make_client(urls[worker_id % len(urls)])

    def make_environment():
        return ScriptBrowserEnv(
            headless=True,
            observation_type="image_som",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 2048},
        )

    environment = make_environment()
    shard_output = f"{args.output}.w{worker_id}"
    completed = set()
    for path in glob.glob(f"{args.output}.w*"):
        if path.endswith(".failed"):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    completed.add((row["task_id"], row["sample"]))

    with open(shard_output, "a", encoding="utf-8") as output, open(
        f"{shard_output}.failed", "a", encoding="utf-8"
    ) as failed:
        for task in tasks:
            key = (f"{task['site']}/{task['task_id']}", task["_sample"])
            if key in completed:
                continue
            try:
                result = run_episode(environment, client, args, task)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
            except Exception:
                failed.write(
                    json.dumps(
                        {
                            "task_id": key[0],
                            "sample": key[1],
                            "error": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                failed.flush()
                try:
                    environment.close()
                except Exception:
                    pass
                environment = make_environment()
    try:
        environment.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vwa-root", required=True, help="VisualWebArena checkout at commit 89f5af2")
    parser.add_argument("--tasks", required=True, help="paper evaluation-task JSONL")
    parser.add_argument("--base-url", required=True, help="one or more comma-separated model server roots")
    parser.add_argument("--model", default="student")
    parser.add_argument("--runs", type=int, default=64, help="samples per task")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.vwa_root = os.path.abspath(args.vwa_root)
    args.output = os.path.abspath(args.output)

    with open(args.tasks, encoding="utf-8") as f:
        base_tasks = [json.loads(line) for line in f if line.strip()]
    tasks = [dict(task, _sample=sample) for sample in range(args.runs) for task in base_tasks]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    shards = [tasks[index :: args.workers] for index in range(args.workers)]
    processes = []
    for worker_id, shard in enumerate(shards):
        if not shard:
            continue
        process = multiprocessing.Process(target=worker, args=(args, worker_id, shard))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    results = {}
    for path in glob.glob(f"{args.output}.w*"):
        if path.endswith(".failed"):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    results[(row["task_id"], row["sample"])] = row
    with open(args.output, "w", encoding="utf-8") as f:
        for key in sorted(results):
            f.write(json.dumps(results[key], ensure_ascii=False) + "\n")
    print(f"wrote {len(results)}/{len(tasks)} episodes to {args.output}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
