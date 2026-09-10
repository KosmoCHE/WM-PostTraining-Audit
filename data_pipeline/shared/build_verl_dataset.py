#!/usr/bin/env python
"""Convert intermediate world-model JSONL into verl RLHFDataset parquet.

Input rows use ``{prompt, target_next_state, meta}``. Output rows use::

    prompt: [{"role": "user", "content": <plain text>}]
    data_source: "wm_alfworld_give",
    reward_model: {"style": "none", "ground_truth": ""}
    extra_info: {target_next_state, teacher_prompt_ids, ...meta, index}

OPSD requires a privileged teacher prefix. When ``--tokenizer`` is provided,
the script injects the target observation before the assistant turn, tokenizes
the prefix, and stores ``teacher_prompt_ids`` in ``extra_info``. The helper is
shared with the training implementation to prevent formatting drift.

Example:
  python -m data_pipeline.shared.build_verl_dataset \
      --in  work/alfworld/gt.jsonl \
      --out data/processed/alfworld/gt.parquet \
      --tokenizer Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import os


def to_verl_row(row, idx, data_source, tokenizer):
    extra_info = dict(row.get("meta") or row.get("metadata") or {})
    if "target_next_state" in row:
        extra_info["target_next_state"] = row["target_next_state"]
    if not extra_info.get("target_next_state"):
        raise ValueError(f"row {idx} is missing target_next_state")
    extra_info["index"] = idx
    if tokenizer is not None:
        from verl.utils.wm_opsd import build_privileged_prompt_ids

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        extra_info["teacher_prompt_ids"] = build_privileged_prompt_ids(
            tokenizer, prompt_text, extra_info["target_next_state"]
        )
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "reward_model": {"style": "none", "ground_truth": ""},
        "extra_info": extra_info,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="intermediate world-model JSONL")
    ap.add_argument("--out", required=True, help="output verl parquet")
    ap.add_argument("--data-source", default="wm_alfworld_give")
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="Hugging Face tokenizer used to precompute OPSD teacher_prompt_ids",
    )
    ap.add_argument("--smoke-out", default=None, help="optional small parquet for smoke tests")
    ap.add_argument("--smoke-n", type=int, default=64, help="number of rows in the smoke subset")
    args = ap.parse_args()

    import pandas as pd

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows = []
    with open(args.inp, encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            rows.append(to_verl_row(json.loads(line), idx, args.data_source, tokenizer))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"[done] {len(rows)} samples -> {args.out}")

    if args.smoke_out:
        os.makedirs(os.path.dirname(args.smoke_out) or ".", exist_ok=True)
        pd.DataFrame(rows[: args.smoke_n]).to_parquet(args.smoke_out)
        print(f"       smoke subset ({min(args.smoke_n, len(rows))}) -> {args.smoke_out}")


if __name__ == "__main__":
    main()
