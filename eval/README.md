# Evaluation

This directory contains the formal task and next-observation evaluators used by the paper. It does not contain mechanism probes, figure generation, case studies, or experiment-result files.

All model calls use an OpenAI-compatible server root such as `http://127.0.0.1:15100`. Start a trained checkpoint with vLLM or another compatible server before running an evaluator:

```bash
vllm serve /path/to/checkpoint \
  --served-model-name student \
  --port 15100
```

## ALFWorld task success

Install ALFWorld and its TextWorld dependencies, download `json_2.1.1`, and point `ALFWORLD_DATA` at its parent directory.

```bash
ALFWORLD_DATA=/path/to/alfworld/data \
python -m eval.alfworld.run_eval \
  --base-url http://127.0.0.1:15100 \
  --variant give \
  --splits valid_seen valid_unseen \
  --runs 64 --max-steps 30 --temperature 1.0 \
  --workers 16 \
  --output outputs/eval/alfworld/gt-give.jsonl
```

Use `--variant nolist` for free-form action generation. Each output row has the standardized fields `task_id`, `sample`, and `success`.

## ScienceWorld task success

Install ScienceWorld and provide a Java runtime through `JAVA_HOME`. The paper uses a fixed 200-item JSONL file with `item_id`, `task`, and `variation` fields.

```bash
JAVA_HOME=/path/to/java \
python -m eval.sciworld.run_eval \
  --items-file /path/to/agentgym_test_200.jsonl \
  --base-url http://127.0.0.1:15100 \
  --runs 64 --max-steps 30 --temperature 1.0 \
  --workers 8 \
  --output outputs/eval/sciworld/gt.jsonl
```

## VisualWebArena task success

The evaluator uses the official VisualWebArena runtime at commit `89f5af29305c3d1e9f97ce4421462060a70c9a03`. Generate per-task configs and deploy the benchmark websites according to that repository's instructions. The paper uses the tier-B prompt, a template-disjoint held-out split, and a 201-task read-only subset.

```bash
python -m eval.vwa.run_eval \
  --vwa-root /path/to/visualwebarena \
  --tasks /path/to/paper_readonly_201.jsonl \
  --base-url http://127.0.0.1:15100 \
  --runs 64 --max-steps 30 --workers 16 \
  --output outputs/eval/vwa/coin.jsonl
```

Multiple model servers can be supplied as a comma-separated `--base-url` value. Infrastructure failures are written to per-worker `.failed` files and excluded from the merged output so they can be rerun.

## Pass@k and coverage

The three task runners emit the same minimal result schema. Compute the paper metrics with:

```bash
python -m eval.metrics \
  --input outputs/eval/alfworld/gt-give.jsonl \
  --k 1 2 4 8 16 32 64 \
  --output outputs/eval/alfworld/gt-give.metrics.json
```

`pass_at_k` is the unbiased task-level estimator. `coverage_tasks` is the number of tasks solved at least once in the first `k` samples; when exactly `k` samples are collected, the two report the same coverage endpoint in percentage and count form.

## Next-observation prediction accuracy

Generate predictions from held-out intermediate JSONL:

```bash
python -m eval.prediction.run_prediction \
  --input /path/to/heldout.jsonl \
  --base-url http://127.0.0.1:15100 \
  --model student --tag gt \
  --limit-per-split 600 --temperature 0 \
  --output outputs/eval/prediction/gt.jsonl
```

ALFWorld uses 600 transitions from each of `valid_seen` and `valid_unseen`. ScienceWorld uses 1,200 transitions from its single held-out split, so pass `--limit-per-split 1200` for that environment.

Judge them with the paper's dynamics-focused prompt. The reported experiments used DeepSeek-V4-Flash at temperature zero, but the endpoint and served model name are explicit arguments:

```bash
python -m eval.prediction.judge \
  --environment alfworld \
  --input outputs/eval/prediction/gt.jsonl \
  --judge-url http://127.0.0.1:8000 \
  --judge-model /path/or/served-name/of/DeepSeek-V4-Flash \
  --output outputs/eval/prediction/gt.judged.jsonl
```
