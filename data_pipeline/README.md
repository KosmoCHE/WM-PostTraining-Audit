# Offline data pipeline

The pipeline begins with trajectory dumps. Live rollout collection is deliberately outside this release because it depends on benchmark deployments and evaluation infrastructure.

## Text environments

ALFWorld and ScienceWorld share the same intermediate format and downstream transforms.

### ALFWorld

```bash
python -m data_pipeline.alfworld.build_wm_samples \
  --in trajectories/alfworld/train.jsonl \
  --out work/alfworld/all.jsonl

python -m data_pipeline.shared.dedup_wm_samples \
  --in work/alfworld/all.jsonl \
  --out work/alfworld/dedup.jsonl

python -m data_pipeline.shared.subsample_wm_samples \
  --in work/alfworld/dedup.jsonl \
  --out work/alfworld/gt.jsonl \
  --n 20000 --seed 42
```

### ScienceWorld

The paper excludes AgentGym test items from training. `--exclude-item-ids` accepts a JSONL file whose `item_id` values have the form `sciworld_<index>`.

```bash
python -m data_pipeline.sciworld.build_wm_samples \
  --in trajectories/sciworld/train.jsonl \
  --exclude-item-ids splits/agentgym_test.jsonl \
  --out work/sciworld/all.jsonl

python -m data_pipeline.shared.dedup_wm_samples \
  --in work/sciworld/all.jsonl \
  --out work/sciworld/dedup.jsonl

python -m data_pipeline.shared.subsample_wm_samples \
  --in work/sciworld/dedup.jsonl \
  --out work/sciworld/gt.jsonl \
  --n 20000 --seed 42
```

### Construct MIS and verl parquet files

Run the following for each text environment. The fixed derangement preserves the target-observation multiset while ensuring that no sample retains its original target.

```bash
ENV_NAME=alfworld  # or sciworld

python -m data_pipeline.shared.build_errframe_dataset \
  --in work/$ENV_NAME/gt.jsonl \
  --out work/$ENV_NAME/mis.jsonl \
  --seed 42

python -m data_pipeline.shared.build_verl_dataset \
  --in work/$ENV_NAME/gt.jsonl \
  --out data/processed/$ENV_NAME/gt.parquet \
  --data-source wm_${ENV_NAME} \
  --tokenizer Qwen/Qwen2.5-7B-Instruct

python -m data_pipeline.shared.build_verl_dataset \
  --in work/$ENV_NAME/mis.jsonl \
  --out data/processed/$ENV_NAME/mis.parquet \
  --data-source wm_${ENV_NAME}_mis \
  --tokenizer Qwen/Qwen2.5-7B-Instruct
```

Passing `--tokenizer` stores the privileged teacher prefix required by OPSD. The same parquet files remain valid for GRPO.

## VisualWebArena

The paper uses a template-disjoint train/evaluation split and builds 4,478 transition samples from the training-side rollout dumps.

```bash
python -m data_pipeline.vwa.split_templates \
  --config-dir /path/to/visualwebarena/config_files/vwa \
  --out-dir data/splits/vwa \
  --seed 42 --pool-size 200

python -m data_pipeline.vwa.build_wm_samples \
  --dump-root trajectories/vwa \
  --config-root /path/to/visualwebarena/config_files/vwa \
  --runs collect_B_r0 collect_B_r1 collect_B_r2 \
  --out data/processed/vwa/coin.parquet
```

`--config-root` must contain generated per-task files at `test_<site>/<task_id>.json`. The aggregated files used by `split_templates` may be named either `test_<site>.json` or `test_<site>.raw.json`.
