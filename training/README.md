# Training entry points

The launchers reproduce the paper's training configurations while exposing machine-specific paths through environment variables.

## Shared environment variables

| Variable | Meaning | Default |
|---|---|---|
| `PYTHON_BIN` | Python executable with the training stack | `python` |
| `MODEL_PATH` | Hugging Face model ID or local model path | required |
| `DATA_ROOT` | Root containing processed parquet files | `<repo>/data/processed` |
| `OUTPUT_ROOT` | Checkpoint and runtime-output root | `<repo>/outputs` |
| `N_GPUS` | GPUs used by verl | `8` |
| `USE_WANDB` | Set to `1` to enable Weights & Biases | `0` |
| `WANDB_PROJECT` | Weights & Biases project name | `wm-post-training-audit` |

Hydra overrides may be appended to any command.

## Text GRPO

```bash
bash training/run_text_grpo.sh <environment> <condition> [hydra overrides...]
```

Supported paper runs:

| Environment | Conditions | Input |
|---|---|---|
| `alfworld` | `gt`, `mis`, `coin`, `gt-perm`, `mis-perm` | `data/processed/alfworld/{gt,mis}.parquet` |
| `sciworld` | `gt`, `mis`, `coin`, `mis-perm` | `data/processed/sciworld/{gt,mis}.parquet` |

GT/MIS/PERM runs require `EMBEDDING_MODEL_PATH`. Optional embedding-server controls are `EMBEDDING_PYTHON_BIN`, `EMBEDDING_PORT`, `EMBEDDING_GPU`, `EMBEDDING_MEM_FRACTION`, and `EMBEDDING_STARTUP_TIMEOUT`.

The text runs use batch size 32, eight samples per prompt, response length 512, learning rate `1e-6`, KL coefficient `0.01`, entropy coefficient `0.001`, and temperature `1.0`. ALFWorld uses prompt length 1536; ScienceWorld uses 2560.

## OPSD

```bash
bash training/run_opsd.sh <gt|mis> [hydra overrides...]
```

OPSD uses full-vocabulary generalized JSD with alpha `0.5`, importance-sampling clip `2.0`, a fixed base teacher, batch size 32, and one rollout per prompt. Input parquet files must be built with `build_verl_dataset --tokenizer ...` so `teacher_prompt_ids` are present.

## VisualWebArena COIN

```bash
bash training/run_vwa_coin.sh [hydra overrides...]
```

The VWA run uses Qwen2.5-VL-7B, a Bernoulli(0.5) reward, prompt length 8192, response length 512, batch size 32, eight samples per prompt, a frozen vision tower, KL coefficient `0.01`, entropy coefficient `0.001`, and learning rate `1e-6`.
