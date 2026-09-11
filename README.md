# World-Model Post-Training Audit

Training, offline data-preparation, and formal evaluation code for *Does Learning to Predict the World Help LLM Agents Act? Auditing World-Model Post-Training*.

This release intentionally contains only the code needed to prepare training data, train the reported conditions, and reproduce the formal task and next-observation metrics. Mechanism analysis, figure generation, experiment logs, model checkpoints, and datasets are not included.

## Included experiments

| Setting | Released training conditions |
|---|---|
| ALFWorld · Qwen2.5-7B-Instruct · GRPO | GT, MIS, COIN, GT-PERM, MIS-PERM |
| ScienceWorld · Qwen2.5-7B-Instruct · GRPO | GT, MIS, COIN, MIS-PERM |
| ALFWorld · Qwen2.5-7B-Instruct · OPSD | OPSD-GT, OPSD-MIS |
| VisualWebArena · Qwen2.5-VL-7B-Instruct · GRPO | COIN |

`GT` uses ground-truth next observations, `MIS` uses a fixed derangement of next observations, `COIN` uses an i.i.d. Bernoulli(0.5) reward, and `*-PERM` permutes response-level advantages within each GRPO group.

## Repository layout

```text
data_pipeline/   Offline trajectory-to-training-data transforms
training/        Canonical launchers for the paper experiments
eval/            Task success, pass@k, and prediction-accuracy evaluators
verl/            Vendored verl v0.7.1 plus the OPSD and audit extensions
docs/            Input schemas and data-preparation notes
tests/           CPU-only checks for preprocessing and evaluation helpers
```

## Installation

The reported runs used eight H100 GPUs. Create an environment compatible with CUDA, PyTorch, vLLM, and verl v0.7.1, then install the vendored framework and preprocessing dependencies:

```bash
python -m pip install -e ./verl
python -m pip install -r requirements-data.txt
python -m pip install -r requirements-eval.txt
```

The embedding-reward experiments additionally require an OpenAI-compatible embedding endpoint. The launch script can start Qwen3-Embedding-8B through SGLang when `EMBEDDING_MODEL_PATH` is set.

## Prepare data

The release starts from trajectory dumps rather than live environment collection. See [data_pipeline/README.md](data_pipeline/README.md) for reproducible commands and [docs/data-formats.md](docs/data-formats.md) for the expected trajectory schemas.

Prepared files follow this convention:

```text
data/processed/alfworld/{gt,mis}.parquet
data/processed/sciworld/{gt,mis}.parquet
data/processed/vwa/coin.parquet
```

Data, trajectories, checkpoints, and runtime logs are ignored by Git.

## Train

Examples:

```bash
# ALFWorld GT with observation-matching reward
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
EMBEDDING_MODEL_PATH=Qwen/Qwen3-Embedding-8B \
bash training/run_text_grpo.sh alfworld gt

# ScienceWorld COIN
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
bash training/run_text_grpo.sh sciworld coin

# ALFWorld OPSD-MIS
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
bash training/run_opsd.sh mis

# VisualWebArena COIN
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
bash training/run_vwa_coin.sh
```

All paths and runtime choices can be overridden with environment variables. See [training/README.md](training/README.md) for the full mapping and defaults.

## Evaluate

The evaluation runners target OpenAI-compatible model servers and emit a shared task-level JSONL schema. ALFWorld supports list and no-list regimes, ScienceWorld uses the fixed AgentGym item list, and VisualWebArena uses the paper's tier-B prompt and pinned upstream runtime. Next-observation prediction and the dynamics-focused LLM judge are also included.

See [eval/README.md](eval/README.md) for environment setup and commands. The shortest metric command is:

```bash
python -m eval.metrics --input outputs/eval/<run>.jsonl --k 1 8 16 64
```

## Scope and provenance

The vendored `verl/` tree is based on `volcengine/verl` v0.7.1 at commit `bec9ef7`; project-specific changes implement privileged on-policy self-distillation, the observation-matching reward, within-group advantage permutation, random reward, and multimodal reference deduplication. Upstream provenance is recorded in [verl/VENDOR_SHA.txt](verl/VENDOR_SHA.txt).

This repository is licensed under Apache-2.0. The vendored framework retains its upstream license and notice files.
