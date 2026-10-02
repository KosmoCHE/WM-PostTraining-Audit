<h1 align="center">World-Model Post-Training Audit</h1>

<p align="center">
  <em>Does Learning to Predict the World Help Agents Act?</em><br>
  <em>Auditing World-Model Post-Training</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.33335"><img src="https://img.shields.io/badge/arXiv-2609.33335-b31b1b.svg?style=for-the-badge" alt="arXiv: 2609.33335"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg?style=for-the-badge" alt="License: Apache-2.0"></a>
</p>
<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%3E%3D3.10-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python >= 3.10"></a>
  <a href="verl/VENDOR_SHA.txt"><img src="https://img.shields.io/badge/verl-v0.7.1-6f42c1.svg?style=flat-square" alt="verl v0.7.1"></a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> &bull;
  <a href="#highlights">Highlights</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#experiments">Experiments</a> &bull;
  <a href="#citation">Citation</a>
</p>

---

## Introduction

This repository contains the training, offline data-preparation, and formal evaluation code for [*Does Learning to Predict the World Help Agents Act? Auditing World-Model Post-Training*](https://arxiv.org/abs/2609.33335).

The audit asks whether post-training improvements come from learning to predict the environment, or from other effects introduced by the optimization process. It compares ground-truth next observations with in-distribution mismatched targets and with independent random rewards across interactive text environments and VisualWebArena.

The release is intentionally focused on the code required to prepare training data, train the reported conditions, and reproduce the formal task and next-observation metrics. Mechanism analysis, figure generation, experiment logs, model checkpoints, and datasets are not included.

## Highlights

| | |
|---|---|
| **Controlled audit conditions** | Ground-truth targets, mismatched targets, response-level advantage permutations, and independent random rewards isolate different sources of post-training gains. |
| **Multiple environments** | ALFWorld, ScienceWorld, and VisualWebArena cover text-only and multimodal interaction. |
| **Reproducible pipeline** | Offline trajectory transforms, canonical training launchers, and standardized JSONL evaluation outputs are included. |
| **Formal metrics** | The release computes task success, pass@k, coverage, next-observation accuracy, and dynamics-focused judge scores. |

## Quick Start

### Installation

The reported runs used eight H100 GPUs. Create an environment compatible with CUDA, PyTorch, vLLM, and `verl` v0.7.1, then install the vendored framework and preprocessing dependencies:

```bash
python -m pip install -e ./verl
python -m pip install -r requirements-data.txt
python -m pip install -r requirements-eval.txt
```

The embedding-reward experiments additionally require an OpenAI-compatible embedding endpoint. The launch script can start `Qwen3-Embedding-8B` through SGLang when `EMBEDDING_MODEL_PATH` is set.

### Prepare data

The release starts from trajectory dumps rather than live environment collection. See [data_pipeline/README.md](data_pipeline/README.md) for reproducible commands and [docs/data-formats.md](docs/data-formats.md) for the expected trajectory schemas.

Prepared files follow this convention:

```text
data/processed/alfworld/{gt,mis}.parquet
data/processed/sciworld/{gt,mis}.parquet
data/processed/vwa/coin.parquet
```

Data, trajectories, checkpoints, and runtime logs are ignored by Git.

### Train

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

### Evaluate

The evaluation runners target OpenAI-compatible model servers and emit a shared task-level JSONL schema. ALFWorld supports list and no-list regimes, ScienceWorld uses the fixed AgentGym item list, and VisualWebArena uses the paper's tier-B prompt and pinned upstream runtime. Next-observation prediction and the dynamics-focused LLM judge are also included.

See [eval/README.md](eval/README.md) for environment setup and commands. The shortest metric command is:

```bash
python -m eval.metrics --input outputs/eval/<run>.jsonl --k 1 8 16 64
```

## Experiments

| Setting | Released training conditions |
|---|---|
| ALFWorld · Qwen2.5-7B-Instruct · GRPO | GT, MIS, COIN, GT-PERM, MIS-PERM |
| ScienceWorld · Qwen2.5-7B-Instruct · GRPO | GT, MIS, COIN, MIS-PERM |
| ALFWorld · Qwen2.5-7B-Instruct · OPSD | OPSD-GT, OPSD-MIS |
| VisualWebArena · Qwen2.5-VL-7B-Instruct · GRPO | COIN |

`GT` uses ground-truth next observations, `MIS` uses a fixed derangement of next observations, `COIN` uses an i.i.d. Bernoulli(0.5) reward, and `*-PERM` permutes response-level advantages within each GRPO group.

## Repository Structure

```text
data_pipeline/   Offline trajectory-to-training-data transforms
training/        Canonical launchers for the paper experiments
eval/            Task success, pass@k, and prediction-accuracy evaluators
verl/            Vendored verl v0.7.1 plus the OPSD and audit extensions
docs/            Input schemas and data-preparation notes
tests/           CPU-only checks for preprocessing and evaluation helpers
```

## Scope and Provenance

The vendored `verl/` tree is based on [`volcengine/verl`](https://github.com/volcengine/verl) v0.7.1 at commit `bec9ef7`. Project-specific changes implement privileged on-policy self-distillation, the observation-matching reward, within-group advantage permutation, random reward, and multimodal reference deduplication. Upstream provenance is recorded in [verl/VENDOR_SHA.txt](verl/VENDOR_SHA.txt).

The formal evaluation harnesses pin the VisualWebArena runtime to commit `89f5af29305c3d1e9f97ce4421462060a70c9a03`. Refer to [eval/README.md](eval/README.md) for environment-specific setup and evaluation assumptions.

## Citation

```bibtex
@misc{che2026learning,
  title         = {Does Learning to Predict the World Help Agents Act? Auditing World-Model Post-Training},
  author        = {Xinyu Che and Hang Yan and Yanchen Liu and Haochen Liu and Ruifeng Li and Anran Shi and Heng Wang and Jun Liu},
  year          = {2026},
  eprint        = {2609.33335},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.33335},
}
```

## License

This repository is released under the [Apache License 2.0](LICENSE). The vendored framework retains its upstream license and notice files.
