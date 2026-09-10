#!/usr/bin/env bash
# Train the fixed-teacher OPSD-GT or OPSD-MIS condition from the paper.
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "gt" && "$1" != "mis" ) ]]; then
  echo "Usage: bash training/run_opsd.sh <gt|mis> [Hydra overrides...]" >&2
  exit 2
fi

condition=$1
shift

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
model_path=${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-7B-Instruct or a local checkpoint}
data_root=${DATA_ROOT:-$repo_root/data/processed}
output_root=${OUTPUT_ROOT:-$repo_root/outputs}
n_gpus=${N_GPUS:-8}
train_file=$data_root/alfworld/$condition.parquet
reward_file=$repo_root/verl/recipe/wm_opsd/zero_reward.py
run_name=${RUN_NAME:-opsd-$condition}
output_dir=${OUTPUT_DIR:-$output_root/$run_name}

if [[ ! -f "$train_file" ]]; then
  echo "Training data not found: $train_file" >&2
  exit 1
fi

mkdir -p "$output_dir"
export PYTHONPATH="$repo_root/verl:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1

logger='["console"]'
if [[ ${USE_WANDB:-0} == "1" ]]; then
  logger='["console","wandb"]'
  export WANDB_DIR=${WANDB_DIR:-$output_root/wandb}
fi

"$python_bin" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  "data.train_files=$train_file" \
  "data.val_files=$train_file" \
  data.train_batch_size=32 \
  data.max_prompt_length=1536 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  "custom_reward_function.path=$reward_file" \
  custom_reward_function.name=compute_score \
  "actor_rollout_ref.model.path=$model_path" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
  actor_rollout_ref.actor.self_distillation.alpha=0.5 \
  actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
  actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0 \
  actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
  actor_rollout_ref.actor.self_distillation.span_mode=next_state \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=1 \
  trainer.critic_warmup=0 \
  "trainer.logger=$logger" \
  "trainer.project_name=${WANDB_PROJECT:-wm-post-training-audit}" \
  "trainer.experiment_name=$run_name" \
  "trainer.n_gpus_per_node=$n_gpus" \
  trainer.nnodes=1 \
  trainer.save_freq=622 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  "trainer.default_local_dir=$output_dir" \
  "$@"
