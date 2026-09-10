#!/usr/bin/env bash
# Train the VisualWebArena COIN condition from the paper.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
model_path=${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-VL-7B-Instruct or a local checkpoint}
data_root=${DATA_ROOT:-$repo_root/data/processed}
output_root=${OUTPUT_ROOT:-$repo_root/outputs}
n_gpus=${N_GPUS:-8}
train_file=${TRAIN_FILE:-$data_root/vwa/coin.parquet}
reward_file=$repo_root/verl/recipe/wm_opsd/random_reward.py
run_name=${RUN_NAME:-vwa-coin}
output_dir=${OUTPUT_DIR:-$output_root/$run_name}
ray_spill_dir=${RAY_SPILL_DIR:-$output_root/ray-spill}
ray_object_store_memory=${RAY_OBJECT_STORE_MEMORY:-100000000000}

if [[ ! -f "$train_file" ]]; then
  echo "Training data not found: $train_file" >&2
  exit 1
fi

mkdir -p "$output_dir" "$ray_spill_dir"
export PYTHONPATH="$repo_root/verl:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export RWML_RANDOM_REWARD_P=${RWML_RANDOM_REWARD_P:-0.5}
export RWML_RANDOM_REWARD_SEED=${RWML_RANDOM_REWARD_SEED:-0}

logger='["console"]'
if [[ ${USE_WANDB:-0} == "1" ]]; then
  logger='["console","wandb"]'
  export WANDB_DIR=${WANDB_DIR:-$output_root/wandb}
fi

"$python_bin" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  "data.train_files=$train_file" \
  "data.val_files=$train_file" \
  data.train_batch_size=32 \
  data.max_prompt_length=8192 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.image_key=images \
  "custom_reward_function.path=$reward_file" \
  custom_reward_function.name=compute_score \
  "actor_rollout_ref.model.path=$model_path" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.freeze_vision_tower=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
  actor_rollout_ref.rollout.agent.num_workers=32 \
  trainer.balance_batch=False \
  "+ray_kwargs.ray_init.object_store_memory=$ray_object_store_memory" \
  "+ray_kwargs.ray_init.object_spilling_directory=$ray_spill_dir" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=1 \
  "+ray_kwargs.ray_init.runtime_env.env_vars.RWML_RANDOM_REWARD_P=$RWML_RANDOM_REWARD_P" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.RWML_RANDOM_REWARD_SEED=$RWML_RANDOM_REWARD_SEED" \
  trainer.critic_warmup=0 \
  "trainer.logger=$logger" \
  "trainer.project_name=${WANDB_PROJECT:-wm-post-training-audit}" \
  "trainer.experiment_name=$run_name" \
  "trainer.n_gpus_per_node=$n_gpus" \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  "trainer.default_local_dir=$output_dir" \
  "$@"
