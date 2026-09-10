#!/usr/bin/env bash
# Train a text-environment GRPO condition from the paper.
set -euo pipefail

usage() {
  echo "Usage: bash training/run_text_grpo.sh <alfworld|sciworld> <gt|mis|coin|gt-perm|mis-perm> [Hydra overrides...]" >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

environment=$1
condition=$2
shift 2

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
model_path=${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-7B-Instruct or a local checkpoint}
data_root=${DATA_ROOT:-$repo_root/data/processed}
output_root=${OUTPUT_ROOT:-$repo_root/outputs}
n_gpus=${N_GPUS:-8}

case "$environment" in
  alfworld)
    max_prompt_length=1536
    save_freq=622
    ;;
  sciworld)
    max_prompt_length=2560
    save_freq=1000
    if [[ "$condition" == "gt-perm" ]]; then
      echo "ScienceWorld GT-PERM is not a paper experiment." >&2
      exit 2
    fi
    ;;
  *)
    usage
    exit 2
    ;;
esac

adv_estimator=grpo
needs_embedding=1
case "$condition" in
  gt)
    train_file=$data_root/$environment/gt.parquet
    reward_file=$repo_root/verl/recipe/wm_opsd/rwml_reward.py
    ;;
  mis)
    train_file=$data_root/$environment/mis.parquet
    reward_file=$repo_root/verl/recipe/wm_opsd/rwml_reward.py
    ;;
  coin)
    train_file=$data_root/$environment/gt.parquet
    reward_file=$repo_root/verl/recipe/wm_opsd/random_reward.py
    needs_embedding=0
    ;;
  gt-perm)
    train_file=$data_root/$environment/gt.parquet
    reward_file=$repo_root/verl/recipe/wm_opsd/rwml_reward.py
    adv_estimator=grpo_perm
    ;;
  mis-perm)
    train_file=$data_root/$environment/mis.parquet
    reward_file=$repo_root/verl/recipe/wm_opsd/rwml_reward.py
    adv_estimator=grpo_perm
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ ! -f "$train_file" ]]; then
  echo "Training data not found: $train_file" >&2
  exit 1
fi

run_name=${RUN_NAME:-${environment}-${condition}}
output_dir=${OUTPUT_DIR:-$output_root/$run_name}
mkdir -p "$output_dir"

export PYTHONPATH="$repo_root/verl:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1

runtime_overrides=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=1"
)

if [[ "$adv_estimator" == "grpo_perm" ]]; then
  export VERL_USE_EXTERNAL_MODULES=recipe.wm_opsd.rwml_perm_adv
  export RWML_PERM_SEED=${RWML_PERM_SEED:-1}
  runtime_overrides+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=recipe.wm_opsd.rwml_perm_adv"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RWML_PERM_SEED=$RWML_PERM_SEED"
  )
fi

embedding_pid=
if [[ "$needs_embedding" == "1" ]]; then
  embedding_model_path=${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH for GT, MIS, and PERM runs}
  embedding_python=${EMBEDDING_PYTHON_BIN:-$python_bin}
  embedding_port=${EMBEDDING_PORT:-13151}
  embedding_gpu=${EMBEDDING_GPU:-0}
  embedding_mem_fraction=${EMBEDDING_MEM_FRACTION:-0.2}
  embedding_timeout=${EMBEDDING_STARTUP_TIMEOUT:-900}
  embedding_log=$output_dir/embedding-server.log

  CUDA_VISIBLE_DEVICES=$embedding_gpu "$embedding_python" -m sglang.launch_server \
    --model-path "$embedding_model_path" \
    --is-embedding \
    --served-model-name embedding \
    --host 127.0.0.1 \
    --port "$embedding_port" \
    --tp 1 \
    --mem-fraction-static "$embedding_mem_fraction" \
    >"$embedding_log" 2>&1 &
  embedding_pid=$!
  trap '[[ -n "${embedding_pid:-}" ]] && kill "$embedding_pid" 2>/dev/null || true' EXIT

  deadline=$((SECONDS + embedding_timeout))
  until curl -fsS "http://127.0.0.1:$embedding_port/health" >/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "Embedding server did not become ready. See $embedding_log" >&2
      exit 1
    fi
    sleep 5
  done
  runtime_overrides+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.RWML_EMB_URL=http://127.0.0.1:$embedding_port/v1/embeddings"
  )
else
  export RWML_RANDOM_REWARD_P=${RWML_RANDOM_REWARD_P:-0.5}
  export RWML_RANDOM_REWARD_SEED=${RWML_RANDOM_REWARD_SEED:-0}
  runtime_overrides+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.RWML_RANDOM_REWARD_P=$RWML_RANDOM_REWARD_P"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RWML_RANDOM_REWARD_SEED=$RWML_RANDOM_REWARD_SEED"
  )
fi

logger='["console"]'
if [[ ${USE_WANDB:-0} == "1" ]]; then
  logger='["console","wandb"]'
  export WANDB_DIR=${WANDB_DIR:-$output_root/wandb}
fi

"$python_bin" -m verl.trainer.main_ppo \
  "algorithm.adv_estimator=$adv_estimator" \
  "data.train_files=$train_file" \
  "data.val_files=$train_file" \
  data.train_batch_size=32 \
  "data.max_prompt_length=$max_prompt_length" \
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
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
  "${runtime_overrides[@]}" \
  trainer.critic_warmup=0 \
  "trainer.logger=$logger" \
  "trainer.project_name=${WANDB_PROJECT:-wm-post-training-audit}" \
  "trainer.experiment_name=$run_name" \
  "trainer.n_gpus_per_node=$n_gpus" \
  trainer.nnodes=1 \
  "trainer.save_freq=$save_freq" \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  "trainer.default_local_dir=$output_dir" \
  "$@"
