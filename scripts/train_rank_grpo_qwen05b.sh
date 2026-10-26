#!/usr/bin/env bash
# Stage 2 (Rank-GRPO, on-policy) of ConvRec-R1 for Qwen2.5-0.5B-Instruct, starting from the SFT checkpoint.
#   paper : settings recovered from the released checkpoint-15800/training_args.bin: 4 GPUs x 32 per device x
#           3 grad-accum = 384 completions (48 prompts x 8 generations) per step, 2 epochs (15,958 steps),
#           lr 1e-6 constant, KL beta 1e-3, vLLM colocate (tensor parallel 2, 50% GPU memory), bf16,
#           gradient checkpointing, checkpoint every 200 steps.
#   smoke : 1 GPU, 64 prompts, 10 steps, completions capped at 512 tokens. Only checks that the pipeline runs.
#
# REWARD=exp_inf    Rank-GRPO (exp) with Gamma = infinity (default, the released checkpoint)
# REWARD=log_decay  Rank-GRPO (log), i.e. DCG@k:N
# SFT_CKPT=...      SFT checkpoint to start from (default: the released one, see scripts/download_qwen05b.sh)
#
# Usage (from anywhere, env activated):  bash scripts/train_rank_grpo_qwen05b.sh [smoke|paper]
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"
REWARD="${REWARD:-exp_inf}"
SFT_CKPT="${SFT_CKPT:-results/Qwen/Qwen2.5-0.5B-Instruct/checkpoint-1500}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p logs

[[ -f "$SFT_CKPT/config.json" ]] || { echo "SFT checkpoint not found: $SFT_CKPT (run scripts/download_qwen05b.sh)"; exit 1; }

COMMON=(--train_path processed_datasets/grpo/grpo_dataset --model_name Qwen/Qwen2.5-0.5B-Instruct
        --sft_model_path "$SFT_CKPT" --catalog_path reddit_catalog/gt_catalog.pkl --reward_func "$REWARD"
        --mu 1 --lr 1e-6 --kl_beta 1e-3 --adam_beta1 0.9 --adam_beta2 0.99 --optim paged_adamw_8bit
        --num_generations 8 --max_prompt_length 2048 --use_vllm --vllm_mode colocate
        --seed 3407 --bf16 --gradient_checkpointing --wandb_project rank_grpo)

case "$MODE" in
  paper)
    accelerate launch --config_file configs/qwen25_0.5b_grpo_4gpu.yaml train_rank_grpo.py "${COMMON[@]}" \
      --output_dir "results/grpo/Qwen/Qwen2.5-0.5B-Instruct_${REWARD}_lr1e-06_kl0.001_mu1" \
      --num_train_epochs 2 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
      --gradient_accumulation_steps 3 --max_completion_length 1024 \
      --vllm_gpu_memory_utilization 0.5 --vllm_tensor_parallel_size 2 \
      --save_strategy steps --save_steps 200 --logging_steps 10 \
      2>&1 | tee "logs/rank_grpo_qwen25_0.5b_${REWARD}_paper.txt"
    ;;
  smoke)
    accelerate launch --config_file configs/qwen25_0.5b_1gpu.yaml train_rank_grpo.py "${COMMON[@]}" \
      --output_dir "results/smoke/rank_grpo_${REWARD}" --num_train_samples 64 --max_steps 10 \
      --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 \
      --max_completion_length 512 --vllm_gpu_memory_utilization 0.3 --vllm_tensor_parallel_size 1 \
      --save_strategy no --logging_steps 1 \
      2>&1 | tee "logs/rank_grpo_qwen25_0.5b_${REWARD}_smoke.txt"
    ;;
  *)
    echo "usage: $0 [smoke|paper]"; exit 1
    ;;
esac
