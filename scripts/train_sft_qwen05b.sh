#!/usr/bin/env bash
# Stage 1 (SFT) of ConvRec-R1 for Qwen2.5-0.5B-Instruct on the released REDDIT-V2 SFT data.
#   paper : settings of the released checkpoint (its training_args.bin): 2 GPUs x 12 per device x 8 grad-accum
#           = 192 sequences/step, lr 5e-5 cosine, 5% warm-up, 10 epochs (4,990 steps; the paper continues RL from
#           checkpoint-1500), bf16, gradient checkpointing, evaluation every 10 steps on the full validation set.
#   smoke : 1 GPU, 200 rows, 20 steps. Only checks that the pipeline runs (e.g. on an 8 GB laptop GPU).
#
# Usage (from anywhere, env activated):  bash scripts/train_sft_qwen05b.sh [smoke|paper]
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p logs

COMMON=(--dataset_path processed_datasets/sft_dataset --model_name Qwen/Qwen2.5-0.5B-Instruct
        --learning_rate 5e-5 --warmup_ratio 0.05 --lr_scheduler_type cosine --optim paged_adamw_8bit
        --max_length 1024 --bf16 --gradient_checkpointing --seed 3407)

case "$MODE" in
  paper)
    # writes results/Qwen/Qwen2.5-0.5B-Instruct/checkpoint-* (same place as the released SFT checkpoint)
    accelerate launch --config_file configs/qwen25_0.5b_sft.yaml train_sft.py "${COMMON[@]}" \
      --num_train_epochs 10 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
      --gradient_accumulation_steps 8 --save_strategy steps --save_steps 100 \
      --eval_strategy steps --eval_steps 10 --logging_steps 10 \
      2>&1 | tee logs/sft_qwen25_0.5b_paper.txt
    ;;
  smoke)
    accelerate launch --config_file configs/qwen25_0.5b_1gpu.yaml train_sft.py "${COMMON[@]}" \
      --output_dir results/smoke/sft --num_train_samples 200 --max_steps 20 \
      --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 2 \
      --save_strategy no --eval_strategy steps --eval_steps 10 --logging_steps 1 --dataset_num_proc 4 \
      2>&1 | tee logs/sft_qwen25_0.5b_smoke.txt
    ;;
  *)
    echo "usage: $0 [smoke|paper]"; exit 1
    ;;
esac
