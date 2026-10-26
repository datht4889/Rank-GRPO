#!/usr/bin/env bash
# REDDIT-V2 test-set evaluation of Qwen2.5-0.5B-Instruct: zero-shot, SFT (checkpoint-1500) and
# SFT + Rank-GRPO (exp) (checkpoint-15800). Greedy decoding with vLLM, Recall/NDCG@{5,10,15,20}.
#
# Usage (from anywhere, env activated):  bash scripts/eval_qwen05b.sh [all|zero_shot|sft|rank_grpo]
#   GPU_UTIL=0.8   fraction of GPU memory given to vLLM (the 0.5B model fits in 8 GB)
#   TP=1           vLLM tensor-parallel size
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH="${1:-all}"
GPU_UTIL="${GPU_UTIL:-0.8}"
TP="${TP:-1}"
mkdir -p logs

COMMON=(--model_name Qwen/Qwen2.5-0.5B-Instruct
        --dataset_path processed_datasets/sft_dataset
        --catalog_path reddit_catalog/gt_catalog.pkl
        --gpu_memory_utilization "$GPU_UTIL" --tensor_parallel_size "$TP")

run() {  # run <name> <model args...>
  local name="$1"; shift
  python evaluate/eval_grpo_test.py "${COMMON[@]}" --save_dir "results_eval/$name" "$@" 2>&1 | tee "logs/eval_test_$name.txt"
}

if [[ "$WHICH" == all || "$WHICH" == zero_shot ]]; then
  run zero_shot --model_path Qwen/Qwen2.5-0.5B-Instruct
fi
if [[ "$WHICH" == all || "$WHICH" == sft ]]; then
  run sft_ckpt1500 --model_root results/Qwen/Qwen2.5-0.5B-Instruct --checkpoint 1500
fi
if [[ "$WHICH" == all || "$WHICH" == rank_grpo ]]; then
  run rank_grpo_ckpt15800 --model_root results/grpo/Qwen/Qwen2.5-0.5B-Instruct_lr1e-06_kl0.001 --checkpoint 15800
fi

# Side-by-side with the paper (Table 1)
python3 - <<'EOF'
import json, os
paper = {  # R@5 R@10 R@15 R@20 / N@5 N@10 N@15 N@20
    "zero_shot":           ([0.0021, 0.0021, 0.0021, 0.0021], [0.0023, 0.0022, 0.0021, 0.0021]),
    "sft_ckpt1500":        ([0.0642, 0.1027, 0.1212, 0.1308], [0.0502, 0.0625, 0.0678, 0.0704]),
    "rank_grpo_ckpt15800": ([0.0946, 0.1468, 0.1813, 0.2047], [0.0744, 0.0915, 0.1016, 0.1079]),
}
print(f"\n{'run':22s} {'':6s} " + " ".join(f"{m}@{k}".ljust(6) for m in ("R", "N") for k in (5, 10, 15, 20)))
for name, (pr, pn) in paper.items():
    path = os.path.join("results_eval", name, "analysis_state.json")
    if not os.path.exists(path):
        continue
    e = json.load(open(path))["log_history"][0]
    ours = [e[f"eval_recall@{k}"] for k in (5, 10, 15, 20)] + [e[f"eval_ndcg@{k}"] for k in (5, 10, 15, 20)]
    print(f"{name:22s} {'ours':6s} " + " ".join(f"{v:.4f}" for v in ours))
    print(f"{'':22s} {'paper':6s} " + " ".join(f"{v:.4f}" for v in pr + pn))
EOF
