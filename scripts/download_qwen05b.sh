#!/usr/bin/env bash
# Download the released REDDIT-V2 data and Qwen2.5-0.5B checkpoints (public Google Drive links from the
# original README) and unpack them where the training / evaluation scripts expect them:
#   processed_datasets/{sft_dataset,grpo/grpo_dataset}
#   results/Qwen/Qwen2.5-0.5B-Instruct/checkpoint-1500                        (SFT)
#   results/grpo/Qwen/Qwen2.5-0.5B-Instruct_lr1e-06_kl0.001/checkpoint-15800  (Rank-GRPO exp_inf, on-policy)
#
# Usage (from anywhere):  bash scripts/download_qwen05b.sh [--prune-cache]
#   --prune-cache   delete processed_datasets/**/cache-*.arrow (~8.6 GB of tokenization cache, not needed)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p downloads

fetch() {  # fetch <drive_id> <file_name> <expected_bytes>
  local id="$1" out="downloads/$2" size="$3"
  if [[ -f "$out" && "$(stat -c%s "$out")" == "$size" ]]; then echo "already downloaded: $out"; return; fi
  echo "downloading $out ..."
  curl -L --fail -C - -o "$out" "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t"
  [[ "$(stat -c%s "$out")" == "$size" ]] || { echo "size mismatch for $out (expected $size bytes)"; exit 1; }
}

unpack() {  # unpack <zip> <dest_dir>   (skips macOS metadata entries)
  echo "unpacking $1 -> $2"
  mkdir -p "$2"
  python3 - "$1" "$2" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    members = [m for m in z.namelist() if not m.startswith("__MACOSX/") and "/._" not in m]
    z.extractall(sys.argv[2], members)
EOF
}

# 1) Dataset (reuses ./processed_datasets.zip if it was already downloaded to the repo root)
if [[ -f processed_datasets/grpo/grpo_dataset/train/state.json && -f processed_datasets/sft_dataset/test/state.json ]]; then
  echo "dataset already present: processed_datasets/"
else
  if [[ -f processed_datasets.zip && "$(stat -c%s processed_datasets.zip)" == "792594362" ]]; then
    unpack processed_datasets.zip .
  else
    fetch 11tOfUMlVOylkkcnwPqGM_0IuiIeHjLle processed_datasets.zip 792594362
    unpack downloads/processed_datasets.zip .
  fi
fi

# 2) SFT checkpoint-1500
if [[ -f results/Qwen/Qwen2.5-0.5B-Instruct/checkpoint-1500/model.safetensors ]]; then
  echo "SFT checkpoint already present"
else
  fetch 1Br-CMrzgUzJ7_fwFVahCqzJ8TyGZ5Zfz qwen05b_sft_ckpt1500.zip 787893402
  unpack downloads/qwen05b_sft_ckpt1500.zip results/Qwen
fi

# 3) Rank-GRPO (exp_inf) checkpoint-15800
if [[ -f results/grpo/Qwen/Qwen2.5-0.5B-Instruct_lr1e-06_kl0.001/checkpoint-15800/model.safetensors ]]; then
  echo "Rank-GRPO checkpoint already present"
else
  fetch 1VNSkbXGpawfj1amRMhfoiPQHsOKcYJzA qwen05b_rank_grpo_ckpt15800.zip 788757427
  unpack downloads/qwen05b_rank_grpo_ckpt15800.zip results/grpo/Qwen
fi

if [[ "${1:-}" == "--prune-cache" ]]; then
  echo "removing tokenization caches ..."
  find processed_datasets -name 'cache-*.arrow' -delete
fi

echo "Done. The zip files in downloads/ can be deleted."
