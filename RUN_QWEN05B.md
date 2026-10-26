# Chạy lại ConvRec-R1 / Rank-GRPO với Qwen2.5-0.5B-Instruct (REDDIT-V2)

Branch `qwen05b-repro`. Code train/eval gốc dùng chung cho mọi backbone nhưng không chạy được ở bản release
(lỗi import, thiếu hàm eval). Branch này vá các lỗi đó và thêm script riêng cho Qwen2.5-0.5B. Thuật toán
Rank-GRPO (`libs/trl/rank_grpo_trainer.py`, `libs/reward_funcs.py`) giữ nguyên để tái hiện đúng bản gốc.

## Cấu hình thật của checkpoint đã release

Đọc từ `training_args.bin` và `trainer_state.json` trong hai checkpoint trên Google Drive:

| | SFT (checkpoint-1500) | Rank-GRPO exp (checkpoint-15800) |
|---|---|---|
| GPU | 2 | 4 |
| Batch mỗi step | 12/GPU × 8 grad-accum × 2 = 192 mẫu | 32/GPU × 3 grad-accum × 4 = 384 completion (48 prompt × 8) |
| Learning rate | 5e-5, cosine, warm-up 5% | 1e-6, constant |
| Độ dài | 10 epoch (4,990 step), RL bắt đầu từ step 1500 | 2 epoch (15,958 step), checkpoint 15800 |
| Khác | max_length 1024, bf16, gradient checkpointing, paged_adamw_8bit | KL 1e-3, G = 8, temperature 1.0, top-p 1.0, vLLM colocate (TP 2, 50% bộ nhớ), loss `bnpo` |

Lưu ý: checkpoint RL được train với `epsilon = 0.2`, khác giá trị 0.06/0.08 trong paper và trong
`train_rank_grpo.py`. Vì chạy on-policy (μ = 1) nên importance ratio luôn bằng 1, giá trị clip không ảnh hưởng.

## Kaggle: một script duy nhất

[`run_full.sh`](run_full.sh) chạy các stage `setup data eval sft eval_sft rl eval_rl` (mặc định:
`setup data eval rl eval_rl`). Code lấy từ repo `github.com/datht4889/Rank-GRPO` (biến `REPO_URL`, `REPO_REF`,
`GITHUB_TOKEN` nếu repo private): sửa code ở máy, push lên, rồi chạy lại trên Kaggle. Tự nhận GPU: T4 dùng fp16
và vLLM `half` (engine V0), GPU từ Ampere trở lên dùng bf16. Dữ liệu và checkpoint lấy từ dataset Kaggle chỉ đọc
`/kaggle/input/datasets/datoggg/rankgrpo_data` (thư mục hoặc zip, tên tùy ý), không có thì tải từ Google Drive.

Cell trên Kaggle (GPU T4 x2, Internet On, đã gắn dataset dữ liệu):

```
!git clone -q https://github.com/datht4889/Rank-GRPO.git /tmp/rankgrpo/Rank-GRPO 2>/dev/null || git -C /tmp/rankgrpo/Rank-GRPO pull -q
!bash /tmp/rankgrpo/Rank-GRPO/run_full.sh setup data eval
```

## Các bước (Linux hoặc WSL2)

```bash
# 0. (khuyến nghị) chạy trong file system của WSL cho nhanh, ví dụ: cp -r /mnt/c/.../Rank-GRPO ~/ && cd ~/Rank-GRPO
# 1. Môi trường: Python 3.10, torch 2.7.1 (cu126), vLLM 0.10.0, TRL 0.21.0, datasets 4.x, rồi cài Rank-GRPO vào TRL
bash scripts/setup_env.sh && source .venv/bin/activate

# 2. Dữ liệu + checkpoint Qwen (bỏ qua phần đã có; --prune-cache xóa ~8.6 GB cache tokenize)
bash scripts/download_qwen05b.sh --prune-cache

# 3. Eval trên test: zero-shot, SFT, Rank-GRPO; cuối cùng in bảng so với Bảng 1 của paper
bash scripts/eval_qwen05b.sh            # hoặc: zero_shot | sft | rank_grpo

# 4. Chạy thử training trên 1 GPU (chỉ kiểm tra pipeline, không để so số liệu)
bash scripts/train_sft_qwen05b.sh smoke
bash scripts/train_rank_grpo_qwen05b.sh smoke

# 5. Train đúng cấu hình paper (SFT: 2 GPU, RL: 4 GPU)
bash scripts/train_sft_qwen05b.sh paper
bash scripts/train_rank_grpo_qwen05b.sh paper                         # Rank-GRPO (exp)
REWARD=log_decay bash scripts/train_rank_grpo_qwen05b.sh paper        # Rank-GRPO (log)

# 6. Đường cong validation theo checkpoint (cần các checkpoint trung gian của lần train của bạn)
python evaluate/eval_grpo_val.py --model_root results/grpo/Qwen/Qwen2.5-0.5B-Instruct_exp_inf_lr1e-06_kl0.001_mu1 \
  --init_model results/Qwen/Qwen2.5-0.5B-Instruct/checkpoint-1500 \
  --dataset_path processed_datasets/sft_dataset --catalog_path reddit_catalog/gt_catalog.pkl
```

Số tham chiếu (paper, Bảng 1, R@20 / N@20): zero-shot 0.0021 / 0.0021, SFT 0.1308 / 0.0704,
SFT + Rank-GRPO (exp) 0.2047 / 0.1079.

## Những gì đã sửa

- `libs/logs.py`: bỏ dòng tự import chính nó (gây `ImportError` khi chạy `train_rank_grpo.py`).
- `libs/metrics.py`: thêm `evaluate_direct_match_truncate`, hàm mà mọi script eval import nhưng repo không có.
  Đây là bản dựng lại: giữ 20 item đầu, rồi loại item đã nhắc và item ngoài catalog như `evaluate_direct_match`.
- `train_rank_grpo.py`, `train_sft.py`: thêm `--output_dir`, `--max_steps`, `--num_train_samples`, `--fp16`,
  `--save_total_limit`, `--save_only_model` (và `--sft_model_path` cho RL, `--stop_at_step` cho SFT: dừng ở step 1500
  mà không rút ngắn lịch cosine 10 epoch như paper); mặc định catalog là `reddit_catalog/gt_catalog.pkl`;
  bỏ import thừa trong `train_sft.py`.
- `libs/trl/rank_grpo_trainer.py`: vLLM colocate nhận `dtype` từ biến môi trường `RANK_GRPO_VLLM_DTYPE`
  (cần `half` trên T4). Không đổi gì trong thuật toán.
- `evaluate/*.py`: sửa import sai tên, đường dẫn `libs` không còn phụ thuộc thư mục đang đứng; eval test nhận
  `--model_path` (dùng cho zero-shot); eval validation chỉ lấy các checkpoint có trên đĩa (chạy được với checkpoint release);
  thêm `--dtype` cho vLLM.
- `configs/qwen25_0.5b_grpo_4gpu.yaml`, `configs/qwen25_0.5b_1gpu.yaml`: config accelerate mới.
  `gradient_accumulation_steps` trong yaml phải bằng tham số truyền cho script (các script trong `scripts/` đã khớp).
- `.gitattributes` giữ LF cho file `.sh` (checkout trên Windows làm `INSTALL.SH` thành CRLF và chạy lỗi trong WSL),
  `.gitignore` cho dữ liệu, checkpoint và output.

## Giới hạn đã biết

- Chế độ `smoke` trên GPU 8 GB chưa được kiểm chứng: vLLM, policy, reference model và optimizer có thể không vừa
  bộ nhớ. DeepSpeed đẩy tham số và optimizer sang CPU, nên nên tăng RAM cho WSL (ví dụ `memory=12GB` trong
  `%UserProfile%\.wslconfig`).
- Phiên bản DeepSpeed không được README gốc ghi rõ; script dùng `>=0.16,<0.18`.
- Không có code cho Stage 1 (Remap-Reflect-Adjust), baseline GRPO/GSPO/DPO, hay checkpoint trung gian.
