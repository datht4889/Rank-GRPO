#!/usr/bin/env bash
# Create a Python 3.10 environment with the versions pinned in the original README, then install
# the Rank-GRPO trainer into TRL. Linux / WSL2 only (vLLM has no native Windows build).
#
# Usage (from anywhere):  bash scripts/setup_env.sh [venv_dir]      (default venv_dir: .venv)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-.venv}"
command -v uv >/dev/null || { echo "uv not found, install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

uv venv --python 3.10 "$VENV"
source "$VENV/bin/activate"

# Same pins as the original README; torch comes from the CUDA 12.6 index first.
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install vllm==0.10.0 trl==0.21.0 transformers==4.55.4 accelerate==1.10.0
# DeepSpeed builds from source and imports torch at build time, hence --no-build-isolation.
uv pip install setuptools wheel ninja
uv pip install --no-build-isolation "deepspeed>=0.16,<0.18"
uv pip install bitsandbytes
# datasets>=4 is required to read processed_datasets/grpo (it was saved with the `List` feature type).
uv pip install "datasets>=4.0,<5" wandb editdistance regex seaborn matplotlib

bash INSTALL.SH
python -c "import torch, trl, vllm; from trl import RankGRPOTrainer; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| trl', trl.__version__, '| vllm', vllm.__version__)"
echo "Done. Activate with: source $VENV/bin/activate"
