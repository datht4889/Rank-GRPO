#!/usr/bin/env bash
# =====================================================================================================
# run_full.sh: reproduce ConvRec-R1 / Rank-GRPO (ICLR 2026) with Qwen2.5-0.5B-Instruct on REDDIT-V2,
# in one self-contained script. Made for Kaggle notebooks (Accelerator: "GPU T4 x2", Internet: On),
# also works on any Linux machine with NVIDIA GPUs (compute capability >= 7.0).
#
# What it does: uses the fixed code of github.com/datht4889/Rank-GRPO (the authors' release does not run
# as-is), installs the pinned environment, gets the released data/checkpoints and runs the selected stages:
#
#   setup     code checkout + Python 3.10 venv (uv) with the README pins, Rank-GRPO trainer installed into TRL
#   data      REDDIT-V2 processed data + released Qwen SFT (ckpt-1500) and Rank-GRPO (ckpt-15800), copied
#             from the read-only INPUT_ROOT (default /kaggle/input/datasets/datoggg/rankgrpo_data, folders or
#             zip files with any names), otherwise downloaded from Google Drive
#   eval      test-set eval of zero-shot / released SFT / released Rank-GRPO, compared with paper Table 1
#   noise     responder-noise diagnostic of the SFT policy (checks the ReMaRk claim; scripts/responder_noise.py)
#   sft       SFT from the base model (paper: checkpoint-1500 of a 10-epoch cosine run), time-budgeted
#   eval_sft  test-set eval of the own SFT checkpoint
#   rl        Rank-GRPO from the SFT checkpoint (own one if `sft` ran, else the released one), time-budgeted
#   eval_rl   test-set eval of the latest own Rank-GRPO checkpoint
#
# Code: if this file sits in a writable checkout of the repo, that checkout is used as-is (no git calls).
# Otherwise the repo is cloned / updated into $WORK/Rank-GRPO (REPO_URL, REPO_REF; GITHUB_TOKEN for a
# private repo). Edit code locally, push, then re-run on Kaggle.
#
# Usage
#   bash run_full.sh                         # default: setup data eval rl eval_rl
#   bash run_full.sh setup data eval         # only re-evaluate the released checkpoints (cheapest check)
#   RL_HOURS=9 REWARD=log_decay bash run_full.sh setup data rl eval_rl
#   bash run_full.sh setup data sft eval_sft # SFT stage alone
#
# Kaggle: New Notebook > Settings: Accelerator "GPU T4 x2", Internet "On" (needs a verified phone);
#   attach the data dataset (/kaggle/input/datasets/datoggg/rankgrpo_data), then in a cell:
#     !git clone -q https://github.com/datht4889/Rank-GRPO.git /tmp/rankgrpo/Rank-GRPO 2>/dev/null || git -C /tmp/rankgrpo/Rank-GRPO pull -q
#     !bash /tmp/rankgrpo/Rank-GRPO/run_full.sh setup data eval
#   For runs longer than a few hours use "Save Version > Save & Run All (Commit)" so the session does not
#   stop when the browser is idle.
#   Writes: big files in WORK=/tmp/rankgrpo (not kept), results in /kaggle/working/rankgrpo_outputs:
#   summary.txt, logs/, results_eval/, checkpoints/ (kept as the notebook output, 20 GB limit).
#
# Kaggle limits: 12 h per session, ~30 GPU h per week, 20 GB saved in /kaggle/working. The full paper RL
# run (15,958 steps x 384 completions on 4 H100) does not fit; the rl stage trains for RL_HOURS with the
# paper's per-step batch and keeps the last checkpoints, so expect only partial gains over SFT.
# =====================================================================================================
set -euo pipefail

# ----------------------------------------- CONFIG (env vars) -----------------------------------------
WORK="${WORK:-/tmp/rankgrpo}"                 # big files: venv, repo, data, checkpoints (not saved by Kaggle)
if [[ -z "${OUT:-}" ]]; then
  if [[ -d /kaggle/working ]]; then OUT=/kaggle/working/rankgrpo_outputs; else OUT="$WORK/outputs"; fi
fi
# Read-only input searched for the data / checkpoints (extracted folders or zip files, any names). Nothing is
# ever written there: everything is copied into $WORK first.
if [[ -z "${INPUT_ROOT:-}" ]]; then
  if [[ -d /kaggle/input/datasets/datoggg/rankgrpo_data ]]; then INPUT_ROOT=/kaggle/input/datasets/datoggg/rankgrpo_data
  else INPUT_ROOT=/kaggle/input; fi
fi
REPO_URL="${REPO_URL:-https://github.com/datht4889/Rank-GRPO.git}"
REPO_REF="${REPO_REF:-main}"                  # branch / tag / commit to run
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

PRECISION="${PRECISION:-auto}"                # auto (bf16 if supported, else fp16) | bf16 | fp16 | fp32
VLLM_DTYPE="${VLLM_DTYPE:-}"                  # empty = auto (half on GPUs without bf16, e.g. T4)
DS_ZERO="${DS_ZERO:-3}"                       # DeepSpeed ZeRO stage: 3 (authors) | 2 (faster for a 0.5B model)
DS_OFFLOAD_OPT="${DS_OFFLOAD_OPT:-cpu}"       # optimizer offload: cpu (authors; accelerate swaps in DeepSpeed CPU Adam) | none
DS_OFFLOAD_PARAM="${DS_OFFLOAD_PARAM:-none}"  # ZeRO-3 param offload: none | cpu
TRAIN_OPTIM="${TRAIN_OPTIM:-paged_adamw_8bit}"  # HF optimizer; with DS_OFFLOAD_OPT=cpu it is replaced by CPU Adam anyway
EVAL_MEM="${EVAL_MEM:-0.85}"                  # vLLM GPU memory fraction for evaluation

REWARD="${REWARD:-exp_inf}"                   # exp_inf = Rank-GRPO (exp), log_decay = Rank-GRPO (log)
OPSD_COEF="${OPSD_COEF:-0}"                   # Rank-OPSD: weight of the privileged-hint self-distillation advantage (0 = off)
OPSD_SCOPE="${OPSD_SCOPE:-dead}"              # dead = only (group, rank) cells where all 8 lists miss | all
RL_TRAIN_SET="${RL_TRAIN_SET:-rows}"          # rows = Rank-GRPO (one row per responder) | union = ReMaRk (one row per request, union of all responders' items)
RL_INIT="${RL_INIT:-}"                        # SFT checkpoint to start RL from (default: own SFT if any, else released)
RL_HOURS="${RL_HOURS:-4}"                     # wall-clock budget of the rl stage (0 = no limit)
RL_MAX_STEPS="${RL_MAX_STEPS:--1}"            # -1 = 2 epochs as in the paper (15,958 steps)
RL_BATCH="${RL_BATCH:-384}"                   # completions per optimizer step (paper: 48 prompts x 8 generations)
RL_PER_DEVICE="${RL_PER_DEVICE:-4}"           # micro-batch per GPU; grad-accum = RL_BATCH / (RL_PER_DEVICE x GPUs)
RL_SAVE_STEPS="${RL_SAVE_STEPS:-50}"
VLLM_TRAIN_MEM="${VLLM_TRAIN_MEM:-0.35}"      # vLLM GPU memory fraction while training (colocated with the policy)

SFT_HOURS="${SFT_HOURS:-10}"                  # wall-clock budget of the sft stage (0 = no limit)
SFT_STOP_STEP="${SFT_STOP_STEP:-1500}"        # paper continues RL from checkpoint-1500
SFT_BATCH="${SFT_BATCH:-192}"                 # sequences per optimizer step (paper: 12 x 8 x 2 GPUs)
SFT_PER_DEVICE="${SFT_PER_DEVICE:-4}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-100}"

COPY_CKPT="${COPY_CKPT:-1}"                   # copy final own checkpoints (weights only, ~1 GB each) to $OUT

DEFAULT_STAGES="setup data eval rl eval_rl"
# -----------------------------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO:-}" ]]; then
  if [[ -f "$SCRIPT_DIR/train_rank_grpo.py" && -w "$SCRIPT_DIR" ]]; then REPO="$SCRIPT_DIR"   # run from a checkout
  else REPO="$WORK/Rank-GRPO"; fi
fi
VENV="$WORK/venv"
SFT_RELEASED="results/Qwen/Qwen2.5-0.5B-Instruct/checkpoint-1500"
RL_RELEASED="results/grpo/Qwen/Qwen2.5-0.5B-Instruct_lr1e-06_kl0.001/checkpoint-15800"
SFT_OWN_DIR="results/own/sft"
RL_TAG="${RL_TRAIN_SET}"; [[ "$OPSD_COEF" == 0 ]] || RL_TAG="${RL_TRAIN_SET}_opsd${OPSD_COEF}_${OPSD_SCOPE}"
RL_OWN_DIR="results/own/rank_grpo_${REWARD}_${RL_TAG}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$WORK}"
export HF_HOME="${HF_HOME:-$WORK/hf}"             # base model download for the zero-shot eval
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uv-cache}"  # cleaned after setup
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg                         # Kaggle notebooks export an inline backend that scripts cannot use
export DS_SKIP_CUDA_CHECK=1                   # DeepSpeed JIT-builds its CPU Adam op against the system CUDA
[[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]] && export CUDA_HOME=/usr/local/cuda

log() { echo -e "\n[$(date +%H:%M:%S)] $*"; }
die() { echo -e "\nERROR: $*" >&2; exit 1; }
secs() { awk "BEGIN{print int($1*3600)}"; }

activate() {
  [[ -f "$VENV/bin/activate" ]] || die "environment missing, run the 'setup' stage first"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  cd "$REPO"
  mkdir -p logs results_eval "$OUT"
}

detect_gpus() {
  command -v nvidia-smi >/dev/null || die "no NVIDIA GPU found (Kaggle: Settings > Accelerator > GPU T4 x2)"
  NGPU=$(nvidia-smi -L | wc -l)
  local cc; cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader | tr -d ' ')
  local major=${cc%%.*}
  (( major >= 7 )) || die "GPU compute capability $cc is not supported by vLLM (Kaggle P100). Use 'GPU T4 x2'."
  if [[ "$PRECISION" == auto ]]; then
    if (( major >= 8 )); then PRECISION=bf16; else PRECISION=fp16; fi
  fi
  if [[ -z "$VLLM_DTYPE" ]]; then
    if (( major >= 8 )); then VLLM_DTYPE=auto; else VLLM_DTYPE=half; fi
  fi
  # vLLM's V1 engine needs compute capability >= 8.0; use the V0 engine on T4/V100
  (( major >= 8 )) || export VLLM_USE_V1=0
  case "$PRECISION" in
    bf16) PREC_FLAG=(--bf16); ACCEL_MP=bf16 ;;
    fp16) PREC_FLAG=(--fp16); ACCEL_MP=fp16 ;;
    fp32) PREC_FLAG=(); ACCEL_MP=no ;;
    *) die "PRECISION must be auto|bf16|fp16|fp32" ;;
  esac
  log "GPUs: $NGPU x $(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader) (compute $cc) | train precision: $PRECISION | vLLM dtype: $VLLM_DTYPE"
}

grad_accum() {  # grad_accum <global_batch> <per_device>
  local b=$1 pd=$2
  (( b % (pd * NGPU) == 0 )) || die "batch $b is not divisible by per-device $pd x $NGPU GPUs"
  echo $(( b / (pd * NGPU) ))
}

write_accel_cfg() {  # write_accel_cfg <file> <grad_accum>   (same DeepSpeed ZeRO-3 setup as the authors' configs)
  cat > "$1" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: $2
  offload_optimizer_device: $DS_OFFLOAD_OPT
  offload_param_device: $( [[ "$DS_ZERO" == 3 ]] && echo "$DS_OFFLOAD_PARAM" || echo none )
  zero3_init_flag: false
  zero3_save_16bit_model: true
  zero_stage: $DS_ZERO
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: $ACCEL_MP
num_machines: 1
num_processes: $NGPU
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
}

run_logged() {  # run_logged <logfile> <hours, 0 = no limit> <cmd...>; hitting the time budget is a normal end
  local logf=$1 hours=$2; shift 2
  local rc=0
  if [[ "$hours" == 0 ]]; then
    "$@" 2>&1 | tee "$logf" || rc=$?
  else
    timeout --signal=TERM --kill-after=120 "$(secs "$hours")" "$@" 2>&1 | tee "$logf" || rc=$?
  fi
  pkill -f "train_(rank_grpo|sft).py" 2>/dev/null || true   # leftover workers after a timeout
  if (( rc == 124 || rc == 137 )); then
    log "time budget of ${hours} h reached: stopped, the checkpoints saved so far are kept"
    return 0
  fi
  return $rc
}

latest_ckpt() {  # latest_ckpt <dir> [preferred_step]: prints nothing if there is no checkpoint
  local d=$1 pref=${2:-}
  if [[ -n "$pref" && -f "$d/checkpoint-$pref/config.json" ]]; then echo "$d/checkpoint-$pref"; return 0; fi
  { ls -d "$d"/checkpoint-* 2>/dev/null || true; } | sort -V | tail -1
}

locate_copy() {  # locate_copy <search root> <marker> <dest> <subdirs> <need_qwen 0|1>
  # Finds <marker> (e.g. "checkpoint-1500/model.safetensors") under the search root, either in an extracted
  # folder or inside any .zip (whatever its name), and copies <subdirs> (comma-separated, relative to the folder
  # holding the marker) into <dest>. The search root is only read (Kaggle inputs are read-only).
  # Skips tokenization caches and macOS / notebook metadata. Returns 3 if nothing matches.
  python3 - "$@" <<'EOF'
import json, os, shutil, sys, zipfile
root, marker, dest, subdirs, need_qwen = sys.argv[1:6]
subdirs = [s for s in subdirs.split(",") if s]
cfg_rel = os.path.join(os.path.dirname(marker), "config.json")
junk = lambda p: any(t in p for t in ("__MACOSX", ".ipynb_checkpoints")) or os.path.basename(p).startswith(("cache-", "._"))
is_qwen = lambda text: json.loads(text).get("model_type") == "qwen2"

files = []
for d, dirs, names in os.walk(root):
    dirs[:] = [x for x in dirs if not junk(x)]
    files += [os.path.join(d, n) for n in names]

for f in files:                                   # 1) extracted folders
    rel = os.path.relpath(f, root).replace(os.sep, "/")
    if not ("/" + rel).endswith("/" + marker):
        continue
    src = f[: len(f) - len(marker)]
    if need_qwen == "1" and not (os.path.exists(os.path.join(src, cfg_rel)) and is_qwen(open(os.path.join(src, cfg_rel)).read())):
        continue
    for s in subdirs or [""]:
        shutil.copytree(os.path.join(src, s), os.path.join(dest, s), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("cache-*.arrow", "._*", ".ipynb_checkpoints", "__MACOSX"))
    print(f"copied {src} -> {dest}"); sys.exit(0)

for f in files:                                   # 2) zip archives
    if not f.endswith(".zip"):
        continue
    with zipfile.ZipFile(f) as z:
        names = z.namelist()
        for n in names:
            if junk(n) or not ("/" + n).endswith("/" + marker):
                continue
            prefix = n[: len(n) - len(marker)]
            if need_qwen == "1" and not (prefix + cfg_rel in names and is_qwen(z.read(prefix + cfg_rel))):
                continue
            keep = [prefix + s.rstrip("/") + "/" if s else prefix for s in subdirs or [""]]
            for m in names:
                if m.endswith("/") or junk(m) or not any(m.startswith(k) for k in keep):
                    continue
                out = os.path.join(dest, m[len(prefix):])
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(m) as fi, open(out, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            print(f"extracted {f}:{prefix} -> {dest}"); sys.exit(0)
sys.exit(3)
EOF
}

fetch_drive() {  # fetch_drive <drive_id> <out_file> <expected_bytes>
  local id=$1 out=$2 size=$3
  if [[ -f "$out" && "$(stat -c%s "$out")" == "$size" ]]; then return; fi
  log "downloading $(basename "$out") from Google Drive ..."
  curl -L --fail --retry 3 -C - -o "$out" "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t" \
    || die "download failed; attach the file as a Kaggle dataset instead (see the README links) and re-run"
  [[ "$(stat -c%s "$out")" == "$size" ]] || die "size mismatch for $out (Drive quota?); attach it as a Kaggle dataset instead"
}

get_artifact() {  # get_artifact <marker> <dest> <subdirs> <need_qwen> <zip name> <drive id> <bytes>
  # 1) already in WORK, 2) in $INPUT_ROOT (folder or zip, any name), 3) download the release zip from Google Drive
  local marker=$1 dest=$2 subdirs=$3 qwen=$4 zipname=$5 id=$6 size=$7
  if [[ -f "$dest/$marker" ]]; then echo "present: $dest/$marker"; return; fi
  if [[ -d "$INPUT_ROOT" ]] && locate_copy "$INPUT_ROOT" "$marker" "$dest" "$subdirs" "$qwen"; then return; fi
  log "'$marker' not found in $INPUT_ROOT, downloading $zipname"
  mkdir -p "$WORK/downloads"
  fetch_drive "$id" "$WORK/downloads/$zipname" "$size"
  locate_copy "$WORK/downloads" "$marker" "$dest" "$subdirs" "$qwen" || die "unexpected archive layout in $zipname"
}

eval_models() {  # eval_models name=model_path ...   (one vLLM job per GPU in parallel)
  local -a names=() pids=()
  local spec i=0 failed=0
  for spec in "$@"; do
    local name=${spec%%=*} path=${spec#*=}
    log "eval $name ($path) on GPU $(( i % NGPU ))"
    CUDA_VISIBLE_DEVICES=$(( i % NGPU )) python evaluate/eval_grpo_test.py \
      --model_name "$MODEL" --model_path "$path" --dtype "$VLLM_DTYPE" \
      --dataset_path processed_datasets/sft_dataset --catalog_path reddit_catalog/gt_catalog.pkl \
      --gpu_memory_utilization "$EVAL_MEM" --tensor_parallel_size 1 --save_dir "results_eval/$name" \
      > "logs/eval_$name.txt" 2>&1 &
    names+=("$name"); pids+=($!); i=$(( i + 1 ))
    if (( ${#pids[@]} >= NGPU )); then
      wait "${pids[0]}" || { echo "eval ${names[0]} FAILED, last lines of logs/eval_${names[0]}.txt:"; tail -20 "logs/eval_${names[0]}.txt"; failed=1; }
      names=("${names[@]:1}"); pids=("${pids[@]:1}")
    fi
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "eval ${names[$i]} FAILED, last lines of logs/eval_${names[$i]}.txt:"; tail -20 "logs/eval_${names[$i]}.txt"; failed=1; }
  done
  summary
  return $failed
}

summary() {  # table of all results_eval/* next to paper Table 1; also written to $OUT/summary.txt
  python3 - "$OUT/summary.txt" <<'EOF'
import glob, json, os, sys
paper = {  # R@5 R@10 R@15 R@20 N@5 N@10 N@15 N@20 (Table 1, REDDIT-V2 test)
    "zero_shot":           [0.0021, 0.0021, 0.0021, 0.0021, 0.0023, 0.0022, 0.0021, 0.0021],
    "sft_ckpt1500":        [0.0642, 0.1027, 0.1212, 0.1308, 0.0502, 0.0625, 0.0678, 0.0704],
    "rank_grpo_ckpt15800": [0.0946, 0.1468, 0.1813, 0.2047, 0.0744, 0.0915, 0.1016, 0.1079],
}
ks = (5, 10, 15, 20)
lines = [f"{'run':40s} {'':6s} " + " ".join(f"{m}@{k}".ljust(6) for m in ("R", "N") for k in ks)]
for path in sorted(glob.glob("results_eval/*/analysis_state.json")):
    name = path.split(os.sep)[-2]
    e = json.load(open(path))["log_history"][0]
    ours = [e[f"eval_recall@{k}"] for k in ks] + [e[f"eval_ndcg@{k}"] for k in ks]
    lines.append(f"{name:40s} {'ours':6s} " + " ".join(f"{v:.4f}" for v in ours))
    if name in paper:
        lines.append(f"{'':40s} {'paper':6s} " + " ".join(f"{v:.4f}" for v in paper[name]))
text = "\n".join(lines)
print("\n" + text)
open(sys.argv[1], "w").write(text + "\n")
EOF
  cp -r results_eval "$OUT/" 2>/dev/null || true
  find "$OUT/results_eval" -name "*.pkl" -delete 2>/dev/null || true   # raw generations, large
  cp -r logs "$OUT/" 2>/dev/null || true
}

save_ckpt() {  # save_ckpt <checkpoint dir> <name>: weights + config + tokenizer + trainer_state to $OUT
  [[ "$COPY_CKPT" == 1 && -d "$1" ]] || return 0
  mkdir -p "$OUT/checkpoints/$2"
  find "$1" -maxdepth 1 -type f \( -name "*.json" -o -name "*.safetensors" -o -name "*.txt" -o -name "*.jinja" \) \
    -exec cp {} "$OUT/checkpoints/$2/" \;
  log "saved $1 -> $OUT/checkpoints/$2"
}

# ---------------------------------------------- STAGES -----------------------------------------------
stage_setup() {
  mkdir -p "$WORK" "$OUT"
  log "disk space for WORK=$WORK:"; df -h "$WORK" | tail -1
  local free_gb; free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
  (( free_gb >= 30 )) || echo "WARNING: only ${free_gb} GB free in $WORK (about 25-30 GB are needed); set WORK=... to a larger disk"
  if [[ "$REPO" == "$SCRIPT_DIR" ]]; then
    log "using the code in $REPO ($(git -C "$REPO" log -1 --format='%h %s' 2>/dev/null || echo 'not a git checkout'))"
  else
    local url=$REPO_URL
    [[ -n "${GITHUB_TOKEN:-}" ]] && url="https://${GITHUB_TOKEN}@${REPO_URL#https://}"
    if [[ ! -d "$REPO/.git" ]]; then
      log "cloning $REPO_URL ($REPO_REF)"
      git clone -q "$url" "$REPO"
      git -C "$REPO" remote set-url origin "$REPO_URL"   # do not keep the token in .git/config
    fi
    git -C "$REPO" fetch -q "$url" "$REPO_REF"
    git -C "$REPO" checkout -q -B run "FETCH_HEAD"
    log "code: $REPO @ $(git -C "$REPO" log -1 --format='%h %s')"
  fi

  command -v uv >/dev/null || pip install -q uv
  # Kaggle's image sets UV_* variables (e.g. UV_SYSTEM_PYTHON) that make uv install into the system Python;
  # always target the venv explicitly.
  unset UV_SYSTEM_PYTHON UV_PYTHON UV_PROJECT_ENVIRONMENT
  env | grep '^UV_' || true
  [[ -x "$VENV/bin/python" ]] || uv venv -q --python 3.10 "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  local vpip=(uv pip install -q --python "$VENV/bin/python")
  log "installing pinned packages into $VENV (torch 2.7.1 cu126, vLLM 0.10.0, TRL 0.21.0, ...)"
  "${vpip[@]}" torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
  # versions contemporary with TRL 0.21 (transformers 5.x breaks TRL 0.21 imports)
  "${vpip[@]}" vllm==0.10.0 trl==0.21.0 transformers==4.55.4 accelerate==1.10.0
  "${vpip[@]}" setuptools wheel ninja
  "${vpip[@]}" --no-build-isolation "deepspeed>=0.16,<0.18"
  # datasets>=4 is needed to read processed_datasets/grpo (saved with the `List` feature type)
  "${vpip[@]}" bitsandbytes "datasets>=4.0,<5" wandb editdistance regex seaborn matplotlib transformers==4.55.4 accelerate==1.10.0
  [[ "$(command -v python)" == "$VENV/bin/python" ]] || die "venv python not active: $(command -v python)"
  "$VENV/bin/python" -c "import torch, trl" || die "packages missing from $VENV after install"
  (cd "$REPO" && bash INSTALL.SH)
  "$VENV/bin/python" -c "from trl import RankGRPOTrainer" || die "INSTALL.SH did not install RankGRPOTrainer into TRL"
  python -c "import torch, trl, vllm, deepspeed; from trl import RankGRPOTrainer; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| trl', trl.__version__, '| vllm', vllm.__version__, '| deepspeed', deepspeed.__version__)"
  uv cache clean -q || true
  log "disk after setup:"; df -h "$WORK" /kaggle/working 2>/dev/null | tail -n +2 || true
}

stage_data() {
  activate
  if [[ -d "$INPUT_ROOT" ]]; then
    log "read-only input $INPUT_ROOT contains:"; { find "$INPUT_ROOT" -maxdepth 3 || true; } | head -40
  fi
  get_artifact "grpo/grpo_dataset/train/state.json" processed_datasets "grpo,sft_dataset" 0 \
    processed_datasets.zip 11tOfUMlVOylkkcnwPqGM_0IuiIeHjLle 792594362
  [[ -f processed_datasets/sft_dataset/test/state.json ]] || die "processed_datasets/sft_dataset is missing next to grpo/ in the input"
  get_artifact "checkpoint-1500/model.safetensors" "${SFT_RELEASED%/*}" checkpoint-1500 1 \
    Qwen2.5-0.5B-Instruct.zip 1Br-CMrzgUzJ7_fwFVahCqzJ8TyGZ5Zfz 787893402
  get_artifact "checkpoint-15800/model.safetensors" "${RL_RELEASED%/*}" checkpoint-15800 1 \
    Qwen2.5-0.5B-Instruct_lr1e-06_kl0.001.zip 1VNSkbXGpawfj1amRMhfoiPQHsOKcYJzA 788757427
  rm -rf "$WORK/downloads"
  du -sh processed_datasets "$SFT_RELEASED" "$RL_RELEASED" 2>/dev/null || true
}

stage_eval() {
  activate; detect_gpus
  eval_models "zero_shot=$MODEL" "sft_ckpt1500=$SFT_RELEASED" "rank_grpo_ckpt15800=$RL_RELEASED"
}

stage_sft() {
  activate; detect_gpus
  local ga; ga=$(grad_accum "$SFT_BATCH" "$SFT_PER_DEVICE")
  write_accel_cfg "$WORK/accel_sft.yaml" "$ga"
  log "SFT: $NGPU GPU x $SFT_PER_DEVICE x grad-accum $ga = $SFT_BATCH seqs/step, stop at step $SFT_STOP_STEP, budget ${SFT_HOURS} h"
  run_logged logs/train_sft.txt "$SFT_HOURS" \
    accelerate launch --config_file "$WORK/accel_sft.yaml" train_sft.py \
      --dataset_path processed_datasets/sft_dataset --model_name "$MODEL" --output_dir "$SFT_OWN_DIR" \
      --learning_rate 5e-5 --warmup_ratio 0.05 --lr_scheduler_type cosine --optim "$TRAIN_OPTIM" \
      --num_train_epochs 10 --stop_at_step "$SFT_STOP_STEP" --max_length 1024 \
      --per_device_train_batch_size "$SFT_PER_DEVICE" --per_device_eval_batch_size "$SFT_PER_DEVICE" \
      --gradient_accumulation_steps "$ga" --gradient_checkpointing "${PREC_FLAG[@]}" \
      --save_strategy steps --save_steps "$SFT_SAVE_STEPS" --save_total_limit 2 --save_only_model \
      --eval_strategy no --logging_steps 10 --dataset_num_proc 4 --seed 3407
  local ck; ck=$(latest_ckpt "$SFT_OWN_DIR" "$SFT_STOP_STEP")
  [[ -n "$ck" ]] || die "SFT produced no checkpoint, see logs/train_sft.txt"
  save_ckpt "$ck" "sft_$(basename "$ck")"
}

stage_eval_sft() {
  activate; detect_gpus
  local ck; ck=$(latest_ckpt "$SFT_OWN_DIR" "$SFT_STOP_STEP")
  [[ -n "$ck" ]] || die "no own SFT checkpoint in $SFT_OWN_DIR (run the 'sft' stage)"
  eval_models "own_sft_$(basename "$ck")=$ck"
}

stage_rl() {
  activate; detect_gpus
  local init=$RL_INIT
  if [[ -z "$init" ]]; then
    init=$(latest_ckpt "$SFT_OWN_DIR" "$SFT_STOP_STEP")
    [[ -n "$init" ]] || init=$SFT_RELEASED
  fi
  [[ -f "$init/config.json" ]] || die "SFT checkpoint not found: $init (run the 'data' stage)"
  local ga; ga=$(grad_accum "$RL_BATCH" "$RL_PER_DEVICE")
  (( RL_BATCH % 8 == 0 )) || die "RL_BATCH must be a multiple of the 8 generations per prompt"
  write_accel_cfg "$WORK/accel_rl.yaml" "$ga"
  export RANK_GRPO_VLLM_DTYPE="$VLLM_DTYPE"
  local train_path=processed_datasets/grpo/grpo_dataset
  if [[ "$RL_TRAIN_SET" == union ]]; then
    train_path=processed_datasets/grpo/grpo_union
    [[ -f "$train_path/train/state.json" ]] || python scripts/build_union_dataset.py --dst "$train_path/train"
  elif [[ "$RL_TRAIN_SET" != rows ]]; then
    die "RL_TRAIN_SET must be rows or union"
  fi
  log "Rank-GRPO ($REWARD, train set: $RL_TRAIN_SET, opsd coef $OPSD_COEF scope $OPSD_SCOPE) from $init: $NGPU GPU x $RL_PER_DEVICE x grad-accum $ga = $RL_BATCH completions/step, budget ${RL_HOURS} h"
  run_logged "logs/train_rank_grpo_${REWARD}.txt" "$RL_HOURS" \
    accelerate launch --config_file "$WORK/accel_rl.yaml" train_rank_grpo.py \
      --train_path "$train_path" --model_name "$MODEL" --sft_model_path "$init" \
      --catalog_path reddit_catalog/gt_catalog.pkl --reward_func "$REWARD" --output_dir "$RL_OWN_DIR" \
      --mu 1 --lr 1e-6 --kl_beta 1e-3 --adam_beta1 0.9 --adam_beta2 0.99 --optim "$TRAIN_OPTIM" \
      --num_train_epochs 2 --max_steps "$RL_MAX_STEPS" \
      --per_device_train_batch_size "$RL_PER_DEVICE" --per_device_eval_batch_size "$RL_PER_DEVICE" \
      --gradient_accumulation_steps "$ga" --num_generations 8 \
      --max_prompt_length 2048 --max_completion_length 1024 \
      --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization "$VLLM_TRAIN_MEM" --vllm_tensor_parallel_size 1 \
      --gradient_checkpointing "${PREC_FLAG[@]}" \
      --save_strategy steps --save_steps "$RL_SAVE_STEPS" --save_total_limit 2 --save_only_model \
      --logging_steps 1 --seed 3407 --wandb_project rank_grpo --opsd_coef "$OPSD_COEF" --opsd_scope "$OPSD_SCOPE"
  local ck; ck=$(latest_ckpt "$RL_OWN_DIR")
  [[ -n "$ck" ]] || die "Rank-GRPO produced no checkpoint (budget too short for $RL_SAVE_STEPS steps?), see logs/train_rank_grpo_${REWARD}.txt"
  save_ckpt "$ck" "rank_grpo_${REWARD}_${RL_TAG}_$(basename "$ck")"
  python3 - "$ck/trainer_state.json" <<'EOF'
import json, sys
h = [e for e in json.load(open(sys.argv[1]))["log_history"] if "rewards/pos_1/mean" in e]
if h:
    first, last = h[:max(1, len(h) // 10)], h[-max(1, len(h) // 10):]
    avg = lambda rows, k: sum(r[k] for r in rows) / len(rows)
    print(f"\nper-rank train reward (first 10% -> last 10% of {h[-1]['step']} steps):")
    for k in (1, 2, 5, 10, 15, 20):
        key = f"rewards/pos_{k}/mean"
        print(f"  rank {k:2d}: {avg(first, key):.4f} -> {avg(last, key):.4f}")
EOF
}

stage_noise() {  # responder-noise diagnostic (ReMaRk claim) on the SFT policy: 500 requests x 8 lists
  activate; detect_gpus
  local init=${RL_INIT:-$SFT_RELEASED}
  run_logged logs/responder_noise.txt 0 python scripts/responder_noise.py --model "$init" --dtype "$VLLM_DTYPE"     --n_requests "${NOISE_REQUESTS:-500}" --out "$OUT/responder_noise"
  cp logs/responder_noise.txt "$OUT/" 2>/dev/null || true
}

stage_eval_rl() {
  activate; detect_gpus
  local ck; ck=$(latest_ckpt "$RL_OWN_DIR")
  [[ -n "$ck" ]] || die "no own Rank-GRPO checkpoint in $RL_OWN_DIR (run the 'rl' stage)"
  eval_models "own_rank_grpo_${REWARD}_${RL_TAG}_$(basename "$ck")=$ck"
}

main() {
  local stages=("$@")
  (( ${#stages[@]} )) || read -r -a stages <<< "$DEFAULT_STAGES"
  local s
  for s in "${stages[@]}"; do
    case "$s" in setup|data|eval|noise|sft|eval_sft|rl|eval_rl) ;; *) die "unknown stage '$s'" ;; esac
  done
  log "stages: ${stages[*]} | WORK=$WORK | OUT=$OUT"
  for s in "${stages[@]}"; do
    log "==================== stage: $s ===================="
    "stage_$s"
  done
  log "all done. Outputs (summary, logs, metrics, checkpoints) in $OUT"
}

# run only when executed (not when sourced, e.g. to call single functions)
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
