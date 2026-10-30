"""Build the Kaggle notebook for probe/rollout_probe.py.

Usage: python -I probe/kaggle/build_rollout_nb.py [v1|v2]
Writes probe/kaggle/rollout[_v2]/{crs-rollout-probe.ipynb, kernel-metadata.json}.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)
# scorer = (name, model, hint style or "" for no hint, dtype, score batch size); "student" (no hint) is the reference
# for A. model: "SFT" / "RL" = released checkpoints from Drive, "HINTSFT" = hint-SFT checkpoint from kernel_sources
RUNS = {
    "v1": dict(account="wibu505", run="rollout-probe-v1-sft05b-g8-teachers-self-rl-qwen3-4b-8b-n300", sources=[],
               scorers=[("student", "SFT", "", "fp16", 16), ("student_hint", "SFT", "v1", "fp16", 16),
                        ("rl15800", "RL", "", "fp16", 16),
                        ("qwen3_4b", "Qwen/Qwen3-4B", "", "fp16", 8), ("qwen3_4b_hint", "Qwen/Qwen3-4B", "v1", "fp16", 8),
                        ("qwen3_8b", "Qwen/Qwen3-8B", "", "fp16", 8), ("qwen3_8b_hint", "Qwen/Qwen3-8B", "v1", "fp16", 8)]),
    "v2": dict(account="wibukunn", run="rollout-probe-v2-hintsft-ckpt202-hint-v1-v2-n300",
               sources=["wibukunn/rankgrpo"],  # output of v11-hintsft-v2hint-replay-lr1e-5-1ep-eval-probe
               scorers=[("student", "SFT", "", "fp16", 16), ("student_hint_v1", "SFT", "v1", "fp16", 16),
                        ("student_hint_v2", "SFT", "v2", "fp16", 16), ("hintsft", "HINTSFT", "", "fp16", 16),
                        ("hintsft_hint_v2", "HINTSFT", "v2", "fp16", 16), ("hintsft_hint_v1", "HINTSFT", "v1", "fp16", 16)]),
}
CFG = RUNS[sys.argv[1] if len(sys.argv) > 1 else "v1"]
ACCOUNT, RUN, SCORERS = CFG["account"], CFG["run"], CFG["scorers"]

MD = f"""# {RUN}

Rollout-scoring probe for on-policy (self-)distillation in Rank-GRPO (REDDIT-V2, 300 training requests with >= 2 responders).
The released SFT checkpoint-1500 (Qwen2.5-0.5B) samples G = 8 lists per request at temperature 1. Each scorer gives the
log-prob of every item the student wrote; A = lp_scorer - lp_student. AUC of A: revealed gold (drawn responder, in the hint)
vs miss, and other responders' gold (NOT revealed) vs miss, within request, also on dead rank groups.
Scorers: SFT student (reference), SFT + hint (self-distillation), released Rank-GRPO checkpoint-15800, Qwen3-4B and Qwen3-8B with
and without the hint."""

SETUP = f'''RUN_NAME = "{RUN}"
import os, subprocess, sys
OUT = f"/kaggle/working/{{RUN_NAME}}"
os.makedirs(OUT, exist_ok=True)
os.environ["HF_HOME"] = "/tmp/hf"
r = subprocess.run("pip install -q editdistance regex", shell=True, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
subprocess.run("python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.device_count())'", shell=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv && df -h /tmp /kaggle/working | tail -2", shell=True)'''

DATA = '''import zipfile
def drive(fid, out, size):
    subprocess.run(f'curl -L --fail --retry 3 -s -o {out} "https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"',
                   shell=True, check=True)
    assert os.path.getsize(out) == size, (out, os.path.getsize(out))
junk = lambda m: m.startswith("__MACOSX/") or "/._" in m or "cache-" in m
subprocess.run("rm -rf /tmp/rankgrpo && git clone -q --depth 1 https://github.com/datht4889/Rank-GRPO.git /tmp/rankgrpo/Rank-GRPO", shell=True, check=True)
drive("11tOfUMlVOylkkcnwPqGM_0IuiIeHjLle", "/tmp/pd.zip", 792594362)
with zipfile.ZipFile("/tmp/pd.zip") as z:
    z.extractall("/tmp/rankgrpo/data", [m for m in z.namelist() if "grpo/grpo_dataset/train/" in m and not junk(m)])
os.remove("/tmp/pd.zip")
TRAIN = [os.path.join(r, f) for r, _, fs in os.walk("/tmp/rankgrpo/data") for f in fs if f.endswith(".arrow")][0]
CKPT = {}
NEED = __NEED__
for key, fid, size, step in (("SFT", "1Br-CMrzgUzJ7_fwFVahCqzJ8TyGZ5Zfz", 787893402, "checkpoint-1500"),
                             ("RL", "1VNSkbXGpawfj1amRMhfoiPQHsOKcYJzA", 788757427, "checkpoint-15800")):
    if key not in NEED:
        continue
    drive(fid, "/tmp/ck.zip", size)
    with zipfile.ZipFile("/tmp/ck.zip") as z:
        z.extractall(f"/tmp/ckpt_{key}", [m for m in z.namelist() if not junk(m)])
    os.remove("/tmp/ck.zip")
    CKPT[key] = [r for r, _, fs in os.walk(f"/tmp/ckpt_{key}") if os.path.basename(r) == step and "config.json" in fs][0]
if "HINTSFT" in NEED:  # attached kernel output (kernel_sources)
    CKPT["HINTSFT"] = [r for r, _, fs in os.walk("/kaggle/input") if os.path.basename(r).startswith("hintsft_checkpoint")
                       and "config.json" in fs][0]
print(TRAIN, os.path.getsize(TRAIN), CKPT)'''

LOOP = '''import time, shutil, json
SCORERS = {scorers!r}  # (name, model, hint, dtype, score batch size)
env = dict(os.environ, RANKGRPO_REPO="/tmp/rankgrpo/Rank-GRPO", PROBE_TRAIN=TRAIN, TOKENIZERS_PARALLELISM="false",
           HF_HOME="/tmp/hf", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
status = {{}}

def run(tag, args, timeout_h):
    t0 = time.time()
    p = subprocess.Popen(f"python /kaggle/working/rollout_probe.py {{args}} --out {{OUT}} > {{OUT}}/run_{{tag}}.log 2>&1",
                         shell=True, env=env, start_new_session=True)
    try:
        rc = p.wait(timeout=timeout_h * 3600)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)  # also kills the python child, which would otherwise keep GPU memory
        p.wait()
        rc = "timeout"
    subprocess.run("pkill -9 -f rollout_probe.py", shell=True)
    status[tag] = (rc, round((time.time() - t0) / 60, 1))
    print(tag, status[tag], flush=True)
    if rc != 0:
        print("".join(open(f"{{OUT}}/run_{{tag}}.log").readlines()[-25:]))
    json.dump(status, open(f"{{OUT}}/status.json", "w"), indent=2)
    return rc

assert run("rollout", f"rollout --student {{CKPT['SFT']}} --n_requests 300 --G 8 --gen_bs 8 --max_new_tokens 512", 2) == 0
for name, model, hint, dtype, bs in SCORERS:
    m = CKPT.get(model, model)
    run(name, f"score --name {{name}} --model {{m}} {{('--hint --hint_style ' + hint) if hint else ''}} --dtype {{dtype}} --score_bs {{bs}}", 2.5)
    shutil.rmtree("/tmp/hf/hub", ignore_errors=True)  # free disk before the next model
run("report", "report --boot 300", 0.5)
print(open(f"{{OUT}}/run_report.log").read()[-6000:])'''


def src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main():
    probe_src = open(os.path.join(PROBE, "rollout_probe.py"), encoding="utf-8").read()
    need = sorted({m for _, m, _, _, _ in SCORERS if m in ("SFT", "RL", "HINTSFT")} | {"SFT"})
    cells = [("markdown", MD), ("code", SETUP), ("code", DATA.replace("__NEED__", repr(need))),
             ("code", "%%writefile /kaggle/working/rollout_probe.py\n" + probe_src), ("code", LOOP.format(scorers=SCORERS))]
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, text) in enumerate(cells):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src(text)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
            compile(text.split("\n", 1)[1] if text.startswith("%%") else text, f"cell{i}", "exec")
        nb["cells"].append(c)
    d = os.path.join(HERE, "rollout" if RUN.startswith("rollout-probe-v1") else "rollout_" + RUN.split("-")[2])
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, "crs-rollout-probe.ipynb"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    meta = json.load(open(os.path.join(HERE, "kernel-metadata.json"), encoding="utf-8"))
    meta.update(id=f"{ACCOUNT}/crs-rollout-probe", title="crs-rollout-probe", code_file="crs-rollout-probe.ipynb",
                kernel_sources=CFG["sources"])
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    print(ACCOUNT, RUN, len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
