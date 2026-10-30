"""Build HF-backend hint-probe notebooks (one per model family) from the v3 notebook + local hint_probe.py.

Usage: python -I probe/kaggle/build_family_nb.py [family ...]
Writes probe/kaggle/<family>/{crs-hint-probe.ipynb, kernel-metadata.json}.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)

RUNS = {
    "llama": dict(
        account="wibu505",
        run="v4-hint-probe-llama-3.2-1b-3b-3.1-8b-n300",
        desc="Llama 3.2-1B / 3.2-3B / 3.1-8B Instruct (ungated unsloth mirrors, same weights as meta-llama)",
        models=[("unsloth/Llama-3.2-1B-Instruct", "fp16", 16, "auto"), ("unsloth/Llama-3.2-3B-Instruct", "fp16", 8, "auto"),
                ("unsloth/Llama-3.1-8B-Instruct", "fp16", 8, "auto")],
    ),
    "gemma": dict(
        account="wibukun2003",
        run="v4-hint-probe-gemma-3-270m-1b-4-e2b-e4b-n300",
        desc="Gemma 3 270M / 1B it (ungated unsloth mirrors) and Gemma 4 E2B / E4B it. Gemma 3 overflows in fp16, "
             "so the small ones run in fp32; Gemma 4 runs fp16 and is retried in fp32 if the output degenerates.",
        models=[("unsloth/gemma-3-270m-it", "fp32", 16, "auto"), ("unsloth/gemma-3-1b-it", "fp32", 8, "auto"),
                ("google/gemma-4-E2B-it", "fp16", 8, "auto"), ("google/gemma-4-E4B-it", "fp16", 8, "auto")],
    ),
    "gemma_rerun": dict(
        account="wibukunnnnn",
        run="v5-hint-probe-gemma-3-270m-fp32-gemma-4-e2b-e4b-bf16-n300",
        desc="rerun of the v4 Gemma models that failed: Gemma 3 270M it in fp32 on one GPU (v4 split it over 2 GPUs "
             "and crashed), Gemma 4 E2B / E4B it in bf16 (v4 fp16 gave NaN, fp32 did not fit and hung)",
        models=[("unsloth/gemma-3-270m-it", "fp32", 16, "cuda:0"), ("google/gemma-4-E2B-it", "bf16", 8, "auto"),
                ("google/gemma-4-E4B-it", "bf16", 8, "auto")],
    ),
    "gemma4_e4b": dict(
        account="datogg",
        run="v6-hint-probe-gemma-4-e4b-bf16-bs2-n300",
        desc="rerun of Gemma 4 E4B it in bf16 with batch size 2 (v5 ran out of GPU memory during generation at bs 8)",
        models=[("google/gemma-4-E4B-it", "bf16", 2, "auto")],
    ),
    "gemma_t4": dict(
        account="wibukunnnnn",
        run="v7-hint-probe-gemma-3-4b-fp32-gemma-4-e4b-nf4-n300",
        desc="T4-friendly Gemma runs (T4 has no native bf16 and Gemma overflows in fp16): Gemma 3 4B it in fp32, and "
             "Gemma 4 E4B it with bitsandbytes nf4 4-bit weights + fp32 compute (v6 bf16 was too slow: 3 h for 1 of 3 conditions)",
        pip="bitsandbytes",
        models=[("unsloth/gemma-3-4b-it", "fp32", 4, "auto"), ("google/gemma-4-E4B-it", "nf4", 4, "auto")],
    ),
    "gemma3_4b": dict(
        account="wibukunnnnn",
        run="v8-hint-probe-gemma-3-4b-fp32-n300",
        desc="Gemma 3 4B it in fp32 only (v7 cancelled; Gemma 4 dropped: no T4-friendly dtype)",
        models=[("unsloth/gemma-3-4b-it", "fp32", 4, "auto")],
    ),
    "qwen3_8b_thinking": dict(
        account="wibukunnnnn",
        run="v9-hint-probe-qwen3-8b-thinking-n300",
        desc="Qwen3-8B in thinking mode (sampled, T 0.6 / top-p 0.95 / top-k 20, up to 3072 new tokens), conditions "
             "partial, none, full, to compare with the v3 non-thinking run of the same 300 rows",
        timeout_h=10.5,
        models=[("Qwen/Qwen3-8B", "fp16", 8, "auto", "--thinking --max_new_tokens 3072 --conds partial,none,full")],
    ),
}

SETUP = '''RUN_NAME = "{run}"
import os, subprocess, sys
OUT = f"/kaggle/working/{{RUN_NAME}}"
os.makedirs(OUT, exist_ok=True)
os.environ["HF_HOME"] = "/tmp/hf"
r = subprocess.run("pip install -q editdistance regex {pip}", shell=True, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
subprocess.run("python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.device_count())'", shell=True)
import torch, time as _t  # T4 has no native bf16: check that bf16 matmul works and how slow it is vs fp16
for dt in (torch.float16, torch.bfloat16, torch.float32):
    a = torch.randn(4096, 4096, device="cuda", dtype=dt); torch.cuda.synchronize(); t = _t.time()
    for _ in range(10): b = a @ a
    torch.cuda.synchronize(); print(dt, "10 matmul 4096^2: %.3f s" % (_t.time() - t), "finite", bool(b.isfinite().all()))
del a, b; torch.cuda.empty_cache()
subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv && df -h /tmp /kaggle/working | tail -2", shell=True)'''

LOOP = '''import time, shutil, json, math
MODELS = {models!r}  # (model, dtype, batch size, device_map[, extra args])
env = dict(os.environ, RANKGRPO_REPO="/tmp/rankgrpo/Rank-GRPO", PROBE_DATA=TEST, TOKENIZERS_PARALLELISM="false",
           PYTHONPATH="/kaggle/working", HF_HOME="/tmp/hf",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

def run(model, dtype, bs, dmap, extra, tag):
    cmd = (f"python /kaggle/working/hint_probe.py --model {{model}} --n 300 --bs {{bs}} --dtype {{dtype}} --device_map {{dmap}}"
           f" {{extra}} --out_dir {{OUT}} > {{OUT}}/run_{{tag}}_{{dtype}}.log 2>&1")
    p = subprocess.Popen(cmd, shell=True, env=env, start_new_session=True)
    try:
        rc = p.wait(timeout={timeout_h} * 3600)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)  # kill the python child too, otherwise it keeps holding GPU memory
        p.wait()
        rc = "timeout"
    if rc != 0:
        print(f"[{{tag}}] {{dtype}} failed ({{rc}}), last log lines:")
        print("".join(open(f"{{OUT}}/run_{{tag}}_{{dtype}}.log").readlines()[-25:]))
    return rc

def degenerate(tag):
    """True if the run produced no summary, NaN log-probs or (almost) empty answers (fp16 overflow symptoms)."""
    f = f"{{OUT}}/{{tag}}_summary.json"
    if not os.path.exists(f):
        return True
    s = json.load(open(f))
    lp = s.get("gold_logprob_per_token", {{}})  # absent in thinking mode
    return any(math.isnan(v) for v in lp.values()) or any(c["mean_items"] < 1 for c in s["conditions"].values())

status = {{}}
for model, dtype, bs, dmap, *extra in MODELS:
    extra = extra[0] if extra else ""
    tag = model.split("/")[-1] + ("_thinking" if "--thinking" in extra else "")
    t0 = time.time()
    rc = run(model, dtype, bs, dmap, extra, tag)
    bad = degenerate(tag)
    if bad:
        print(f"[{{tag}}] {{dtype}} output degenerate or missing (NaN log-probs / empty answers)", flush=True)
    status[tag] = (dtype, rc, round((time.time() - t0) / 60, 1), "degenerate" if bad else "ok")
    print(tag, status[tag], flush=True)
    subprocess.run("pkill -9 -f hint_probe.py", shell=True)
    shutil.rmtree("/tmp/hf/hub", ignore_errors=True)  # free disk before the next model
json.dump(status, open(f"{{OUT}}/status.json", "w"), indent=2)
print(status)'''


def src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main():
    only = sys.argv[1:]
    base = json.load(open(os.path.join(HERE, "crs-hint-probe.ipynb"), encoding="utf-8"))
    meta = json.load(open(os.path.join(HERE, "kernel-metadata.json"), encoding="utf-8"))
    probe_src = open(os.path.join(PROBE, "hint_probe.py"), encoding="utf-8").read()
    for fam, cfg in RUNS.items():
        if only and fam not in only:
            continue
        nb = copy.deepcopy(base)
        c = nb["cells"]
        assert ''.join(c[4]["source"]).startswith("%%writefile /kaggle/working/hint_probe_vllm.py")
        c[0]["source"] = src(f"# {cfg['run']}\n\nPrivileged-hint probe (same 300 REDDIT-V2 test prompts as the Qwen runs v1/v3), "
                             f"HF transformers backend, greedy, 2x T4: {cfg['desc']}.")
        c[1]["source"] = src(SETUP.format(run=cfg["run"], pip=cfg.get("pip", "")))
        c[3]["source"] = src("%%writefile /kaggle/working/hint_probe.py\n" + probe_src)
        c[5]["source"] = src(LOOP.format(models=cfg["models"], timeout_h=cfg.get("timeout_h", 3)))
        del c[4]
        for x in c:
            if x["cell_type"] == "code":
                x["outputs"], x["execution_count"] = [], None
        d = os.path.join(HERE, fam)
        os.makedirs(d, exist_ok=True)
        json.dump(nb, open(os.path.join(d, "crs-hint-probe.ipynb"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        m = dict(meta, id=f"{cfg['account']}/crs-hint-probe")
        m.pop("id_no", None)
        json.dump(m, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
        for i, x in enumerate(nb["cells"]):  # syntax check every code cell (strip %%writefile magics)
            if x["cell_type"] == "code":
                s = "".join(x["source"])
                compile(s.split("\n", 1)[1] if s.startswith("%%") else s, f"{fam}-cell{i}", "exec")
        print(fam, cfg["account"], cfg["run"], "cells", len(nb["cells"]))


if __name__ == "__main__":
    main()
