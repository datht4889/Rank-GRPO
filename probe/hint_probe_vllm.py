"""vLLM backend for the privileged-hint probe (same rows, prompts and metrics as hint_probe.py).

Usage: python probe/hint_probe_vllm.py --model Qwen/Qwen2.5-7B-Instruct --n 300 --tp 2
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from vllm import LLM, SamplingParams

from hint_probe import (CATALOG, FULL_HINT, PARTIAL_HINT, build_chat, fmt_items, load_catalog, load_rows,
                        score)


def gold_logprob(llm, tok, prompts, targets):
    """Mean per-token log p(target | prompt) via vLLM prompt_logprobs."""
    reqs, spans = [], []
    for p, t in zip(prompts, targets):
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        t_ids = tok(t, add_special_tokens=False)["input_ids"]
        reqs.append({"prompt_token_ids": p_ids + t_ids})
        spans.append((len(p_ids), len(p_ids) + len(t_ids)))
    outs = llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
    res = []
    for o, req, (a, b) in zip(outs, reqs, spans):
        ids = req["prompt_token_ids"]
        res.append(float(np.mean([o.prompt_logprobs[j][ids[j]].logprob for j in range(a, b)])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--max_model_len", type=int, default=6144)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.model.split("/")[-1]

    rows = load_rows(args.n, args.seed)
    catalog = load_catalog(CATALOG)
    extra = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "Qwen3.5" in args.model else {}
    llm = LLM(model=args.model, dtype=args.dtype, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, seed=args.seed, **extra)
    tok = llm.get_tokenizer()

    base = [r["prompt"][0]["content"] for r in rows]
    gold = [[tuple(g) for g in r["groundtruth_with_release_year"]] for r in rows]
    rng = random.Random(args.seed)
    partial_idx = [i for i, g in enumerate(gold) if len(g) >= 2]
    partial_hint = {i: [rng.choice(gold[i])] for i in partial_idx}
    conds = {
        "none": (list(range(len(rows))), lambda i: base[i], lambda i: []),
        "full": (list(range(len(rows))), lambda i: base[i] + FULL_HINT.format(items=fmt_items(gold[i])), lambda i: gold[i]),
        "partial": (partial_idx, lambda i: base[i] + PARTIAL_HINT.format(items=fmt_items(partial_hint[i])), lambda i: partial_hint[i]),
    }

    greedy = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=args.seed)
    summary = {"model": args.model, "backend": "vllm", "n_rows": len(rows), "seed": args.seed, "dtype": args.dtype,
               "tp": args.tp, "gpu": torch.cuda.get_device_name(0), "conditions": {}}
    records = []
    for cond, (idx, make, hints_of) in conds.items():
        t0 = time.time()
        texts = [build_chat(tok, make(i), args.model) for i in idx]
        outs = [o.outputs[0].text.strip() for o in llm.generate(texts, greedy, use_tqdm=False)]
        hints = [hints_of(i) for i in idx]
        summary["conditions"][cond] = score(cond, [rows[i] for i in idx], outs, catalog, hints)
        summary["conditions"][cond]["seconds"] = round(time.time() - t0, 1)
        print(f"[{tag}] {cond}: {summary['conditions'][cond]}", flush=True)
        for i, o in zip(idx, outs):
            records.append({"cond": cond, "row": i, "prompt_tail": make(i)[-400:], "gold": gold[i], "output": o})

    targets = [fmt_items(g) for g in gold]
    lp_none = gold_logprob(llm, tok, [build_chat(tok, base[i], args.model) for i in range(len(rows))], targets)
    lp_full = gold_logprob(llm, tok, [build_chat(tok, base[i] + FULL_HINT.format(items=fmt_items(gold[i])), args.model)
                                      for i in range(len(rows))], targets)
    summary["gold_logprob_per_token"] = {"none": float(np.mean(lp_none)), "full_hint": float(np.mean(lp_full)),
                                         "delta": float(np.mean(np.array(lp_full) - np.array(lp_none)))}

    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, f"{tag}_outputs.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
