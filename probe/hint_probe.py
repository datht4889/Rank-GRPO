"""Privileged-hint probe: can small Qwen models use gold items placed in the prompt?

For each sampled REDDIT-V2 test row we query the model under three conditions:
  none    : original Rank-GRPO prompt (zero-shot baseline)
  full    : prompt + all gold movies of the row, with an instruction to include them
  partial : prompt + one gold movie (rows with >= 2 gold items); scored on the remaining gold items
We also measure teacher-forced log p(gold list | prompt) with and without the full hint, which is
the quantity an on-policy self-distillation teacher would provide.

Usage: python probe/hint_probe.py --model Qwen/Qwen3-0.6B --n 200
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow as pa
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Rank-GRPO repo (for libs/ and the catalog) and the test split; override via env on Kaggle
REPO = os.environ.get("RANKGRPO_REPO", os.path.join(ROOT, "Rank-GRPO"))
sys.path.insert(0, REPO)
from libs.utils import process_rec_raw  # noqa: E402
from libs.metrics import evaluate_direct_match_truncate  # noqa: E402
from libs.data import load_catalog  # noqa: E402

DATA = os.environ.get("PROBE_DATA", os.path.join(REPO, "processed_datasets", "sft_dataset", "test", "data-00000-of-00001.arrow"))
CATALOG = os.path.join(REPO, "reddit_catalog", "gt_catalog.pkl")

FULL_HINT = (
    "\n\nPrivileged hint: after this conversation, the user said they liked the following movies:\n{items}\n"
    "Your 20 recommendations must include all of these movies, followed by other movies the user would also like."
)
PARTIAL_HINT = (
    "\n\nPrivileged hint: after this conversation, the user said they liked this movie:\n{items}\n"
    "Use it to infer the user's taste and recommend 20 movies the user would like."
)


def load_rows(n, seed):
    with pa.memory_map(DATA) as f:
        rows = pa.ipc.open_stream(f).read_all().to_pylist()
    by_prompt = defaultdict(list)
    for r in rows:
        if r["groundtruth_with_release_year"]:
            by_prompt[r["prompt"][0]["content"]].append(r)
    rng = random.Random(seed)
    keys = sorted(by_prompt)
    rng.shuffle(keys)
    return [rng.choice(by_prompt[k]) for k in keys[:n]]


def fmt_items(items):
    return "\n".join(f"{t} ({y})" for t, y in items)


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def contains(recs, gold):
    """Rank (1-based) of `gold` in `recs` by title + year within 2, ignoring catalog membership."""
    for i, (t, y) in enumerate(recs):
        if norm(t) == norm(gold[0]) and abs(int(y) - int(gold[1])) <= 2:
            return i + 1
    return None


def build_chat(tok, content, model_name, thinking=False):
    msgs = [{"role": "user", "content": content}]
    kw = {}
    if "Qwen3" in model_name or "gemma-4" in model_name:  # Qwen3 / Qwen3.5 / Gemma 4: non-thinking unless --thinking
        kw["enable_thinking"] = thinking
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def load_model(name, dtype, device_map="auto", nf4=False):
    # device_map="auto" spreads larger models (7B+) over all visible GPUs; pass "cuda:0" for small ones
    # (auto can split a tiny model with tied embeddings across GPUs, e.g. gemma-3-270m)
    kw = {}
    if nf4:  # 4-bit weights, compute in `dtype` (fp32 on T4 avoids Gemma's fp16 overflow and bf16 emulation)
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device_map, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype, device_map=device_map, **kw)
    return model.eval()


@torch.no_grad()
def generate(model, tok, texts, bs, max_new_tokens, thinking=False):
    """Returns (answers, thoughts). Thinking mode samples with Qwen3's recommended settings (greedy loops in
    thinking mode); an answer whose thinking never closed within max_new_tokens is empty."""
    sample = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if thinking else \
        dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    outs, thoughts = [], []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **sample)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            s = tok.decode(row, skip_special_tokens=True)
            if thinking:
                head, sep, tail = s.partition("</think>")
                thoughts.append(head.replace("<think>", "").strip() if sep else None)
                outs.append(tail.strip())
            else:
                outs.append(re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip())
        print(f"  generated {min(i + bs, len(texts))}/{len(texts)}", flush=True)
    return outs, thoughts


@torch.no_grad()
def gold_logprob(model, tok, prompts, targets, bs):
    """Mean per-token log p(target | prompt) for each pair."""
    res = []
    for i in range(0, len(prompts), bs):
        ids, labels = [], []
        for p, t in zip(prompts[i:i + bs], targets[i:i + bs]):
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            t_ids = tok(t, add_special_tokens=False)["input_ids"]
            ids.append(p_ids + t_ids)
            labels.append([-100] * len(p_ids) + t_ids)
        L = max(map(len, ids))
        pad = tok.pad_token_id
        inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device="cuda")
        lab = torch.tensor([[-100] * (L - len(x)) + x for x in labels], device="cuda")
        att = (torch.arange(L, device="cuda")[None] >= (L - torch.tensor([len(x) for x in ids], device="cuda"))[:, None]).long()
        logits = model(input_ids=inp, attention_mask=att).logits[:, :-1]
        tgt = lab[:, 1:]
        for b in range(logits.size(0)):  # full-vocab softmax only on target positions to save memory
            pos = tgt[b] != -100
            lp = torch.log_softmax(logits[b][pos].float(), -1).gather(-1, tgt[b][pos][:, None])[:, 0]
            res.append(lp.mean().item())
        del logits
        torch.cuda.empty_cache()
    return res


def score(cond, rows, outputs, catalog, hint_items):
    recalls, ndcgs, copy_rates, hit_ranks, n_items, exact20, in_cat, dup = [], [], [], [], [], [], [], []
    lenient = []
    for r, out, hints in zip(rows, outputs, hint_items):
        item = {"raw": out, "seen": r["seen_titles"], "gt": r["groundtruth_with_release_year"]}
        _, item = process_rec_raw(item, "raw", "recs")
        recs = item["recs"]
        n_items.append(len(recs))
        exact20.append(len(recs) == 20)
        top = recs[:20]
        in_cat.append(np.mean([rec in catalog for rec in top]) if top else 0.0)
        dup.append(1 - len(set(top)) / len(top) if top else 0.0)
        gt = item["gt"]
        if cond == "partial":  # score only on gold items that were NOT revealed
            gt = [g for g in gt if tuple(g) not in {tuple(h) for h in hints}]
        rec, ndcg = evaluate_direct_match_truncate({**item, "gt": gt}, 20, "seen", "recs", "gt", catalog)
        recalls.append(rec)
        ndcgs.append(ndcg)
        if cond == "full":
            ranks = [contains(top, g) for g in hints]
            copy_rates.append(np.mean([x is not None for x in ranks]))
            hit_ranks.extend(x for x in ranks if x is not None)
            # format-agnostic: gold title appears anywhere in the raw output
            lenient.append(np.mean([re.search(r"(?<!\w)" + re.escape(norm(g[0])) + r"(?!\w)", norm(out)) is not None
                                    for g in hints]))
    s = {
        "n": len(rows),
        "recall@20": float(np.mean(recalls)),
        "ndcg@20": float(np.mean(ndcgs)),
        "mean_items": float(np.mean(n_items)),
        "pct_exactly_20": float(np.mean(exact20)),
        "pct_in_catalog@20": float(np.mean(in_cat)),
        "pct_duplicate@20": float(np.mean(dup)),
    }
    if cond == "full":
        s["copy_rate"] = float(np.mean(copy_rates))
        s["copy_rate_lenient"] = float(np.mean(lenient))
        s["pct_all_copied"] = float(np.mean([c == 1.0 for c in copy_rates]))
        s["mean_rank_of_copied"] = float(np.mean(hit_ranks)) if hit_ranks else None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "probe", "results"))
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32", "nf4"],
                    help="auto = bf16 if the GPU supports it, else fp16 (T4)")
    ap.add_argument("--device_map", default="auto", help='"auto" (split over GPUs) or e.g. "cuda:0"')
    ap.add_argument("--thinking", action="store_true",
                    help="Qwen3 thinking mode (sampled); skips the teacher-forced log-prob part")
    ap.add_argument("--conds", default="none,full,partial", help="conditions to run, in this order")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.model.split("/")[-1] + ("_thinking" if args.thinking else "")

    rows = load_rows(args.n, args.seed)
    catalog = load_catalog(CATALOG)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16  # T4 = sm75
    else:
        # nf4 = bitsandbytes 4-bit weights with fp32 compute
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "nf4": torch.float32}[args.dtype]
    model = load_model(args.model, dtype, args.device_map, nf4=args.dtype == "nf4")

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

    summary = {"model": args.model, "backend": "hf", "n_rows": len(rows), "seed": args.seed, "dtype": "nf4 weights, fp32 compute" if args.dtype == "nf4" else str(dtype),
               "gpu": torch.cuda.get_device_name(0), "thinking": args.thinking, "max_new_tokens": args.max_new_tokens,
               "conditions": {}}
    records = []

    def dump():  # write after every condition, so a timeout still leaves the finished ones
        with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.out_dir, f"{tag}_outputs.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for cond in args.conds.split(","):
        idx, make, hints_of = conds[cond]
        t0 = time.time()
        print(f"[{tag}] condition={cond} ({len(idx)} rows)", flush=True)
        texts = [build_chat(tok, make(i), args.model, args.thinking) for i in idx]
        outs, thoughts = generate(model, tok, texts, args.bs, args.max_new_tokens, args.thinking)
        sub_rows = [rows[i] for i in idx]
        hints = [hints_of(i) for i in idx]
        summary["conditions"][cond] = score(cond, sub_rows, outs, catalog, hints)
        summary["conditions"][cond]["seconds"] = round(time.time() - t0, 1)
        if args.thinking:
            done = [t for t in thoughts if t is not None]
            summary["conditions"][cond]["pct_thinking_unfinished"] = 1 - len(done) / len(thoughts)
            summary["conditions"][cond]["mean_thinking_tokens"] = (
                float(np.mean([len(tok(t, add_special_tokens=False)["input_ids"]) for t in done])) if done else None)
        for k, (i, o) in enumerate(zip(idx, outs)):
            records.append({"cond": cond, "row": i, "prompt_tail": make(i)[-400:], "gold": gold[i], "output": o,
                            **({"thinking": thoughts[k]} if args.thinking else {})})
        dump()

    if args.thinking:  # teacher-forced log-prob of a bare list is not meaningful after a thinking prompt
        print(json.dumps(summary, indent=2))
        return

    # Teacher-forced log p(gold list | prompt), with vs without the full hint
    targets = [fmt_items(g) for g in gold]
    lp_none = gold_logprob(model, tok, [build_chat(tok, base[i], args.model) for i in range(len(rows))], targets, 2)
    lp_full = gold_logprob(model, tok, [build_chat(tok, base[i] + FULL_HINT.format(items=fmt_items(gold[i])), args.model)
                                        for i in range(len(rows))], targets, 2)
    summary["gold_logprob_per_token"] = {
        "none": float(np.mean(lp_none)),
        "full_hint": float(np.mean(lp_full)),
        "delta": float(np.mean(np.array(lp_full) - np.array(lp_none))),
    }
    dump()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
