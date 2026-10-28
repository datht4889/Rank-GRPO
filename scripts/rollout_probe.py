"""Rollout-scoring probe: does a hinted teacher's per-item signal on the student's OWN rollouts point at relevant items?

On-policy (self-)distillation rewards each sampled token with log p_T(y_t | x, hint, y_<t) - log p_S(y_t | x, y_<t).
This probe measures that signal per recommended item on Rank-GRPO-style rollouts:

  rollout : the student (released SFT checkpoint-1500, Qwen2.5-0.5B) samples G = 8 lists per request at temperature 1
            (TRL GRPO defaults: top_p 1, no top_k, no repetition penalty), like Rank-GRPO training.
  score   : each scorer (model, with or without the privileged hint) gives the summed log-prob of every item line of
            every rollout, teacher-forced on the student's tokens (each scorer uses its own tokenizer; items are mapped
            by character offsets, so summed log-probs are comparable across tokenizers).
  report  : A = lp_scorer - lp_student per item. Items are labelled with the per-reply ground truth of the row Rank-GRPO
            would train on (the "drawn" responder) and with the other responders' items of the same request:
              rev  : hit for the drawn responder (these items are revealed in the hint)
              oth  : hit only for another responder (relevant but NOT revealed) <- the generalisation test
              miss : no responder's item
            AUC(A) of rev vs miss and of oth vs miss, computed within request (pairs only inside a request, the
            relative comparison GRPO-style training uses), with a request bootstrap CI. "dead" restricts to
            (request, rank) groups where all G lists miss the drawn responder at that rank (no Rank-GRPO signal,
            where a distillation term would be added).

Requests: training requests with >= 2 responders (as in scripts/responder_noise.py), so "oth" labels exist.

Usage (stages are separate processes so each model frees the GPU):
  python rollout_probe.py rollout --student <sft ckpt> --out <dir>
  python rollout_probe.py score   --name qwen3_8b_hint --model Qwen/Qwen3-8B --hint --out <dir>
  python rollout_probe.py report  --out <dir>
"""
import argparse
import glob
import json
import os
import pickle
import random
import re
import sys
from collections import defaultdict

import numpy as np

REPO = os.environ.get("RANKGRPO_REPO", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Rank-GRPO"))
sys.path.insert(0, os.path.join(REPO, "libs"))
from utils import process_rec_raw  # noqa: E402
from metrics_align import evaluate_direct_match_aligned  # noqa: E402

TRAIN = os.environ.get("PROBE_TRAIN", os.path.join(REPO, "processed_datasets/grpo/grpo_dataset/train/data-00000-of-00001.arrow"))
CATALOG = os.path.join(REPO, "reddit_catalog/gt_catalog.pkl")
REC_NUM = 20
DEV = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else "cpu"  # CUDA_VISIBLE_DEVICES="" = CPU smoke test
# neutral wording (no "must include"): the teacher should judge the student's list, not be told to copy
HINT = ("\n\nPrivileged hint: after this conversation, the user said they liked the following movies:\n{items}\n"
        "Use this to recommend 20 movies the user would like.")
# v2: v1 reads like movies the user already watched, which the SFT policy learned never to recommend again
HINT_V2 = ("\n\nFor reference, other people recommended these movies for this request (the user has not watched them "
           "yet):\n{items}\nUse them to infer what the user wants and recommend 20 movies.")
HINTS = {"v1": HINT, "v2": HINT_V2}


def fmt_items(items):
    return "\n".join(f"{t} ({y})" for t, y in items)


def load_requests(n, seed):
    import pyarrow as pa
    with pa.memory_map(TRAIN) as f:
        tab = pa.ipc.open_stream(f).read_all()
    prompts = [p[0]["content"] for p in tab.column("prompt").to_pylist()]
    gts = tab.column("groundtruth_with_release_year").to_pylist()
    seen = tab.column("seen_titles").to_pylist()
    groups = defaultdict(list)
    for i, p in enumerate(prompts):
        if gts[i]:
            groups[p].append(i)
    multi = sorted(k for k, v in groups.items() if len(v) >= 2)
    rng = random.Random(seed)
    rng.shuffle(multi)
    reqs = []
    for k in multi[:n]:
        idx = groups[k]
        d = rng.choice(idx)
        reqs.append({"content": k,
                     "drawn_gt": [list(x) for x in gts[d]], "drawn_seen": list(seen[d]),
                     "union_gt": [list(x) for i in idx for x in gts[i]],
                     "union_seen": sorted(set().union(*[set(seen[i]) for i in idx])),
                     "n_responders": len(idx)})
    print(f"{len(groups)} requests, {len(multi)} with >= 2 responders, using {len(reqs)}", flush=True)
    return reqs


def build_chat(tok, content, model_name):
    kw = {"enable_thinking": False} if ("Qwen3" in model_name or "gemma-4" in model_name) else {}
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, **kw)


def parse_items(text):
    """Item lines in order with their character spans; same parsing as libs.utils.process_rec_raw on the full text."""
    items, pos = [], 0
    for line in text.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        if not line.strip():
            continue
        err, it = process_rec_raw({"raw": line}, "raw", "recs")
        if not err and len(it["recs"]) == 1:
            t, y = it["recs"][0]
            items.append([t, y, start, end])
    return items


def label(items, req, catalog):
    recs = [(t, y) for t, y, _, _ in items]
    h_d = evaluate_direct_match_aligned({"recs": recs, "gt": req["drawn_gt"], "seen": req["drawn_seen"]},
                                        REC_NUM, "seen", "recs", "gt", catalog)
    h_u = evaluate_direct_match_aligned({"recs": recs, "gt": req["union_gt"], "seen": req["union_seen"]},
                                        REC_NUM, "seen", "recs", "gt", catalog)
    return [int(x) for x in h_d], [int(x) for x in h_u]


def load_model(name, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except (OSError, ValueError):  # a checkpoint saved without tokenizer files: Rank-GRPO's base model tokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dt, device_map="auto" if DEV == "cuda" else "cpu").eval()
    return tok, model


def stage_rollout(a):
    import torch
    reqs = load_requests(a.n_requests, a.seed)
    tok, model = load_model(a.student, a.dtype)
    texts = [build_chat(tok, r["content"], a.student) for r in reqs]
    torch.manual_seed(a.seed)
    catalog = set(pickle.load(open(CATALOG, "rb")))
    out = open(os.path.join(a.out, "rollouts.jsonl"), "w", encoding="utf-8")
    for i in range(0, len(texts), a.gen_bs):
        enc = tok(texts[i:i + a.gen_bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                 num_return_sequences=a.G, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
        L = enc["input_ids"].shape[1]
        for j, row in enumerate(gen[:, L:]):
            q = i + j // a.G
            text = tok.decode(row, skip_special_tokens=True)
            items = parse_items(text)
            h_d, h_u = label(items, reqs[q], catalog)
            out.write(json.dumps({"req": q, "g": j % a.G, "text": text, "items": items, "hit_drawn": h_d, "hit_union": h_u},
                                 ensure_ascii=False) + "\n")
        print(f"  rollouts {min(i + a.gen_bs, len(texts))}/{len(texts)}", flush=True)
    out.close()
    json.dump(reqs, open(os.path.join(a.out, "requests.json"), "w", encoding="utf-8"), ensure_ascii=False)


def stage_score(a):
    import torch
    reqs = json.load(open(os.path.join(a.out, "requests.json"), encoding="utf-8"))
    rolls = [json.loads(l) for l in open(os.path.join(a.out, "rollouts.jsonl"), encoding="utf-8")]
    tok, model = load_model(a.model, a.dtype)
    seqs = []
    for r in rolls:
        req = reqs[r["req"]]
        content = req["content"] + (HINTS[a.hint_style].format(items=fmt_items(req["drawn_gt"])) if a.hint else "")
        p_ids = tok(build_chat(tok, content, a.model), add_special_tokens=False)["input_ids"]
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        r_ids, offs = enc["input_ids"][:a.max_new_tokens], enc["offset_mapping"][:a.max_new_tokens]
        seqs.append((p_ids, r_ids, offs))
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][0]) + len(seqs[k][1]))  # length-bucketed batches
    res = [None] * len(seqs)
    pad = tok.pad_token_id
    for b0 in range(0, len(order), a.score_bs):
        ks = order[b0:b0 + a.score_bs]
        ids = [seqs[k][0] + seqs[k][1] for k in ks]
        L, R = max(map(len, ids)), max(len(seqs[k][1]) for k in ks)
        inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device=DEV)
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device=DEV)
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=att, logits_to_keep=R + 1).logits  # last R+1 positions
        for bi, k in enumerate(ks):
            r_ids = seqs[k][1]
            n = len(r_ids)
            if n == 0:
                res[k] = []
                continue
            pred = logits[bi, -(n + 1):-1].float()  # position t predicts token t+1: the n response tokens
            lp = torch.log_softmax(pred, -1).gather(-1, torch.tensor(r_ids, device=pred.device)[:, None])[:, 0].cpu().numpy()
            res[k] = lp
        del logits
        if b0 // a.score_bs % 20 == 0:
            print(f"  scored {min(b0 + a.score_bs, len(order))}/{len(order)}", flush=True)
    item_lp = []
    for k, r in enumerate(rolls):
        offs, lp = seqs[k][2], res[k]
        starts = np.array([o[0] for o in offs]) if len(offs) else np.zeros(0)
        per = []
        for _, _, cs, ce in r["items"]:
            m = (starts >= cs) & (starts < ce)
            per.append([float(lp[m].sum()), int(m.sum())])
        item_lp.append({"items": per, "total": float(np.sum(lp)) if len(lp) else 0.0, "n_tok": len(lp)})
    json.dump({"name": a.name, "model": a.model, "hint": a.hint and a.hint_style, "dtype": a.dtype, "item_lp": item_lp},
              open(os.path.join(a.out, f"score_{a.name}.json"), "w"))


def auc_within(score, pos, neg, grp):
    """Pairs (positive, negative) only inside the same group; returns (AUC, n_pairs)."""
    num = den = 0.0
    for g in np.unique(grp[pos]):
        p, n = score[pos & (grp == g)], np.sort(score[neg & (grp == g)])
        if len(n) == 0:
            continue
        num += (np.searchsorted(n, p, "left") + np.searchsorted(n, p, "right")).sum() / 2.0
        den += len(p) * len(n)
    return (num / den if den else float("nan")), int(den)


def boot_ci(score, pos, neg, reqs_of, B=300, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(reqs_of)
    idx_by = {u: np.where(reqs_of == u)[0] for u in uniq}
    vals = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq))
        ii = np.concatenate([idx_by[u] for u in pick])
        slot = np.concatenate([np.full(len(idx_by[u]), t) for t, u in enumerate(pick)])  # a request drawn twice = 2 groups
        v, _ = auc_within(score[ii], pos[ii], neg[ii], slot)
        if not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def stage_report(a):
    rolls = [json.loads(l) for l in open(os.path.join(a.out, "rollouts.jsonl"), encoding="utf-8")]
    scores = {}
    for f in sorted(glob.glob(os.path.join(a.out, "score_*.json"))):
        s = json.load(open(f))
        scores[s["name"]] = s
    assert "student" in scores, "the student scorer (no hint) is the reference"
    G = max(r["g"] for r in rolls) + 1
    # dead (request, rank) groups: every list of the request misses the drawn responder at that rank
    hit_at = defaultdict(int)
    for r in rolls:
        for k, h in enumerate(r["hit_drawn"]):
            hit_at[(r["req"], k)] |= h
    rows = []  # one per ranked item (first 20 parsed items)
    for ri, r in enumerate(rolls):
        for k in range(min(len(r["items"]), REC_NUM)):
            rows.append((ri, r["req"], k, r["hit_drawn"][k], r["hit_union"][k], int(hit_at[(r["req"], k)] == 0)))
    rows = np.array(rows)
    ri, req, rank, hd, hu, dead = rows.T
    rev, oth, miss = hd == 1, (hu == 1) & (hd == 0), hu == 0

    def lp(name, per_token=False):
        v = np.array([scores[name]["item_lp"][i]["items"][k] for i, k in zip(ri, rank)])
        return v[:, 0] / np.maximum(v[:, 1], 1) if per_token else v[:, 0]

    lp_s = lp("student")
    summary = {"G": G, "requests": int(len(np.unique(req))), "lists": len(rolls),
               "ranked_items": int(len(rows)), "rev": int(rev.sum()), "oth": int(oth.sum()), "miss": int(miss.sum()),
               "dead_rank_group_share": float(np.mean([v == 0 for v in hit_at.values()])),
               "items_in_dead_groups": {"oth": int((oth & (dead == 1)).sum()), "miss": int((miss & (dead == 1)).sum())},
               "mean_items_per_list": float(np.mean([len(r["items"]) for r in rolls])),
               "scorers": {}}
    for name in scores:
        if name == "student":
            sig = {"lp": lp_s}
        else:
            sig = {"A=lp-lp_student": lp(name) - lp_s, "lp": lp(name),
                   "A_per_token": lp(name, True) - lp("student", True)}
        out = {}
        for sname, sc in sig.items():
            m = {}
            for tag, pos, neg, sub in (("rev_vs_miss", rev, miss, None), ("oth_vs_miss", oth, miss, None),
                                       ("oth_vs_miss_dead", oth, miss, dead == 1)):
                p, n = (pos, neg) if sub is None else (pos & sub, neg & sub)
                auc, pairs = auc_within(sc, p, n, req)
                m[tag] = {"auc": auc, "pairs": pairs}
                if tag.startswith("oth"):
                    m[tag]["ci95"] = boot_ci(sc, p, n, req, B=a.boot)
            m["mean"] = {"rev": float(sc[rev].mean()) if rev.any() else None,
                         "oth": float(sc[oth].mean()) if oth.any() else None, "miss": float(sc[miss].mean())}
            out[sname] = m
        summary["scorers"][name] = {"model": scores[name]["model"], "hint": scores[name]["hint"], **out}
    json.dump(summary, open(os.path.join(a.out, "rollout_probe_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "scorers"}, indent=1))
    print(f"\n{'scorer':<20}{'signal':<18}{'rev/miss':>9}{'oth/miss':>9}{'  95% CI':>16}{'oth/miss dead':>14}")
    for name, s in summary["scorers"].items():
        for sname, m in s.items():
            if not isinstance(m, dict) or "rev_vs_miss" not in m:
                continue
            ci = m["oth_vs_miss"]["ci95"]
            print(f"{name:<20}{sname:<18}{m['rev_vs_miss']['auc']:>9.3f}{m['oth_vs_miss']['auc']:>9.3f}"
                  f"   [{ci[0]:.3f},{ci[1]:.3f}]{m['oth_vs_miss_dead']['auc']:>14.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rollout", "score", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--student")
    ap.add_argument("--model")
    ap.add_argument("--name")
    ap.add_argument("--hint", action="store_true")
    ap.add_argument("--hint_style", default="v1", choices=list(HINTS))
    ap.add_argument("--n_requests", type=int, default=300)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen_bs", type=int, default=8, help="prompts per generate call (x G sequences)")
    ap.add_argument("--score_bs", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"rollout": stage_rollout, "score": stage_score, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
