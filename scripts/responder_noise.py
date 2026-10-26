"""Responder-noise diagnostic (checks the ReMaRk claim, arXiv 2610.17771) on our own policy.

Rank-GRPO trains on one row per (request, responder): the reward of a generated list depends on which responder's
items are the ground truth. ReMaRk reports that this draw explains ~85% of the advantage variance (Qwen3.5-2B SFT,
list-level NDCG@20) and proposes a union reward (credit for any responder's item, one row per request).

Protocol (same as ReMaRk's crs_consensus_check2.py + adv_variance.py): 500 training requests with >= 2 responders
(seed 42), G = 8 lists at temperature 1 from an SFT policy. For every request:
  S[r][g]      list-level NDCG@20 of list g against responder r            (ReMaRk's statistic)
  H[r][g][k]   Rank-GRPO exp_inf reward: hit of list g at rank k against responder r (the reward actually trained on)
  U[g][k]      union reward: hit at rank k against the union of all responders' items
Responder share of the advantage variance: a[r][g] = advantage when responder r is drawn, abar[g] = mean_r a[r][g];
share = sum (a - abar)^2 / sum a^2, pooled over requests (row-weighted = weighted by the number of responders, as
Rank-GRPO samples rows; request-weighted = uniform). For H the advantage is Rank-GRPO's: per rank, standardised over
the G lists of the group, summed over ranks. Also reported: share of (request, rank) groups with zero reward for
all G lists (no learning signal) under single vs union rewards.

Usage (GPU, env from run_full.sh setup):
  python scripts/responder_noise.py --model <sft ckpt> --out <dir> [--n_requests 500] [--dtype half]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from collections import OrderedDict

import numpy as np
from datasets import load_from_disk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "libs"))
from utils import process_rec_raw  # noqa: E402
from metrics_align import evaluate_direct_match_aligned  # noqa: E402

REC_NUM, G, EPS = 20, 8, 1e-4


def parse(text):
    err, item = process_rec_raw({"raw": text}, "raw", "recs")
    return [] if err else item["recs"]


def hits(recs, gts, seen, catalog):
    item = {"recs": recs, "gt": [list(x) for x in gts], "seen": list(seen)}
    return evaluate_direct_match_aligned(item, REC_NUM, "seen", "recs", "gt", catalog).astype(float)


def ndcg(h, num_gt):
    disc = 1.0 / np.log2(np.arange(REC_NUM) + 2)
    idcg = disc[: min(num_gt, REC_NUM)].sum()
    return float((h * disc).sum() / idcg) if idcg > 0 else 0.0


def resp_share_lists(S, scaled):
    """S: (R, G). Returns (responder part, total) of sum a^2 for one request."""
    a = S - S.mean(axis=1, keepdims=True)
    if scaled:
        sd = S.std(axis=1, keepdims=True)
        a = np.where(sd > 0, a / (sd + EPS), 0.0)
    abar = a.mean(axis=0, keepdims=True)
    return float(((a - abar) ** 2).sum()), float((a ** 2).sum())


def resp_share_ranks(H):
    """H: (R, G, K) per-rank hits. Rank-GRPO advantage per rank: (h - mean_g) / (std_g + eps) (torch std is unbiased)."""
    mu = H.mean(axis=1, keepdims=True)
    sd = H.std(axis=1, ddof=1, keepdims=True)
    a = (H - mu) / (sd + EPS)
    abar = a.mean(axis=0, keepdims=True)
    return float(((a - abar) ** 2).sum()), float((a ** 2).sum())


def pooled(parts, n, weighted):
    parts = np.asarray(parts, dtype=float)
    w = n if weighted else np.ones_like(n)
    resp, tot = parts[:, 0] / n, parts[:, 1] / n
    return float((w * resp).sum() / max((w * tot).sum(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_path", default=os.path.join(ROOT, "processed_datasets/grpo/grpo_dataset/train"))
    ap.add_argument("--catalog_path", default=os.path.join(ROOT, "reddit_catalog/gt_catalog.pkl"))
    ap.add_argument("--n_requests", type=int, default=500)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    catalog = set(pickle.load(open(args.catalog_path, "rb")))
    d = load_from_disk(args.train_path)
    groups = OrderedDict()
    for i, p in enumerate(d["prompt"]):
        groups.setdefault(p[0]["content"], []).append(i)
    multi = [k for k, v in groups.items() if len(v) >= 2]
    random.Random(42).shuffle(multi)
    chosen = multi[: args.n_requests]
    gts_all, seen_all = d["groundtruth_with_release_year"], d["seen_titles"]
    print(f"{len(groups)} requests, {len(multi)} with >= 2 responders; using {len(chosen)}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=4096, seed=42)
    sp = SamplingParams(n=G, temperature=1.0, top_p=1.0, max_tokens=1024, seed=42)
    outs = llm.chat([[{"role": "user", "content": c}] for c in chosen], sp)

    rec = {"lists": [], "S": [], "SU": [], "Hparts": [], "zero_single": 0, "zero_union": 0, "groups": 0,
           "req_zero_single": 0, "req_zero_union": 0}
    S_parts = {False: [], True: []}
    n_resp, per_req = [], []
    for ctx, out in zip(chosen, outs):
        idx = groups[ctx]
        lists = [parse(o.text) for o in out.outputs]
        resp_gts = [[tuple(x) for x in gts_all[i]] for i in idx]
        resp_seen = [set(seen_all[i]) for i in idx]
        H = np.stack([np.stack([hits(l, gt, sn, catalog) for l in lists]) for gt, sn in zip(resp_gts, resp_seen)])  # (R,G,K)
        S = np.array([[ndcg(H[r, g], len(resp_gts[r])) for g in range(G)] for r in range(len(idx))])
        union_gt = list(OrderedDict.fromkeys(x for gt in resp_gts for x in gt))
        union_seen = set().union(*resp_seen)
        U = np.stack([hits(l, union_gt, union_seen, catalog) for l in lists])  # (G,K)
        for scaled in (False, True):
            S_parts[scaled].append(resp_share_lists(S, scaled))
        rec["Hparts"].append(resp_share_ranks(H))
        # groups without signal: a (request, responder, rank) group is dead if all G hits are equal (here: all 0)
        zero_single = float((H.max(axis=1) == 0).mean())          # averaged over responders and ranks
        zero_union = float((U.max(axis=0) == 0).mean())           # averaged over ranks
        n_resp.append(len(idx))
        per_req.append({"n_responders": len(idx), "union_size": len(union_gt),
                        "single_hits_per_list": float(H.sum(axis=2).mean()), "union_hits_per_list": float(U.sum(axis=1).mean()),
                        "dead_rank_share_single": zero_single, "dead_rank_share_union": zero_union,
                        "S": S.round(4).tolist()})
    n = np.array(n_resp, dtype=float)
    summary = {
        "model": args.model, "requests": len(chosen), "mean_responders": round(float(n.mean()), 2), "G": G,
        "responder_share": {
            "list_ndcg_centred": {"row_weighted": pooled(S_parts[False], n, True), "request_weighted": pooled(S_parts[False], n, False)},
            "list_ndcg_group_scaled": {"row_weighted": pooled(S_parts[True], n, True), "request_weighted": pooled(S_parts[True], n, False)},
            "rankgrpo_per_rank_adv": {"row_weighted": pooled(rec["Hparts"], n, True), "request_weighted": pooled(rec["Hparts"], n, False)},
        },
        "dead_rank_group_share": {  # row-weighted, like Rank-GRPO's sampling
            "single": float(np.average([p["dead_rank_share_single"] for p in per_req], weights=n)),
            "union": float(np.average([p["dead_rank_share_union"] for p in per_req], weights=n)),
        },
        "hits_per_list": {"single": float(np.average([p["single_hits_per_list"] for p in per_req], weights=n)),
                          "union": float(np.average([p["union_hits_per_list"] for p in per_req], weights=n))},
        "reference_remark_qwen35_2B": {"list_ndcg_centred_row_weighted": 0.8379, "list_ndcg_group_scaled_row_weighted": 0.852},
    }
    json.dump(summary, open(os.path.join(args.out, "responder_noise_summary.json"), "w"), indent=1)
    with open(os.path.join(args.out, "responder_noise_per_request.jsonl"), "w") as f:
        for p in per_req:
            f.write(json.dumps(p) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "model"}, indent=1))


if __name__ == "__main__":
    main()
