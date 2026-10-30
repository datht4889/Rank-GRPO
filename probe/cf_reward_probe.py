"""CPU probe: is an item-item CF score a usable partial-credit reward for Rank-GRPO on Reddit-v2?

The reward knows the drawn responder's ground truth (and the movies mentioned in the conversation). A CF partial reward
would score every recommended item by its similarity to those items. It is useful only if it ranks the items other
responders named (relevant but unlabelled: false negatives) above plain misses, and if it cannot be gamed by listing
popular movies.

CF models are fitted on Reddit-v2 GRPO train requests (pseudo-user = request, items = union of responders' ground
truth + catalog movies mentioned in the conversation), excluding the 300 rollout-probe requests:
  pop      item popularity (no query)
  cos      item-kNN cosine on co-occurrence
  pmi      positive PMI on co-occurrence (popularity-normalised)
  ease_L   EASE (Steck 2019), closed form, L2 = L
  *_b0.5   score / pop^0.5 (popularity debiasing)
Query at reward time: drawn responder's gt (+ seen items); score(x) = sum_q W[q, x].

Tests
  A rollout  : the student's own rollouts (SFT-1500, G = 8, 300 train requests with >= 2 responders, from
               crs-rollout-probe). Within-request AUC of other-responder hits ("oth") vs misses, all cells and dead
               (request, rank) cells; vs and combined with the student's own log-prob (z-scored within request).
  B val LORO : validation requests with >= 2 responders, leave-one-responder-out over the whole catalog: query = one
               responder (+ seen), targets = the other responders' items; AUC and R@20 / R@50.
  C hacking  : percentile (within request, over the catalog) of the CF score of: oth items, student misses, and a fixed
               top-20-popular list. If the popular list scores like oth items the reward is gameable.

Usage: python -I probe/cf_reward_probe.py [--models ...] [--seen 0|1]
"""
import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow as pa

CRS = os.environ.get("CRS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RG = f"{CRS}/Rank-GRPO"
ROLL = f"{CRS}/probe/results_rollout_v1/rollout-probe-v1-sft05b-g8-teachers-self-rl-qwen3-4b-8b-n300"
OUT = f"{CRS}/probe/results_cf_v1"


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def load(p):
    with pa.memory_map(p) as f:
        return pa.ipc.open_stream(f).read_all()


class Vocab:
    """Catalog ids; gt items match by normalised title and year within 2 (as the eval), seen titles by title."""

    def __init__(self, catalog):
        self.items = sorted(catalog)
        self.exact = {it: i for i, it in enumerate(self.items)}
        self.by_title = defaultdict(list)
        for i, (t, y) in enumerate(self.items):
            self.by_title[norm(t)].append((int(y), i))

    def gt(self, t, y):
        try:
            y = int(str(y))
        except ValueError:
            return None
        c = [(abs(yy - y), i) for yy, i in self.by_title.get(norm(t), []) if abs(yy - y) <= 2]
        return min(c)[1] if c else None

    def seen(self, t):
        return [i for _, i in self.by_title.get(norm(t), [])]

    def rec(self, t, y):
        return self.exact.get((t, int(y)))


def requests_from(table, voc, use_seen):
    prompts = [p[0]["content"] for p in table.column("prompt").to_pylist()]
    gts = table.column("groundtruth_with_release_year").to_pylist()
    seen = table.column("seen_titles").to_pylist()
    req = defaultdict(lambda: {"resp": [], "seen": set()})
    for p, g, s in zip(prompts, gts, seen):
        ids = [i for i in (voc.gt(t, y) for t, y in g) if i is not None]
        if ids:
            req[p]["resp"].append(sorted(set(ids)))
        if use_seen:
            req[p]["seen"].update(i for t in s for i in voc.seen(t))
    return req


def cooc(reqs, n):
    t0 = time.time()
    C = np.zeros((n, n), np.float32)
    pop = np.zeros(n, np.float32)
    for r in reqs.values():
        ids = np.array(sorted(set(i for g in r["resp"] for i in g) | r["seen"]), np.int64)
        if len(ids) == 0:
            continue
        pop[ids] += 1
        C[np.ix_(ids, ids)] += 1
    N = float(len(reqs))
    print(f"co-occurrence {n}x{n} from {int(N)} requests in {time.time() - t0:.0f}s", flush=True)
    return C, pop, N


def build(C, pop, N, m):
    n = C.shape[0]
    if True:
        if m == "cos":
            d = np.sqrt(np.maximum(np.diag(C), 1))
            w = C / d[:, None] / d[None, :]
        elif m == "pmi":
            with np.errstate(divide="ignore"):
                w = np.log(np.maximum(C, 1e-9) * N / np.maximum(pop[:, None] * pop[None, :], 1))
            w = np.where(C >= 2, np.maximum(w, 0), 0).astype(np.float32)
        elif m.startswith("ease_"):
            lam = float(m.split("_")[1])
            P = C.copy()
            P[np.diag_indices(n)] += lam
            P = np.linalg.inv(P)  # float32 to fit in RAM
            d = np.diag(P).copy()
            P /= -d[None, :]
            w = P
        else:
            raise ValueError(m)
        np.fill_diagonal(w, 0)
        return w.astype(np.float32, copy=False)


def scorer(w, pop, m):
    """name -> function(query ids) -> score over the catalog."""
    base, _, beta = m.partition("_b")
    beta = float(beta) if beta else 0.0
    if base == "pop":
        return lambda q: pop.astype(np.float64)
    damp = np.maximum(pop, 1).astype(np.float64) ** beta

    def f(q):
        s = w[q].sum(0).astype(np.float64) if len(q) else np.zeros(w.shape[1])
        return s / damp
    return f


def auc_within(score, pos, neg, grp):
    num = den = 0.0
    for g in np.unique(grp[pos]):
        p, n = score[pos & (grp == g)], np.sort(score[neg & (grp == g)])
        if len(n) == 0:
            continue
        num += (np.searchsorted(n, p, "left") + np.searchsorted(n, p, "right")).sum() / 2.0
        den += len(p) * len(n)
    return (num / den if den else float("nan")), int(den)


def boot(score, pos, neg, grp, B=200, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(grp)
    by = {u: np.where(grp == u)[0] for u in uniq}
    v = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq))
        ii = np.concatenate([by[u] for u in pick])
        slot = np.concatenate([np.full(len(by[u]), k) for k, u in enumerate(pick)])
        a, _ = auc_within(score[ii], pos[ii], neg[ii], slot)
        if not np.isnan(a):
            v.append(a)
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def zwithin(x, grp):
    z = np.zeros_like(x, dtype=np.float64)
    for g in np.unique(grp):
        m = grp == g
        sd = x[m].std()
        z[m] = (x[m] - x[m].mean()) / (sd if sd > 0 else 1)
    return z


def pctile(s, x):
    """Percentile of s[x] among the catalog (ties = mid rank)."""
    srt = np.sort(s)
    return (np.searchsorted(srt, s[x], "left") + np.searchsorted(srt, s[x], "right")) / 2.0 / len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pop", "cos", "pmi", "ease_50", "ease_200", "ease_1000",
                                                    "cos_b0.5", "ease_200_b0.5"])
    ap.add_argument("--seen", type=int, default=1, help="add conversation-mentioned movies to pseudo-users / queries")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    voc = Vocab(pickle.load(open(f"{RG}/reddit_catalog/gt_catalog.pkl", "rb")))
    n = len(voc.items)

    probe_reqs = json.load(open(f"{ROLL}/requests.json", encoding="utf-8"))
    probe_prompts = {r["content"] for r in probe_reqs}
    train = requests_from(load(f"{RG}/processed_datasets/grpo/grpo_dataset/train/data-00000-of-00001.arrow"), voc, a.seen)
    fit_reqs = {k: v for k, v in train.items() if k not in probe_prompts}
    print(f"train requests {len(train)}, fit on {len(fit_reqs)} (probe requests held out)", flush=True)
    C, pop, N = cooc(fit_reqs, n)
    del train, fit_reqs
    top_pop = np.argsort(-pop)[:20]
    res = {"n_items": n, "seen": a.seen, "models": a.models}

    # ---------- A setup: student rollouts ----------
    rolls = [json.loads(l) for l in open(f"{ROLL}/rollouts.jsonl", encoding="utf-8")]
    lp = json.load(open(f"{ROLL}/score_student.json"))["item_lp"]
    hit_at = defaultdict(int)
    for r in rolls:
        for k, h in enumerate(r["hit_drawn"]):
            hit_at[(r["req"], k)] |= h
    rows, qcache = [], {}
    for ri, r in enumerate(rolls):
        pr = probe_reqs[r["req"]]
        if r["req"] not in qcache:
            q = [i for i in (voc.gt(t, y) for t, y in pr["drawn_gt"]) if i is not None]
            if a.seen:  # conversation movies (seen titles of the drawn row)
                q += [i for t in pr["drawn_seen"] for i in voc.seen(t)]
            qcache[r["req"]] = sorted(set(q))
        for k in range(min(len(r["items"]), 20)):
            t, y = r["items"][k][:2]
            rows.append((ri, r["req"], k, r["hit_drawn"][k], r["hit_union"][k], int(hit_at[(r["req"], k)] == 0),
                         -1 if voc.rec(t, y) is None else voc.rec(t, y), lp[ri]["items"][k][0]))
    rows = np.array(rows, dtype=np.float64)
    ri, req, rank, hd, hu, dead, iid, lps = rows.T
    req, iid, dead = req.astype(int), iid.astype(int), dead.astype(int)
    oth, miss = (hu == 1) & (hd == 0), hu == 0
    incat = iid >= 0
    A = {"items": int(len(rows)), "oth": int(oth.sum()), "miss": int(miss.sum()),
         "share_in_catalog": round(float(incat.mean()), 3), "share_miss_in_catalog": round(float(incat[miss].mean()), 3),
         "dead_cells_share": round(float(np.mean([v == 0 for v in hit_at.values()])), 3), "scorers": {}}
    zl = zwithin(lps, req)

    def eval_A(name, s):
        d = {}
        for tag, sub in (("all", None), ("dead", dead == 1)):
            p, ng = (oth, miss) if sub is None else (oth & sub, miss & sub)
            au, pairs = auc_within(s, p, ng, req)
            d[tag] = {"auc": round(au, 3), "pairs": pairs, "ci95": boot(s, p, ng, req, a.boot)}
        A["scorers"][name] = d
        print(f"A {name:<22} oth/miss {d['all']['auc']:.3f} {d['all']['ci95']}  dead {d['dead']['auc']:.3f} "
              f"{d['dead']['ci95']}", flush=True)

    eval_A("student_lp", lps)

    # ---------- B/C setup: validation leave-one-responder-out ----------
    val = requests_from(load(f"{RG}/processed_datasets/sft_dataset/validation/data-00000-of-00001.arrow"), voc, a.seen)
    rng = random.Random(0)
    vq = []
    for v in (v for v in val.values() if len(v["resp"]) >= 2):
        d = rng.randrange(len(v["resp"]))
        q = sorted(set(v["resp"][d]) | v["seen"])
        tgt = sorted(set(i for k, g in enumerate(v["resp"]) if k != d for i in g) - set(q))
        if tgt:
            vq.append((q, tgt))
    del val
    res["B_val_loro"] = {"requests": len(vq)}
    res["C_hacking_pctile"] = {}

    for base in ["pop"] + sorted({m.partition("_b")[0] for m in a.models} - {"pop"}):
        group = [m for m in a.models if m.partition("_b")[0] == base]
        if not group:
            continue
        t0 = time.time()
        w = None if base == "pop" else build(C, pop, N, base)
        for m in group:
            f = scorer(w, pop, m)
            s = np.full(len(rows), np.nan)
            sc = {g: f(qcache[g]) for g in np.unique(req)}
            for j in np.where(incat)[0]:
                s[j] = sc[req[j]][iid[j]]
            s = np.where(np.isnan(s), np.nanmin(s) - 1, s)  # items outside the catalog never get credit
            eval_A(m, s)
            eval_A(m + "+lp", zwithin(s, req) + zl)
            B = {"auc": [], "r20": [], "r50": []}
            Cc = {"oth": [], "pop20": []}
            for q, tgt in vq:
                sv = f(q)
                cand = np.ones(n, bool)
                cand[q] = False
                tm = np.zeros(n, bool)
                tm[tgt] = True
                scv, t = sv[cand], tm[cand]
                top = t[np.argsort(-scv, kind="stable")]
                B["r20"].append(top[:20].sum() / t.sum())
                B["r50"].append(top[:50].sum() / t.sum())
                srt = np.sort(scv[~t])
                B["auc"].append(float(((np.searchsorted(srt, scv[t], "left") + np.searchsorted(srt, scv[t], "right"))
                                       / 2 / len(srt)).mean()))
                Cc["oth"].append(float(pctile(sv, np.array(tgt)).mean()))
                Cc["pop20"].append(float(pctile(sv, top_pop[~np.isin(top_pop, q)]).mean()))
            b = res["B_val_loro"][m] = {k: round(float(np.mean(x)), 4) for k, x in B.items()}
            c = res["C_hacking_pctile"][m] = {k: round(float(np.mean(x)), 3) for k, x in Cc.items()}
            print(f"B {m:<16} AUC {b['auc']:.3f} R@20 {b['r20']:.4f} R@50 {b['r50']:.4f} | C pctile oth {c['oth']:.3f} "
                  f"pop20 {c['pop20']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        del w
    res["A_rollout"] = A
    out = f"{OUT}/cf_probe_seen{a.seen}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    sys.exit(main())
