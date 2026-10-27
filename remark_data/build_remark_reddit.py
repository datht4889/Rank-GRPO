"""Rebuild ReMaRk's Reddit datasets (arXiv 2610.17771) in the Rank-GRPO dataset format.

ReMaRk releases ids and labels only (ReMaRk/data_release): requests, per-responder item ids, GPT-5.6 SFT target item ids
and the catalog. The request text is fetched by submission id from the Arctic Shift API and rendered exactly as ReMaRk
does (books_threads.py / music_threads.py): context = "USER: {title}. {selftext}\n", prompt = header + "Here is the
conversation: " + context, with the music / book header of books_common.py / music_common.py.

Output (same layout and columns as the Rank-GRPO release, so train_sft.py / train_rank_grpo.py / eval work unchanged):
  <out>/<dataset>/processed_datasets/sft_dataset/{train,validation,test}   prompt, completion, seen_titles, groundtruth_with_release_year
  <out>/<dataset>/processed_datasets/grpo/grpo_dataset/train               prompt, seen_titles, groundtruth_with_release_year
  <out>/<dataset>/processed_datasets/grpo/grpo_union/train                 same, one row per train request (ReMaRk union GT)
  <out>/<dataset>/catalog/gt_catalog.pkl                                   sorted list[(name, int year)]
  <out>/<dataset>/build_stats.json
SFT train rows = the responders of requests with an SFT target (completion = the teacher list); validation/test rows =
one per responder (completion = "" since no teacher list is released); GRPO rows = one per train responder.

Known deviations from ReMaRk's files: seen_titles are not released, so they are empty (ground truth already excludes
the requester's items; only the eval-time removal of seen items from the generated list is lost). Posts that Arctic
Shift no longer returns are dropped and counted.

Books needs a catalog with titles: pass --titles <tsv item_id, title, year> from goodreads_titles.py. Items without a
title there are dropped from every list; items that end up with the same (title, year) are merged (lists de-duplicated);
a responder left without items gives no row, and a request left without rows is dropped (all counted).

Usage: python remark_data/build_remark_reddit.py --dataset music|books|books_late [--titles <tsv>]
                                                  [--release ../ReMaRk/data_release] [--out ../data/remark_reddit]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

from datasets import Dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
API = "https://arctic-shift.photon-reddit.com/api/posts/ids"
UA = "research-crs/0.1 (academic dataset build)"
FILE_PREFIX = {"music": "music", "books": "books", "books_late": "books_late"}

REDDIT_HEADER = ('Pretend you are a movie recommender system.\nI will give you a conversation between a user and you '
                 '(a recommender system). Based on the conversation, you need to reply with 20 recommendations. List '
                 'the standardized English title of each movie in each line in the form of "movie name" '
                 '(release_year) with NO extra words or sentences.\n')
HEADERS = {
    "music": (REDDIT_HEADER.replace("a movie recommender", "a music recommender")
              .replace("the standardized English title of each movie", "the standard name of each artist or band")
              .replace('"movie name" (release_year)', '"artist name" (start_year)')),
    "books": (REDDIT_HEADER.replace("a movie recommender", "a book recommender").replace("each movie", "each book")
              .replace('"movie name" (release_year)', '"book title" (first_publication_year)')),
}
HEADERS["books_late"] = HEADERS["books"]
SEP = "Here is the conversation: "


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fetch_posts(ids, cache):
    """Posts by id from Arctic Shift, cached (resumable) in a JSONL file."""
    have = {}
    if os.path.exists(cache):
        for r in jsonl(cache):
            have[r["id"]] = r
    todo = [i for i in ids if i not in have]
    with open(cache, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), 100):
            chunk = todo[b:b + 100]
            q = urllib.parse.urlencode({"ids": ",".join(chunk), "fields": "id,title,selftext,created_utc,author"})
            req = urllib.request.Request(API + "?" + q, headers={"User-Agent": UA})
            delay, data = 2.0, None
            for _ in range(20):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = json.loads(r.read()).get("data")
                    if data is not None:
                        break
                except Exception as exc:  # noqa: BLE001 - network errors: back off and retry
                    print("retry:", repr(exc)[:120], flush=True)
                time.sleep(delay)
                delay = min(delay * 1.7, 60)
            if data is None:
                raise RuntimeError(f"Arctic Shift failed for ids starting {chunk[0]}")
            for x in data:
                have[x["id"]] = x
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
            f.flush()
            if (b // 100) % 20 == 0:
                print(f"  posts {min(b + 100, len(todo))}/{len(todo)}", flush=True)
    return have


def load_catalog(release, prefix, titles):
    """item_id -> (name, int year)."""
    if titles:
        rows = csv.DictReader(open(titles, encoding="utf-8"), delimiter="\t")
        return {int(r["item_id"]): (r["title"], int(r["year"])) for r in rows if r["title"]}
    rows = csv.DictReader(open(os.path.join(release, f"{prefix}_catalog.tsv"), encoding="utf-8"), delimiter="\t")
    return {int(r["item_id"]): (r["name"], int(r["year"])) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(FILE_PREFIX))
    ap.add_argument("--release", default=os.path.join(HERE, "..", "ReMaRk", "data_release"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "remark_reddit"))
    ap.add_argument("--titles", default=None, help="books only: TSV item_id, title, year")
    a = ap.parse_args()
    prefix = FILE_PREFIX[a.dataset]
    if a.dataset != "music" and not a.titles:
        ap.error("books needs --titles (the release catalog has Goodreads work ids only)")
    root = os.path.join(a.out, a.dataset)
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)

    reqs = jsonl(os.path.join(a.release, f"{prefix}_requests.jsonl"))
    resps = defaultdict(list)
    for r in jsonl(os.path.join(a.release, f"{prefix}_responses.jsonl")):
        resps[r["request_id"]].append(r)
    sft = {r["request_id"]: r["item_ids"] for r in jsonl(os.path.join(a.release, f"{prefix}_sft_targets.jsonl"))}
    cat = load_catalog(a.release, prefix, a.titles)
    print(f"{a.dataset}: {len(reqs)} requests, {sum(map(len, resps.values()))} responses, {len(sft)} SFT targets, "
          f"{len(cat)} catalog items", flush=True)

    # books and books_late share submissions: one post cache for both
    cache = os.path.join(a.out, "raw", f"posts_{'books' if a.dataset.startswith('books') else a.dataset}.jsonl")
    posts = fetch_posts([r["request_id"] for r in reqs], cache)

    header = HEADERS[a.dataset]
    st = Counter()

    def items(ids, kind):
        out = []
        for i in ids:
            if i not in cat:
                st[f"dropped_item_without_title_{kind}"] += 1
            elif cat[i] in out:
                st[f"merged_duplicate_item_{kind}"] += 1
            else:
                out.append(cat[i])
        return out

    rows = {s: [] for s in ("train", "validation", "test")}
    grpo, union = [], []  # union: ReMaRk's one row per train request, GT = union of its responders' items
    for r in reqs:
        p = posts.get(r["request_id"])
        if p is None:
            st[f"missing_post_{r['split']}"] += 1
            continue
        body = p.get("selftext") or ""
        st["post_body_removed_or_deleted"] += body.strip() in ("[removed]", "[deleted]")
        prompt = [{"role": "user", "content": header + SEP + f"USER: {p.get('title') or ''}. {body}\n"}]
        rs = sorted(resps[r["request_id"]], key=lambda x: x["responder"])
        assert len(rs) == r["n_responders"], r["request_id"]
        kept, union_gt = 0, []
        for x in rs:
            gt = [[t, str(y)] for t, y in items(x["item_ids"], "gt")]
            if not gt:
                st[f"dropped_responder_no_items_{r['split']}"] += 1
                continue
            kept += 1
            if r["split"] == "train":
                grpo.append({"prompt": prompt, "seen_titles": [], "groundtruth_with_release_year": gt})
                union_gt += [g for g in gt if g not in union_gt]
                if r["request_id"] in sft:
                    comp = "\n".join(f"{t} ({y})" for t, y in items(sft[r["request_id"]], "sft"))
                    rows["train"].append({"prompt": prompt, "completion": [{"role": "assistant", "content": comp}],
                                          "seen_titles": [], "groundtruth_with_release_year": gt})
            else:
                rows[r["split"]].append({"prompt": prompt, "completion": [{"role": "assistant", "content": ""}],
                                         "seen_titles": [], "groundtruth_with_release_year": gt})
        if union_gt:
            union.append({"prompt": prompt, "seen_titles": [], "groundtruth_with_release_year": union_gt})
        if kept:
            st[f"requests_{r['split']}"] += 1
        else:
            st[f"dropped_request_no_rows_{r['split']}"] += 1

    pd = os.path.join(root, "processed_datasets")
    for s, rr in rows.items():
        Dataset.from_list(rr).save_to_disk(os.path.join(pd, "sft_dataset", s))
        st[f"sft_rows_{s}"] = len(rr)
    Dataset.from_list(grpo).save_to_disk(os.path.join(pd, "grpo", "grpo_dataset", "train"))
    st["grpo_rows_train"] = len(grpo)
    Dataset.from_list(union).save_to_disk(os.path.join(pd, "grpo", "grpo_union", "train"))
    st["union_rows_train"] = len(union)
    st["union_mean_items"] = round(sum(len(u["groundtruth_with_release_year"]) for u in union) / max(len(union), 1), 3)
    os.makedirs(os.path.join(root, "catalog"), exist_ok=True)
    pickle.dump(sorted(set(cat.values())), open(os.path.join(root, "catalog", "gt_catalog.pkl"), "wb"))
    st["catalog_items"] = len(set(cat.values()))
    stats = dict(sorted(st.items()))
    json.dump(stats, open(os.path.join(root, "build_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
