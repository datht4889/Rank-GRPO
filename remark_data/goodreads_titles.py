"""Titles and years for the ReMaRk Reddit-Books catalogs (item_id -> Goodreads work ids) from the UCSD Goodreads graph.

ReMaRk's item title is the work's most frequent surface form in the Reddit mentions (LLM-extracted, not released), so it
cannot be rebuilt exactly. We use the rule ReMaRk's books_goodreads_index.py uses for the Goodreads display title, then
the same canonicalisation (books_common.canon: a fixed point of Rank-GRPO's title parser):
  title = title_without_series of the work's best_book (else of its most-rated edition, else the works original_title)
  year  = works original_publication_year if in [1000, 2018], else the earliest valid edition publication_year
of the item's first (most-rated) work id. Items whose title does not canonicalise or that have no valid year are written
with an empty title and dropped by build_remark_reddit.py (counted there).

Usage: python goodreads_titles.py --catalog books_catalog.tsv [--catalog books_late_catalog.tsv ...]
                                  --works goodreads_book_works.json.gz --books goodreads_books.json.gz --out_dir <dir>
Writes <out_dir>/<catalog name>_titles.tsv (item_id, title, year) and goodreads_titles_stats.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import types
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "libs"))
try:
    import transformers  # noqa: F401  (libs/utils imports TrainerCallback only)
except ImportError:
    sys.modules["transformers"] = types.SimpleNamespace(TrainerCallback=object)
from utils import process_rec_raw  # noqa: E402  (Rank-GRPO's title parser)

MIN_YEAR, MAX_YEAR = 1000, 2018
_TYPO = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "…": "..."})


def ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


@lru_cache(maxsize=None)
def _parse_title_once(t):
    _, item = process_rec_raw({"raw": f"{t} (2000)"}, "raw", "rec")
    rec = item["rec"]
    return rec[0][0] if len(rec) == 1 and int(rec[0][1]) == 2000 and rec[0][0] else None


def canon(title):
    """ReMaRk books_common.canon (verbatim logic)."""
    t = ws((title or "").translate(_TYPO))
    while True:
        t2 = ws(re.sub(r"\([^()]*\)", "", t))
        if t2 == t:
            break
        t = t2
    for _ in range(5):
        if not t:
            return None
        c = _parse_title_once(t)
        if c == t:
            return t
        t = c or ""
    return None


def year_ok(y):
    try:
        y = int(y)
    except (TypeError, ValueError):
        return None
    return y if MIN_YEAR <= y <= MAX_YEAR else None


def lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", action="append", required=True)
    ap.add_argument("--works", required=True)
    ap.add_argument("--books", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    cats = {}
    need = set()
    for c in a.catalog:
        rows = list(csv.DictReader(open(c, encoding="utf-8"), delimiter="\t"))
        cats[c] = {int(r["item_id"]): r["goodreads_work_ids"].split(",")[0].strip() for r in rows}
        need |= set(cats[c].values())
    print(f"{len(need)} distinct works needed", flush=True)

    works = {}
    for w in lines(a.works):
        if w["work_id"] in need:
            works[w["work_id"]] = {"best": w.get("best_book_id"), "orig": w.get("original_title") or "",
                                   "year": year_ok(w.get("original_publication_year"))}
    print(f"works found: {len(works)}", flush=True)

    best_ids = {v["best"]: k for k, v in works.items() if v["best"]}
    best_title, top = {}, {}  # work -> title of best_book; work -> (ratings, title) of most-rated edition
    min_year = {}
    for i, b in enumerate(lines(a.books)):
        wid = b.get("work_id")
        if wid not in works:
            continue
        t = b.get("title_without_series") or b.get("title") or ""
        if b.get("book_id") in best_ids and best_ids[b["book_id"]] == wid:
            best_title[wid] = t
        r = int(b.get("ratings_count") or 0)
        if t and (wid not in top or r > top[wid][0]):
            top[wid] = (r, t)
        y = year_ok(b.get("publication_year"))
        if y and (wid not in min_year or y < min_year[wid]):
            min_year[wid] = y
        if i % 500000 == 0:
            print(f"  editions read {i}", flush=True)

    stats = {"works_needed": len(need), "works_found": len(works)}
    os.makedirs(a.out_dir, exist_ok=True)
    for c, items in cats.items():
        name = os.path.basename(c).replace("_catalog.tsv", "")
        st = {"items": len(items), "no_work": 0, "no_title": 0, "no_year": 0, "canon_failed": 0, "ok": 0,
              "title_from_best_book": 0, "year_from_editions": 0}
        out = []
        for iid, wid in sorted(items.items()):
            w = works.get(wid)
            if w is None:
                st["no_work"] += 1
                out.append((iid, "", ""))
                continue
            raw = best_title.get(wid) or (top.get(wid) or (0, ""))[1] or w["orig"]
            st["title_from_best_book"] += bool(best_title.get(wid))
            year = w["year"] or min_year.get(wid)
            st["year_from_editions"] += w["year"] is None and year is not None
            t = canon(raw) if raw else None
            if not raw:
                st["no_title"] += 1
            elif t is None:
                st["canon_failed"] += 1
            if year is None:
                st["no_year"] += 1
            if t and year:
                st["ok"] += 1
                out.append((iid, t, year))
            else:
                out.append((iid, "", ""))
        st["distinct_title_year"] = len({(t, y) for _, t, y in out if t})
        with open(os.path.join(a.out_dir, f"{name}_titles.tsv"), "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["item_id", "title", "year"])
            wr.writerows(out)
        stats[name] = st
        print(name, json.dumps(st), flush=True)
    json.dump(stats, open(os.path.join(a.out_dir, "goodreads_titles_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
