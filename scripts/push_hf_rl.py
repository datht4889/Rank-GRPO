"""Push the RL (GRPO) training sets to HF datasets and add them to the SFT_CRS collection.

One dataset repo per corpus, two configs (train split each):
  rows  : one row per (request, responder), the Rank-GRPO training set
  union : one row per request, ground truth = union of all responders' items (ReMaRk), scripts/build_union_dataset.py
Plus the catalog (gt_catalog.pkl) used by the hit reward. Token from env HF_TOKEN (never printed).

Usage: HF_TOKEN=... python scripts/push_hf_rl.py [--private 1]
"""
import argparse
import os

from datasets import load_from_disk
from huggingface_hub import HfApi

CRS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RG = f"{CRS}/Rank-GRPO"
RM = f"{CRS}/data/remark_reddit_kaggle_v1/remark_reddit"
SETS = {
    "rl-crs-reddit-v2-movies": (f"{RG}/processed_datasets/grpo", f"{RG}/reddit_catalog/gt_catalog.pkl",
                                "Reddit-v2 (movies), GRPO training set of the Rank-GRPO release (arXiv 2510.20150)."),
    "rl-crs-reddit-music": (f"{RM}/music/processed_datasets/grpo", f"{RM}/music/catalog/gt_catalog.pkl",
                            "ReMaRk (arXiv 2610.17771) Reddit-Music (r/musicsuggestions, MusicBrainz artists)."),
    "rl-crs-reddit-books": (f"{RM}/books/processed_datasets/grpo", f"{RM}/books/catalog/gt_catalog.pkl",
                            "ReMaRk Reddit-Books (r/suggestmeabook, Goodreads works; titles = Goodreads display titles)."),
    "rl-crs-reddit-books-late": (f"{RM}/books_late/processed_datasets/grpo", f"{RM}/books_late/catalog/gt_catalog.pkl",
                                 "ReMaRk Reddit-Books-Late (train requests, same as Reddit-Books)."),
}
COMMON = ("\n\nConfigs (split `train`):\n"
          "- `rows`: one row per (request, responder), the Rank-GRPO training set.\n"
          "- `union`: one row per request; ground truth = union of all responders' items (ReMaRk's fix for responder "
          "noise).\n\nColumns: prompt (chat messages), seen_titles (movies mentioned in the conversation; empty for "
          "ReMaRk sets), groundtruth_with_release_year (list of [title, year]). `gt_catalog.pkl` = sorted list of "
          "(title, year) used by the hit reward. Built with github.com/datht4889/Rank-GRPO (scripts/build_union_dataset.py, "
          "remark_data/). Contains Reddit text: research use only.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", type=int, default=1)
    a = ap.parse_args()
    tok = os.environ["HF_TOKEN"]
    api = HfApi(token=tok)
    user = api.whoami()["name"]
    repos = []
    for name, (root, cat, desc) in SETS.items():
        rid = f"{user}/{name}"
        api.create_repo(rid, repo_type="dataset", private=bool(a.private), exist_ok=True)
        rows = {}
        for cfg, sub in (("rows", "grpo_dataset"), ("union", "grpo_union")):
            d = load_from_disk(f"{root}/{sub}/train")
            rows[cfg] = d.num_rows
            d.push_to_hub(rid, config_name=cfg, split="train", private=bool(a.private), token=tok)
        api.upload_file(path_or_fileobj=cat, path_in_repo="gt_catalog.pkl", repo_id=rid, repo_type="dataset")
        cur = api.hf_hub_download(rid, "README.md", repo_type="dataset")
        head = open(cur, encoding="utf-8").read().rstrip()
        body = f"\n\n# {name}\n\n{desc}{COMMON}\nRows: rows {rows['rows']:,}, union {rows['union']:,}\n"
        if f"# {name}\n" not in head:
            api.upload_file(path_or_fileobj=(head + body).encode(), path_in_repo="README.md", repo_id=rid,
                            repo_type="dataset")
        print(rid, rows, flush=True)
        repos.append(rid)

    col = next((c for c in api.list_collections(owner=user) if c.title == "SFT_CRS"), None)
    assert col is not None, "collection SFT_CRS not found"
    have = {i.item_id for i in api.get_collection(col.slug).items}
    for rid in repos:
        if rid not in have:
            api.add_collection_item(col.slug, item_id=rid, item_type="dataset")
    c = api.get_collection(col.slug)
    print("collection:", c.slug, "private:", c.private, "items:", [i.item_id for i in c.items])


if __name__ == "__main__":
    main()
