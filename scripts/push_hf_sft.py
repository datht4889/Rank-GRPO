"""Push the SFT splits (+ catalog) of Reddit-v2 and ReMaRk's Music/Books/Books-Late to private HF datasets,
grouped in a private collection SFT_CRS. Token from env HF_TOKEN (never printed).

Usage: HF_TOKEN=... python scripts/push_hf_sft.py
"""
import os

from datasets import DatasetDict, load_from_disk
from huggingface_hub import HfApi

CRS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # parent of the repo
RM = f"{CRS}/data/remark_reddit_kaggle_v1/remark_reddit"
SETS = {
    "sft-crs-reddit-v2-movies": (f"{CRS}/Rank-GRPO/processed_datasets/sft_dataset", f"{CRS}/Rank-GRPO/reddit_catalog/gt_catalog.pkl",
                                 "Reddit-v2 (movies), SFT splits from the Rank-GRPO release (arXiv 2510.20150); completion = GPT-4o teacher list."),
    "sft-crs-reddit-music": (f"{RM}/music/processed_datasets/sft_dataset", f"{RM}/music/catalog/gt_catalog.pkl",
                             "ReMaRk (arXiv 2610.17771) Reddit-Music (r/musicsuggestions, MusicBrainz artists); completion = GPT-5.6 teacher list."),
    "sft-crs-reddit-books": (f"{RM}/books/processed_datasets/sft_dataset", f"{RM}/books/catalog/gt_catalog.pkl",
                             "ReMaRk Reddit-Books (r/suggestmeabook, Goodreads works; titles = Goodreads display titles, not ReMaRk's Reddit surface forms); completion = GPT-5.6 teacher list."),
    "sft-crs-reddit-books-late": (f"{RM}/books_late/processed_datasets/sft_dataset", f"{RM}/books_late/catalog/gt_catalog.pkl",
                                  "ReMaRk Reddit-Books-Late (same train as Books, later test 2017-10..12); completion = GPT-5.6 teacher list."),
}
COMMON = ("\n\nColumns: prompt (chat messages), completion (assistant message, one `Title (Year)` per line; empty in validation/test "
          "where no teacher list is released), seen_titles (empty for ReMaRk sets: not released), groundtruth_with_release_year "
          "(one responder's items). `gt_catalog.pkl` = sorted list of (title, year). Rebuilt with remark_data/ in "
          "github.com/datht4889/Rank-GRPO. Contains Reddit text: private, research use only.\n")

api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
repos = []
for name, (path, cat, desc) in SETS.items():
    rid = f"{user}/{name}"
    dd = DatasetDict({s: load_from_disk(f"{path}/{s}") for s in ("train", "validation", "test")})
    print(rid, {s: d.num_rows for s, d in dd.items()}, flush=True)
    api.create_repo(rid, repo_type="dataset", private=True, exist_ok=True)
    dd.push_to_hub(rid, private=True, token=os.environ["HF_TOKEN"])
    api.upload_file(path_or_fileobj=cat, path_in_repo="gt_catalog.pkl", repo_id=rid, repo_type="dataset")
    readme = f"---\nprivate: true\n---\n# {name}\n\n{desc}{COMMON}\nRows: " + ", ".join(f"{s} {d.num_rows:,}" for s, d in dd.items()) + "\n"
    # keep the dataset_info front matter written by push_to_hub, append our text
    cur = api.hf_hub_download(rid, "README.md", repo_type="dataset")
    head = open(cur, encoding="utf-8").read()
    api.upload_file(path_or_fileobj=(head.rstrip() + "\n\n" + readme.split("---\n", 2)[2]).encode(), path_in_repo="README.md",
                    repo_id=rid, repo_type="dataset")
    repos.append(rid)

col = next((c for c in api.list_collections(owner=user) if c.title == "SFT_CRS"), None)
if col is None:
    col = api.create_collection("SFT_CRS", description="SFT data for LLM conversational recommenders (Reddit-v2, ReMaRk Music/Books)",
                                private=True)
have = {i.item_id for i in api.get_collection(col.slug).items}
for rid in repos:
    if rid not in have:
        api.add_collection_item(col.slug, item_id=rid, item_type="dataset")
c = api.get_collection(col.slug)
print("collection:", c.slug, "private:", c.private, "items:", [i.item_id for i in c.items])
for rid in repos:
    print(rid, "private:", api.dataset_info(rid).private)
