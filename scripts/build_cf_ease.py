"""Fit the EASE item-item matrix used by the CF partial-credit reward (libs/cf_reward.py).

Pseudo-users are the GRPO train requests (rows sharing a prompt), items the union of their responders' ground truth and
the catalog movies mentioned in the conversation. Always fitted on the per-responder set (grpo_dataset), so the rows and
union training sets get the same matrix.

Usage: python scripts/build_cf_ease.py [--lam 1000] [--train processed_datasets/grpo/grpo_dataset/train]
                                       [--catalog reddit_catalog/gt_catalog.pkl] [--out cf/ease_l1000.npz]
"""
import argparse
import os
import sys
import time

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.cf_reward import CatalogIndex, fit_ease  # noqa: E402
from libs.data import load_catalog  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--train", default="processed_datasets/grpo/grpo_dataset/train")
    ap.add_argument("--catalog", default="reddit_catalog/gt_catalog.pkl")
    ap.add_argument("--out", default="cf/ease_l1000.npz")
    a = ap.parse_args()
    t0 = time.time()
    index = CatalogIndex(load_catalog(a.catalog))
    ds = load_from_disk(a.train)
    rows = zip((p[0]["content"] for p in ds["prompt"]), ds["groundtruth_with_release_year"], ds["seen_titles"])
    W, n_users = fit_ease(rows, index, a.lam)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, W=W, lam=a.lam)
    print(f"EASE lam {a.lam}: {W.shape[0]} items, {n_users} requests -> {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
