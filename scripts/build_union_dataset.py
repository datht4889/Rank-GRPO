"""ReMaRk union training set (arXiv 2610.17771) from the Rank-GRPO Reddit-v2 GRPO split.

One row per request (unique prompt) instead of one row per (request, responder). Ground truth = union of every
responder's items (each item once), seen titles = union of the responders' seen titles. Trained with the unchanged
Rank-GRPO reward (exp_inf / log_decay), this is ReMaRk's "union" arm with weight 1 per item: a list position earns
credit if any responder named the item. Same columns as grpo_dataset, so train_rank_grpo.py needs no change.

Usage: python scripts/build_union_dataset.py [--src processed_datasets/grpo/grpo_dataset/train]
                                             [--dst processed_datasets/grpo/grpo_union/train]
"""

import argparse
import json
import os
from collections import OrderedDict

from datasets import Dataset, load_from_disk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "processed_datasets/grpo/grpo_dataset/train"))
    ap.add_argument("--dst", default=os.path.join(ROOT, "processed_datasets/grpo/grpo_union/train"))
    args = ap.parse_args()

    d = load_from_disk(args.src)
    prompts, gts, seens = d["prompt"], d["groundtruth_with_release_year"], d["seen_titles"]
    groups = OrderedDict()
    for i, p in enumerate(prompts):
        groups.setdefault(p[0]["content"], []).append(i)

    rows = {"prompt": [], "seen_titles": [], "groundtruth_with_release_year": []}
    for idx in groups.values():
        items = OrderedDict()
        seen = []
        for i in idx:
            for t, y in gts[i]:
                items.setdefault((t, str(y)), None)
            seen += [s for s in seens[i] if s not in seen]
        rows["prompt"].append(prompts[idx[0]])
        rows["seen_titles"].append(seen)
        rows["groundtruth_with_release_year"].append([[t, y] for t, y in items])

    out = Dataset.from_dict(rows)
    out.save_to_disk(args.dst)
    stats = {"rows_in": len(d), "requests_out": len(out),
             "mean_responders": len(d) / len(out),
             "mean_union_size": sum(len(g) for g in rows["groundtruth_with_release_year"]) / len(out)}
    json.dump(stats, open(os.path.join(os.path.dirname(args.dst), "union_stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
