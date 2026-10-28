"""Build the hint-SFT dataset: teach the SFT policy to USE privileged hints (for a self-distillation teacher).

The released SFT policy reads movies placed in the prompt as "already watched" and pushes them down (rollout probe:
AUC 0.03 on revealed gold). Here, for requests with >= 2 responders in the SFT train split:
  hinted : prompt + HINT_V2 listing one responder's items  ->  all responders' items (hint + the others, shuffled,
           minus seen titles) first, then another responder row's GPT-4o list as filler, 20 lines in total
  replay : the same number of original SFT rows (prompt -> GPT-4o list), so the no-hint behaviour stays anchored
The 300 requests used by scripts/rollout_probe.py (seed 42) are excluded so the probe stays held out.

Usage (repo root): python scripts/build_hint_sft.py [--per_prompt 2] [--n_val 300]
Writes processed_datasets/sft_hint_dataset/{train,validation}.
"""
import argparse
import os
import random
import re
import sys
from collections import defaultdict

from datasets import Dataset, load_from_disk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RANKGRPO_REPO", REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout_probe import HINT_V2, load_requests  # noqa: E402

LINE = re.compile(r"^\s*\"?(.+?)\"?\s*\((\d{4})\)\s*$")


def key(t):
    return str(t).strip().casefold()


def parse_list(text):
    out = []
    for line in text.split("\n"):
        m = LINE.match(line)
        if m:
            out.append((m.group(1).strip(), m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_prompt", type=int, default=2, help="hinted examples per request (different hint responder)")
    ap.add_argument("--n_val", type=int, default=300, help="hinted and replay validation examples each")
    ap.add_argument("--probe_requests", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    src = load_from_disk(os.path.join(REPO, "processed_datasets/sft_dataset/train"))
    held_out = {r["content"] for r in load_requests(args.probe_requests, 42)}
    groups = defaultdict(list)
    for i, r in enumerate(src):
        p = r["prompt"][0]["content"]
        if p not in held_out:
            groups[p].append(i)

    hinted = []
    for p, idx in groups.items():
        rows = [src[i] for i in idx]
        with_gold = [k for k, r in enumerate(rows) if r["groundtruth_with_release_year"]]
        if len(with_gold) < 2:
            continue
        seen = {key(t) for r in rows for t in r["seen_titles"]}
        for a in rng.sample(with_gold, min(args.per_prompt, len(with_gold))):
            hint = [tuple(x) for x in rows[a]["groundtruth_with_release_year"]]
            gold, used = [], set(seen)
            for t, y in hint + [tuple(x) for k in with_gold if k != a for x in rows[k]["groundtruth_with_release_year"]]:
                if key(t) not in used:
                    used.add(key(t))
                    gold.append((t, y))
            rng.shuffle(gold)
            target = gold[:20]
            filler_row = rows[rng.choice([k for k in range(len(rows)) if k != a])]
            for t, y in parse_list(filler_row["completion"][0]["content"] if filler_row["completion"] else ""):
                if len(target) >= 20:
                    break
                if key(t) not in used:
                    used.add(key(t))
                    target.append((t, y))
            items = "\n".join(f"{t} ({y})" for t, y in hint)
            hinted.append({"prompt": [{"role": "user", "content": p + HINT_V2.format(items=items)}],
                           "completion": [{"role": "assistant", "content": "\n".join(f"{t} ({y})" for t, y in target)}]})

    pool = [i for idx in groups.values() for i in idx]
    replay = [{"prompt": src[i]["prompt"], "completion": src[i]["completion"]} for i in rng.sample(pool, len(hinted))]
    rng.shuffle(hinted)
    v = args.n_val
    train = hinted[v:] + replay[v:]
    val = hinted[:v] + replay[:v]
    rng.shuffle(train)
    out = os.path.join(REPO, "processed_datasets/sft_hint_dataset")
    Dataset.from_list(train).save_to_disk(os.path.join(out, "train"))
    Dataset.from_list(val).save_to_disk(os.path.join(out, "validation"))
    n_tgt = sum(len(h["completion"][0]["content"].split("\n")) for h in hinted) / max(len(hinted), 1)
    print(f"held out {len(held_out)} probe requests; hinted {len(hinted)} (mean target lines {n_tgt:.1f}), "
          f"replay {len(replay)}; train {len(train)}, validation {len(val)} -> {out}")
    print("example hinted prompt tail:\n" + hinted[0]["prompt"][0]["content"][-600:])
    print("example hinted target:\n" + hinted[0]["completion"][0]["content"])


if __name__ == "__main__":
    main()
