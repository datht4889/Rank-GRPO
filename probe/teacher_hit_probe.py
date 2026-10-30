"""CPU probe: can "distill only where the teacher is right" rescue Rank-GRPO's dead groups?

The released SFT data (datht/sft-crs-reddit-v2-movies) has one GPT-4o list per training row. For the 300 rollout-probe
requests (G = 8 SFT-student lists each) we check:
  * whether the GPT-4o list for a row was written with that row's responder in mind (hit rate on the row's own gold vs
    on the other responders' gold): if so it is a privileged teacher, not a zero-shot one
  * GPT-4o recall vs the student's, and how often a GPT-4o list hits where all 8 student lists miss
    (request level and per (request, rank) cell), i.e. how much signal a "teacher list as extra rollout if it hits"
    scheme (LUFFY-style off-policy guidance) would add
  * how many GPT-4o items the student already samples (the student was SFT-trained on these lists)

Usage: python -I probe/teacher_hit_probe.py <rollout_dir> <sft_train.parquet>
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import transformers  # noqa: F401  (libs/utils.py imports TrainerCallback; not needed for matching)
except ImportError:
    import types
    sys.modules["transformers"] = types.SimpleNamespace(TrainerCallback=object)
from rollout_probe import CATALOG, REC_NUM, evaluate_direct_match_aligned, parse_items  # noqa: E402

rdir, parquet = sys.argv[1], sys.argv[2]
reqs = json.load(open(os.path.join(rdir, "requests.json"), encoding="utf-8"))
rolls = [json.loads(l) for l in open(os.path.join(rdir, "rollouts.jsonl"), encoding="utf-8")]
catalog = set(pickle.load(open(CATALOG, "rb")))

t = pq.read_table(parquet)
rows_by_prompt = defaultdict(list)
for p, c, g, s in zip(t.column("prompt").to_pylist(), t.column("completion").to_pylist(),
                      t.column("groundtruth_with_release_year").to_pylist(), t.column("seen_titles").to_pylist()):
    rows_by_prompt[p[0]["content"]].append({"comp": c[0]["content"] if c else "", "gt": [list(x) for x in g], "seen": list(s)})


def hits(recs, gt, seen):
    return [int(x) for x in evaluate_direct_match_aligned({"recs": recs, "gt": gt, "seen": seen}, REC_NUM, "seen", "recs",
                                                          "gt", catalog)]


def norm(t, y):
    return (str(t).strip().casefold(), int(y))


by_req = defaultdict(list)
for r in rolls:
    by_req[r["req"]].append(r)

own, other, matched = [], [], 0
stu_any_u, t_any_u, rescue_req_u, rescue_req_d, dead_req_u, dead_req_d = [], [], [], [], 0, 0
cells_dead_u = cells_dead_d = cell_rescue_u = cell_rescue_d = 0
stu_recall_u, t_recall_u, overlap = [], [], []
for q, req in enumerate(reqs):
    rows = rows_by_prompt.get(req["content"], [])
    if not rows:
        continue
    matched += 1
    # (1) privileged or not: each GPT-4o list vs its own row's gold and vs the other rows' gold
    for i, r in enumerate(rows):
        recs = [(t_, y) for t_, y, _, _ in parse_items(r["comp"])]
        n_own = sum(hits(recs, r["gt"], r["seen"])) / max(len(r["gt"]), 1)
        others = [x for j, o in enumerate(rows) if j != i for x in o["gt"] if x not in r["gt"]]
        if others:
            own.append(n_own)
            other.append(sum(hits(recs, others, r["seen"])) / len(others))
    # (2) rescue: the GPT-4o list of the drawn responder's row (what a per-row trainer would have) vs the 8 student lists
    drawn = [r for r in rows if r["gt"] == req["drawn_gt"]]
    teacher = drawn[0] if drawn else rows[0]
    t_recs = [(t_, y) for t_, y, _, _ in parse_items(teacher["comp"])]
    t_d = hits(t_recs, req["drawn_gt"], req["drawn_seen"])
    t_u = hits(t_recs, req["union_gt"], req["union_seen"])
    S_d = np.zeros((len(by_req[q]), REC_NUM), int)
    S_u = np.zeros_like(S_d)
    stu_items = set()
    for k, r in enumerate(by_req[q]):
        S_d[k, :len(r["hit_drawn"][:REC_NUM])] = r["hit_drawn"][:REC_NUM]
        S_u[k, :len(r["hit_union"][:REC_NUM])] = r["hit_union"][:REC_NUM]
        stu_items |= {norm(t_, y) for t_, y, _, _ in r["items"]}
    t_d = np.array((t_d + [0] * REC_NUM)[:REC_NUM])
    t_u = np.array((t_u + [0] * REC_NUM)[:REC_NUM])
    stu_recall_u.append(S_u.sum(1).mean() / len(req["union_gt"]))
    t_recall_u.append(t_u.sum() / len(req["union_gt"]))
    overlap.append(np.mean([norm(t_, y) in stu_items for t_, y in t_recs]) if t_recs else 0.0)
    if S_u.sum() == 0:
        dead_req_u += 1
        rescue_req_u.append(t_u.sum() > 0)
    if S_d.sum() == 0:
        dead_req_d += 1
        rescue_req_d.append(t_d.sum() > 0)
    dead_d = S_d.sum(0) == 0
    dead_u = S_u.sum(0) == 0
    cells_dead_d += dead_d.sum()
    cells_dead_u += dead_u.sum()
    cell_rescue_d += (dead_d & (t_d > 0)).sum()
    cell_rescue_u += (dead_u & (t_u > 0)).sum()

out = {
    "requests_matched_in_sft_data": f"{matched}/{len(reqs)}",
    "gpt4o_hit_rate_own_row_gold": float(np.mean(own)), "gpt4o_hit_rate_other_responders_gold": float(np.mean(other)),
    "n_lists_for_own_vs_other": len(own),
    "recall_union_student_mean_over_8": float(np.mean(stu_recall_u)), "recall_union_gpt4o": float(np.mean(t_recall_u)),
    "pct_gpt4o_items_student_also_sampled": float(np.mean(overlap)),
    "dead_requests_drawn": dead_req_d, "dead_requests_union": dead_req_u,
    "pct_dead_requests_rescued_by_gpt4o_drawn": float(np.mean(rescue_req_d)) if rescue_req_d else None,
    "pct_dead_requests_rescued_by_gpt4o_union": float(np.mean(rescue_req_u)) if rescue_req_u else None,
    "dead_cells_drawn": int(cells_dead_d), "dead_cells_union": int(cells_dead_u),
    "pct_dead_cells_where_gpt4o_same_rank_hits_drawn": float(cell_rescue_d / cells_dead_d),
    "pct_dead_cells_where_gpt4o_same_rank_hits_union": float(cell_rescue_u / cells_dead_u),
}
print(json.dumps(out, indent=2))
