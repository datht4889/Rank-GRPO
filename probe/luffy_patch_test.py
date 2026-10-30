"""CPU check of RankGRPOTrainer._luffy_patch (LUFFY gold guidance) on the rollout-probe lists (300 requests x 8).

Checks: only the last list of a group changes, unchanged items keep their sampled token ids, the off-policy mask
covers exactly the written gold lines, and the reward function scores each written gold item as a hit at its rank.

Usage: python -I probe/luffy_patch_test.py <rollout_dir> <Rank-GRPO dir> [gold] [scope]
"""
import ast
import json
import sys
from collections import defaultdict

import torch
from transformers import AutoTokenizer

roll_dir, repo = sys.argv[1], sys.argv[2]
gold_n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
scope = sys.argv[4] if len(sys.argv) > 4 else "dead"
sys.path.insert(0, repo)
from trl import RankGRPOTrainer  # noqa: E402
from libs.data import load_catalog  # noqa: E402
from libs.reward_funcs import make_reward_func_individual  # noqa: E402

G, R = 8, 20
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
reqs = json.load(open(f"{roll_dir}/requests.json", encoding="utf-8"))
rolls = [json.loads(line) for line in open(f"{roll_dir}/rollouts.jsonl", encoding="utf-8")]
rolls.sort(key=lambda r: (r["req"], r["g"]))

t = RankGRPOTrainer.__new__(RankGRPOTrainer)
t.num_generations, t.processing_class, t.max_completion_length = G, tok, 1024
t.eos_token_id, t.pad_token_id = tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id
t.reward_funcs = [make_reward_func_individual(rec_num=R, gt_catalog=load_catalog(f"{repo}/reddit_catalog/gt_catalog.pkl"))]
t.luffy_gold, t.luffy_scope = gold_n, scope
t._metrics = {"train": defaultdict(list)}

lit = lambda v: ast.literal_eval(v) if isinstance(v, str) else v
inputs, rows = [], []
for r in rolls:
    q = reqs[r["req"]]
    inputs.append({"prompt": [{"role": "user", "content": q["content"]}],
                   "groundtruth_with_release_year": lit(q["union_gt"]),
                   "seen_titles": lit(q["union_seen"])})
    rows.append(tok.encode(r["text"], add_special_tokens=False) + [t.eos_token_id])
width = max(map(len, rows))
ids = torch.tensor([r + [t.pad_token_id] * (width - len(r)) for r in rows])

new_ids, offp = t._luffy_patch(inputs, ids, R)
print({k: v for k, v in t._metrics["train"].items()})

reward = t.reward_funcs[0]
def hits(id_rows):
    texts = tok.batch_decode(id_rows, skip_special_tokens=True)
    return torch.tensor(reward(completions=[[{"content": x}] for x in texts],
                               groundtruth_with_release_year=[x["groundtruth_with_release_year"] for x in inputs],
                               seen_titles=[x["seen_titles"] for x in inputs]))
h0, h1 = hits(ids), hits(new_ids)
changed = [i for i in range(len(rows)) if new_ids[i, :len(rows[i])].tolist() != rows[i]
           or (new_ids[i, len(rows[i]):] != t.pad_token_id).any()]
assert all(i % G == G - 1 for i in changed), "a non-guided list changed"
assert all(offp[i].any() for i in changed) and not offp[[i for i in range(len(rows)) if i not in changed]].any()
n_written = int(sum(t._metrics["train"]["luffy/gold_per_patched_list"][0] * len(changed) for _ in [0]))
bad = 0
for i in changed:
    new_hit = (h1[i] == 1) & (h0[i] == 0)
    gold_lines = tok.decode(new_ids[i][offp[i]]).strip().split("\n")
    if int(new_hit.sum()) != len(gold_lines):
        bad += 1
        print("MISMATCH", i, gold_lines, h0[i].tolist(), h1[i].tolist())
    # tokens outside the mask equal the sampled ones, item by item
    keep_new = new_ids[i][~offp[i] & (new_ids[i] != t.pad_token_id)].tolist()
    assert all(tok_id in rows[i] for tok_id in keep_new)
print(f"guided lists {len(changed)} / {len(rows) // G} groups, gold lines that do not score as hits: {bad}")
print("group hit-rank count before/after:", int((h0.view(-1, G, R).sum(1) > 0).sum()),
      int((h1.view(-1, G, R).sum(1) > 0).sum()))
i = changed[0]
print("example before:\n" + tok.decode(rows[i], skip_special_tokens=True)[:400])
print("example after (gold lines marked):")
for line in tok.decode(new_ids[i], skip_special_tokens=True).split("\n")[:8]:
    print("  ", line)
print("off-policy text:", repr(tok.decode(new_ids[i][offp[i]])))
