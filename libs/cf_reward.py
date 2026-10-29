# libs/cf_reward.py
"""Item-item collaborative-filtering partial credit for Rank-GRPO (EASE, Steck 2019).

The hit reward only knows the row's ground truth, so a relevant movie nobody in the thread named gets the same 0 as a
bad one. The CF score gives every recommended catalog movie a value in [0, 1]: the percentile, over the catalog, of
its EASE score given the row's ground truth + the movies mentioned in the conversation (the query). Hits get 1;
repeated items, seen (already mentioned) movies, items outside the catalog and empty ranks get 0.

Offline probe (probe/cf_reward_probe.py, Reddit-v2): on the SFT student's own rollouts EASE (lam 1000, with seen items)
ranks other responders' hits above misses at AUC 0.785 (student log-prob 0.798, both 0.821) and does not reward a
fixed top-20-popular list (catalog percentile 0.55 vs 0.78 for real hits); cosine item-kNN is gameable (0.93).

The trainer adds coef * (score - group mean) per (group, rank) to the advantage, NOT divided by the group std: with
std normalisation a small shaping term in an all-miss group would be rescaled to full advantage size.
"""
import re
from collections import defaultdict

import numpy as np

from libs.utils import process_rec_raw


def norm_title(s):
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


class CatalogIndex:
    """Catalog ids: ground truth by normalised title and year within 2 (as the hit metric), seen titles by title."""

    def __init__(self, catalog):
        self.items = sorted(catalog)
        self.exact = {it: i for i, it in enumerate(self.items)}
        self.by_title = defaultdict(list)
        for i, (t, y) in enumerate(self.items):
            self.by_title[norm_title(t)].append((int(y), i))

    def gt(self, t, y):
        try:
            y = int(str(y))
        except ValueError:
            return None
        c = [(abs(yy - y), i) for yy, i in self.by_title.get(norm_title(t), []) if abs(yy - y) <= 2]
        return min(c)[1] if c else None

    def seen(self, t):
        return [i for _, i in self.by_title.get(norm_title(t), [])]

    def query(self, gt, seen):
        q = {i for i in (self.gt(t, y) for t, y in gt) if i is not None}
        g = set(q)
        q.update(i for t in seen for i in self.seen(t))
        return sorted(q), g


def fit_ease(rows, index, lam):
    """EASE on pseudo-users = requests (rows with the same prompt), items = union of their ground truth + seen."""
    n = len(index.items)
    users = defaultdict(set)
    for prompt, gt, seen in rows:
        q, _ = index.query(gt, seen)
        users[prompt].update(q)
    C = np.zeros((n, n), np.float64)
    for ids in users.values():
        if ids:
            ids = np.fromiter(ids, np.int64)
            C[np.ix_(ids, ids)] += 1
    C[np.diag_indices(n)] += lam
    P = np.linalg.inv(C)
    W = -P / np.diag(P)[None, :]
    np.fill_diagonal(W, 0)
    return W.astype(np.float32), len(users)


class CFScorer:
    def __init__(self, path, catalog):
        d = np.load(path)
        self.W = d["W"]
        self.index = CatalogIndex(catalog)
        assert self.W.shape[0] == len(self.index.items), "CF matrix was built for another catalog"

    def scores(self, completions, gts, seens, rec_num):
        """(N, rec_num) float32 partial credit in [0, 1]."""
        out = np.zeros((len(completions), rec_num), np.float32)
        n = self.W.shape[0]
        for k, (text, gt, seen) in enumerate(zip(completions, gts, seens)):
            err, it = process_rec_raw({"raw": text}, "raw", "recs")
            if err or not it["recs"]:
                continue
            q, g = self.index.query(gt, seen)
            if not q:
                continue
            s = self.W[q].sum(0)
            srt = np.sort(s)
            seen_set, used = set(seen), set()
            for r, (t, y) in enumerate(it["recs"][:rec_num]):
                i = self.index.exact.get((t, int(y)))
                if i is None or t in seen_set or i in used:
                    continue
                used.add(i)
                if i in g:
                    out[k, r] = 1.0
                else:
                    out[k, r] = (np.searchsorted(srt, s[i], "left") + np.searchsorted(srt, s[i], "right")) / (2.0 * n)
        return out
