#!/usr/bin/env bash
# Build ReMaRk's Reddit-Music, Reddit-Books and Reddit-Books-Late in the Rank-GRPO dataset format (CPU, needs internet).
#   bash remark_data/build_all.sh            # OUT=/kaggle/working/remark_reddit on Kaggle, else ../data/remark_reddit
# Steps: ReMaRk data_release (git) -> post text by id from Arctic Shift -> Goodreads titles for the book catalogs
# (UCSD Goodreads graph, ~2.2 GB download) -> datasets (incl. ReMaRk union GRPO set per request) -> build_summary.json.
# DATASETS="music books books_late" selects what to build.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -d /kaggle/working ]]; then OUT=${OUT:-/kaggle/working/remark_reddit}; TMP=${TMP:-/tmp/remark_build}
else OUT=${OUT:-$REPO/../data/remark_reddit}; TMP=${TMP:-$REPO/../data/remark_build_tmp}; fi
DATASETS=${DATASETS:-music books books_late}
GR=https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads
PY=${PY:-python3}
mkdir -p "$OUT" "$TMP"
log() { echo "[$(date +%H:%M:%S)] $*"; }
quiet() { grep -v 'Saving the dataset' || true; }

$PY -c "import datasets, editdistance, regex" 2>/dev/null || $PY -m pip install -q "datasets>=4,<5" editdistance regex
REL=${REL:-$TMP/ReMaRk/data_release}  # REL=<existing ReMaRk/data_release> skips the clone
[[ -d $REL ]] || git clone -q --depth 1 https://github.com/LiXin97/ReMaRk.git "$TMP/ReMaRk"
log "ReMaRk release at $REL ($(git -C "$REL" log -1 --format='%h %cs'))"

if [[ " $DATASETS " == *" books"* ]]; then
  for f in goodreads_book_works.json.gz goodreads_books.json.gz; do
    [[ -s $TMP/$f ]] || { log "download $f"; curl -sSfL --retry 5 -o "$TMP/$f.part" "$GR/$f" && mv "$TMP/$f.part" "$TMP/$f"; }
  done
  log "Goodreads titles"
  $PY "$REPO/remark_data/goodreads_titles.py" --catalog "$REL/books_catalog.tsv" --catalog "$REL/books_late_catalog.tsv" \
    --works "$TMP/goodreads_book_works.json.gz" --books "$TMP/goodreads_books.json.gz" --out_dir "$OUT/goodreads_titles"
fi

for d in $DATASETS; do
  log "build $d"
  extra=()
  [[ $d == music ]] || extra=(--titles "$OUT/goodreads_titles/${d}_titles.tsv")
  $PY "$REPO/remark_data/build_remark_reddit.py" --dataset "$d" --release "$REL" --out "$OUT" "${extra[@]}" 2>&1 | quiet
done

$PY - "$OUT" "$REL/export_counts.json" <<'EOF'
import json, os, sys
out, ref = sys.argv[1], json.load(open(sys.argv[2]))
s = {"remark_export_counts": ref}
gt = os.path.join(out, "goodreads_titles", "goodreads_titles_stats.json")
if os.path.exists(gt):
    s["goodreads_titles"] = json.load(open(gt))
for d in ("music", "books", "books_late"):
    p = os.path.join(out, d, "build_stats.json")
    if os.path.exists(p):
        s[d] = json.load(open(p))
        r = ref[d]
        b = s[d]
        s[d]["match_remark"] = {k: (b.get(k), r[v]) for k, v in [("sft_rows_train", "sft_train_rows"),
            ("grpo_rows_train", "train_responses"), ("sft_rows_validation", "validation_responses"),
            ("sft_rows_test", "test_responses"), ("requests_test", "test_requests"), ("catalog_items", "catalog")]}
json.dump(s, open(os.path.join(out, "build_summary.json"), "w"), indent=1)
for d in ("music", "books", "books_late"):
    if d in s:
        print(d, json.dumps(s[d]["match_remark"]))
EOF
log "done: $OUT"
