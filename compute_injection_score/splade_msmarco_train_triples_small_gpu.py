"""
Compute SPLADE scores for MS MARCO training triples using Pyserini's pre-built index.

Uses LuceneImpactSearcher with the pre-built SPLADE index (msmarco-v1-passage.splade-pp-ed)
so no document encoding is needed. Only queries are encoded (via the SPLADE model),
and document scores are looked up from the index.

The pre-built index (~5GB) is auto-downloaded on first run.
"""
import os, tarfile, tqdm, json, numpy as np
from pyserini.search.lucene import LuceneImpactSearcher
from sentence_transformers import LoggingHandler, util
import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Initialize SPLADE searcher with pre-built index
# Auto-downloads the index and query encoder on first use
logging.info("Loading pre-built SPLADE index (will download ~5GB on first run)...")
searcher = LuceneImpactSearcher.from_prebuilt_index(
    'msmarco-v1-passage.splade-pp-ed',
    'naver/splade-cocondenser-ensembledistil'
)
logging.info("SPLADE searcher ready.")

CACHE_DIR = "score_files"
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# Load MS MARCO data
# ============================================================
data_folder = "msmarco-data"
os.makedirs(data_folder, exist_ok=True)

# Read queries
queries = {}
queries_filepath = os.path.join(data_folder, "queries.train.tsv")
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, "queries.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
            tar_filepath,
        )
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)
with open(queries_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

# Read training triples (teacher scores)
train_filepath = os.path.join(
    data_folder, "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv"
)
if not os.path.exists(train_filepath):
    logging.info("Download teacher scores")
    util.http_get(
        "https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1",
        train_filepath,
    )

# Parse training triples to find all unique query IDs and their associated doc IDs
logging.info("Parsing training triples...")
query_doc_pairs = {}  # {qid: set(dids)}
with open(train_filepath, "rt") as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True, desc="reading triples"):
        pos_score, neg_score, qid, pos_id, neg_id = line.strip().split("\t")
        if qid not in query_doc_pairs:
            query_doc_pairs[qid] = set()
        query_doc_pairs[qid].add(pos_id)
        query_doc_pairs[qid].add(neg_id)

unique_qids = list(query_doc_pairs.keys())
total_pairs = sum(len(dids) for dids in query_doc_pairs.values())
logging.info("Unique queries: {}  |  Total query-doc pairs: {}".format(len(unique_qids), total_pairs))

# ============================================================
# Compute SPLADE scores via index search
# ============================================================
scores_dict_path = os.path.join(CACHE_DIR, "1_splade_scores_train_triples_small_gpu.json")
logging.info("Computing SPLADE scores via pre-built index...")

scores_dict = {}
all_scores = []
missing_count = 0

for qid in tqdm.tqdm(unique_qids, desc="SPLADE scoring"):
    query_text = queries[qid]
    needed_dids = query_doc_pairs[qid]

    # Search the index — returns scored results for this query
    hits = searcher.search(query_text, k=1000)
    hit_scores = {hit.docid: hit.score for hit in hits}

    scores_dict[qid] = {}
    for did in needed_dids:
        score = hit_scores.get(did, 0.0)
        if did not in hit_scores:
            missing_count += 1
        scores_dict[qid][did] = float(score)
        all_scores.append(float(score))

logging.info("Done. {} / {} pairs had no SPLADE score (assigned 0).".format(missing_count, total_pairs))

with open(scores_dict_path, "w+") as fp:
    json.dump(scores_dict, indent=True, fp=fp)
logging.info("Saved scores to {}".format(scores_dict_path))

# Print score statistics for normalization calibration
all_scores = np.array(all_scores)
print("\n=== SPLADE Score Statistics (for normalization) ===")
print("Count:   {}".format(len(all_scores)))
print("Zeros:   {} ({:.1f}%)".format(missing_count, 100 * missing_count / len(all_scores)))
print("Min:     {:.4f}".format(np.min(all_scores)))
print("Max:     {:.4f}".format(np.max(all_scores)))
print("Mean:    {:.4f}".format(np.mean(all_scores)))
print("Median:  {:.4f}".format(np.median(all_scores)))
print("P1:      {:.4f}".format(np.percentile(all_scores, 1)))
print("P5:      {:.4f}".format(np.percentile(all_scores, 5)))
print("P95:     {:.4f}".format(np.percentile(all_scores, 95)))
print("P99:     {:.4f}".format(np.percentile(all_scores, 99)))
print("=== Use these to set global_min_splade and global_max_splade ===")
