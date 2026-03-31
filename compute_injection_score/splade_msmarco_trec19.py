"""
Compute SPLADE scores for TREC DL'19 top-1000 candidates using Pyserini's pre-built index.
"""
import json, tqdm, os
import numpy as np
from pyserini.search.lucene import LuceneImpactSearcher
from sentence_transformers import LoggingHandler
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Initialize SPLADE searcher with pre-built index
logging.info("Loading pre-built SPLADE index...")
searcher = LuceneImpactSearcher.from_prebuilt_index(
    'msmarco-v1-passage.splade-pp-ed',
    'naver/splade-cocondenser-ensembledistil'
)
logging.info("SPLADE searcher ready.")

CACHE_DIR = "score_files"
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# Parse top-1000 file to find unique queries and their documents
# ============================================================
data_folder = 'msmarco-data'
filename = "msmarco-passagetest2019-top1000.tsv"
top1000_filepath = os.path.join(data_folder, filename)

logging.info("Parsing top-1000 file...")
unique_queries = {}   # {qid: query_text}
query_doc_pairs = {}  # {qid: set(pids)}

with open(top1000_filepath) as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True, desc="parsing top-1000"):
        qid, pid, query, passage = line.strip().split("\t")
        unique_queries[qid] = query
        if qid not in query_doc_pairs:
            query_doc_pairs[qid] = set()
        query_doc_pairs[qid].add(pid)

logging.info("Unique queries: {}  |  Unique documents: {}".format(
    len(unique_queries), sum(len(d) for d in query_doc_pairs.values())))

# ============================================================
# Compute SPLADE scores via index search
# ============================================================
logging.info("Computing SPLADE scores via pre-built index...")
scores = {}
missing_count = 0
total_count = 0

for qid, query_text in tqdm.tqdm(unique_queries.items(), desc="SPLADE scoring"):
    needed_dids = query_doc_pairs[qid]

    hits = searcher.search(query_text, k=1000)
    hit_scores = {hit.docid: hit.score for hit in hits}

    scores[qid] = {}
    for did in needed_dids:
        score = hit_scores.get(did, 0.0)
        if did not in hit_scores:
            missing_count += 1
        scores[qid][did] = float(score)
        total_count += 1

logging.info("Done. {} / {} pairs had no SPLADE score (assigned 0).".format(missing_count, total_count))

scores_dict_path = os.path.join(CACHE_DIR, "3_trec19_splade_scores.json")
with open(scores_dict_path, "w+") as fp:
    json.dump(scores, indent=True, fp=fp)
logging.info("Saved scores to {}".format(scores_dict_path))
