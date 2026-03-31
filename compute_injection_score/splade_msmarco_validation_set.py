"""
Compute SPLADE scores for MS MARCO validation triples using Pyserini's pre-built index.
"""
import gzip
import os
import tarfile
import tqdm
import json
import numpy as np
# Force safetensors loading to avoid torch.load vulnerability check (needed for PyTorch <2.6)
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
os.environ.setdefault("HF_SAFETENSORS", "1")
from transformers import AutoModelForMaskedLM
_orig_from_pretrained = AutoModelForMaskedLM.from_pretrained
AutoModelForMaskedLM.from_pretrained = classmethod(
    lambda cls, *args, **kwargs: _orig_from_pretrained(*args, **{**kwargs, "use_safetensors": True})
)
from pyserini.search.lucene import LuceneImpactSearcher
from sentence_transformers import LoggingHandler, util
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
# Load MS MARCO data
# ============================================================
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)

queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

# ============================================================
# Parse validation triples to find unique query-doc pairs
# ============================================================
train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    logging.info("Download " + os.path.basename(train_eval_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

logging.info("Parsing validation triples...")
query_doc_pairs = {}  # {qid: set(dids)}
with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()
        if qid not in query_doc_pairs:
            query_doc_pairs[qid] = set()
        query_doc_pairs[qid].add(pos_id)
        query_doc_pairs[qid].add(neg_id)

unique_qids = list(query_doc_pairs.keys())
logging.info("Unique queries: {}".format(len(unique_qids)))

# ============================================================
# Compute SPLADE scores via index search
# ============================================================
logging.info("Computing SPLADE scores via pre-built index...")
scores = {}
missing_count = 0
total_count = 0

for qid in tqdm.tqdm(unique_qids, desc="SPLADE scoring"):
    query_text = queries[qid]
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

scores_dict_path = os.path.join(CACHE_DIR, "5_splade_scores_train-eval_triples.json")
with open(scores_dict_path, "w+") as fp:
    json.dump(scores, indent=True, fp=fp)
logging.info("Saved scores to {}".format(scores_dict_path))
