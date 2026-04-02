import json, sys, tqdm, tarfile, os, gzip, logging
from datetime import datetime
from pyserini.index.lucene import LuceneIndexReader as IndexReader
from pyserini import analysis, search
from pyserini.pyclass import autoclass
from sentence_transformers import LoggingHandler, util

index_reader = IndexReader.from_prebuilt_index('msmarco-v1-passage')
b = 0.68
k1 = 0.82
similarity_bm25 = autoclass('org.apache.lucene.search.similarities.BM25Similarity')(k1, b)

### Load data
data_folder = 'msmarco-data'
filename = "top1000.dev"
top1000_filepath = os.path.join(data_folder, filename)

scores = {}
with open(top1000_filepath) as fIn:
  for line in tqdm.tqdm(fIn, unit_scale=True):  #tsv: qid, pid, query, passage
    qid, pid, query, passage = line.strip().split("\t")
    if qid not in scores.keys():
        scores[qid] = {}
    score = index_reader.compute_query_document_score(pid, query, similarity=similarity_bm25)
    scores[qid][pid] = score


scores_dict_path = "2_msmarco_DEV_bm25_scores.json"
with open(scores_dict_path, "w+") as fp:
    json.dump(scores, indent=True, fp = fp)
