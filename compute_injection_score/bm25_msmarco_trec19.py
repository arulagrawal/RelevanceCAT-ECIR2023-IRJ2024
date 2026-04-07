import json, os, tqdm
from pyserini.index.lucene import LuceneIndexReader as IndexReader
from pyserini.pyclass import autoclass

index_reader = IndexReader.from_prebuilt_index('msmarco-v1-passage')
b = 0.68
k1 = 0.82
similarity_bm25 = autoclass('org.apache.lucene.search.similarities.BM25Similarity')(k1, b)

data_folder = 'train/msmarco-data'
filename = "msmarco-passagetest2019-top1000.tsv"
top1000_filepath = os.path.join(data_folder, filename)

scores = {}
with open(top1000_filepath, encoding="utf-8") as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True):
        qid, pid, query, passage = line.strip().split("\t")
        if qid not in scores:
            scores[qid] = {}
        score = index_reader.compute_query_document_score(pid, query, similarity=similarity_bm25)
        scores[qid][pid] = score

out_dir = "train/msmarco-data/injection_scores"
os.makedirs(out_dir, exist_ok=True)
scores_dict_path = os.path.join(out_dir, "3_trec19_bm25_scores.json")
with open(scores_dict_path, "w", encoding="utf-8") as fp:
    json.dump(scores, fp)

print(f"Saved {scores_dict_path}")
