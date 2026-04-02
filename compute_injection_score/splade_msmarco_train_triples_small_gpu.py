"""
Compute SPLADE scores for MS MARCO training triples using Pyserini's pre-built index.

Two-phase approach:
  Phase 1: Batch-encode all queries on GPU/MPS (~30 min) — saves to disk, resumable
  Phase 2: Search pre-built SPLADE index with pre-encoded queries (no BERT needed, fast)

The pre-built index (~5GB) is auto-downloaded on first run.
"""
import os, tarfile, tqdm, json, numpy as np, multiprocessing
# Bypass torch.load vulnerability check for PyTorch <2.6 (e.g. torch-directml pinned to 2.4)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pyserini.search.lucene import LuceneImpactSearcher
from sentence_transformers import LoggingHandler, util
import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Select device for query encoding
if torch.cuda.is_available():
    device = "cuda"
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
logging.info("Query encoding device: {}".format(device))

CACHE_DIR = "score_files"
os.makedirs(CACHE_DIR, exist_ok=True)
BATCH_SIZE = 128
QUERY_MAX_LENGTH = 64
ENCODE_CHUNK = 50000   # save encoded queries every 50K for resumability
SEARCH_CHUNK = 10000   # search 10K queries at a time

# SPLADE quantization parameters (must match Pyserini's SpladeQueryEncoder)
WEIGHT_RANGE = 5
QUANT_RANGE = 256

# ============================================================
# Load MS MARCO data
# ============================================================
data_folder = "msmarco-data"
os.makedirs(data_folder, exist_ok=True)

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

train_filepath = os.path.join(
    data_folder, "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv"
)
if not os.path.exists(train_filepath):
    logging.info("Download teacher scores")
    util.http_get(
        "https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1",
        train_filepath,
    )

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
# Phase 1: Batch-encode all queries on GPU
# ============================================================
encoded_queries_path = os.path.join(CACHE_DIR, "splade_encoded_queries.json")

if os.path.exists(encoded_queries_path):
    logging.info("Phase 1 SKIP: encoded queries already exist at {}".format(encoded_queries_path))
    with open(encoded_queries_path, "r") as f:
        encoded_queries = json.load(f)
else:
    logging.info("Phase 1: Batch-encoding {} queries on {}...".format(len(unique_qids), device))

    splade_model_name = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(splade_model_name)
    splade_model = _orig_from_pretrained(splade_model_name, use_safetensors=True).to(device)
    splade_model.eval()

    reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

    # Load partial encoded queries if resuming
    partial_enc_path = encoded_queries_path + ".partial"
    if os.path.exists(partial_enc_path):
        with open(partial_enc_path, "r") as f:
            encoded_queries = json.load(f)
        logging.info("Resuming with {} queries already encoded".format(len(encoded_queries)))
    else:
        encoded_queries = {}

    done_qids = set(encoded_queries.keys())
    remaining_qids_enc = [qid for qid in unique_qids if qid not in done_qids]
    logging.info("{} queries remaining to encode".format(len(remaining_qids_enc)))

    for chunk_start in tqdm.trange(0, len(remaining_qids_enc), ENCODE_CHUNK, desc="Phase 1: encoding chunks"):
        chunk_qids = remaining_qids_enc[chunk_start : chunk_start + ENCODE_CHUNK]
        chunk_texts = [queries[qid] for qid in chunk_qids]

        # Batch encode on GPU
        for batch_start in range(0, len(chunk_texts), BATCH_SIZE):
            batch_texts = chunk_texts[batch_start : batch_start + BATCH_SIZE]
            batch_qids = chunk_qids[batch_start : batch_start + BATCH_SIZE]

            tokens = tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=QUERY_MAX_LENGTH, padding=True, add_special_tokens=True,
            ).to(device)

            with torch.no_grad():
                logits = splade_model(**tokens)["logits"]

            # SPLADE aggregation: max over tokens of log(1 + ReLU(logits))
            reps = torch.max(
                torch.log(1 + torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1),
                dim=1,
            )[0].cpu().numpy()

            # Convert to quantized {term: weight} dicts (matching Pyserini format)
            for i, qid in enumerate(batch_qids):
                nonzero = np.nonzero(reps[i])[0]
                weights = reps[i][nonzero]
                encoded_queries[qid] = {
                    reverse_vocab[int(idx)]: int(round(float(w) / WEIGHT_RANGE * QUANT_RANGE))
                    for idx, w in zip(nonzero, weights)
                }

        # Save partial after each chunk
        with open(partial_enc_path, "w") as f:
            json.dump(encoded_queries, f)

    # Save final and clean up
    with open(encoded_queries_path, "w") as f:
        json.dump(encoded_queries, f)
    if os.path.exists(partial_enc_path):
        os.remove(partial_enc_path)

    # Free GPU memory
    del splade_model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Phase 1 done: encoded {} queries".format(len(encoded_queries)))

# ============================================================
# Phase 2: Batch search pre-built index with pre-encoded queries
# ============================================================
logging.info("Phase 2: Loading pre-built SPLADE index...")
searcher = LuceneImpactSearcher.from_prebuilt_index(
    'msmarco-v1-passage.splade-pp-ed',
    'naver/splade-cocondenser-ensembledistil'
)
min_idf = searcher.min_idf
idf = searcher.idf

scores_dict_path = os.path.join(CACHE_DIR, "1_splade_scores_train_triples_small_gpu.json")
threads = multiprocessing.cpu_count()
logging.info("SPLADE index ready. Batch searching with {} threads...".format(threads))

# Load partial results if resuming
partial_path = scores_dict_path + ".partial"
if os.path.exists(partial_path):
    with open(partial_path, "r") as f:
        scores_dict = json.load(f)
    logging.info("Resuming search with {} queries already scored".format(len(scores_dict)))
else:
    scores_dict = {}

done_qids = set(scores_dict.keys())
remaining_qids = [qid for qid in unique_qids if qid not in done_qids]
logging.info("{} queries remaining to search".format(len(remaining_qids)))

# Import Java types for batch search
from pyserini.pyclass import autoclass
JHashMap = autoclass('java.util.HashMap')
JInt = autoclass('java.lang.Integer')
JArrayList = autoclass('java.util.ArrayList')
JString = autoclass('java.lang.String')

for chunk_start in tqdm.trange(0, len(remaining_qids), SEARCH_CHUNK, desc="Phase 2: batch searching"):
    chunk_qids = remaining_qids[chunk_start : chunk_start + SEARCH_CHUNK]

    # Build Java ArrayLists for batch search
    query_lst = JArrayList()
    qid_lst = JArrayList()
    for qid in chunk_qids:
        jquery = JHashMap()
        for token, weight in encoded_queries[qid].items():
            if token in idf and idf[token] > min_idf:
                jquery.put(token, JInt(int(weight)))
        query_lst.add(jquery)
        qid_lst.add(JString(qid))

    # Java-side multi-threaded batch search — no Python loop overhead
    results = searcher.object.batch_search(query_lst, qid_lst, 1000, threads)

    # Extract results back to Python
    for entry in results.entrySet().toArray():
        qid = entry.getKey()
        hits = entry.getValue()
        hit_scores = {hit.docid: hit.score for hit in hits}
        scores_dict[qid] = {}
        for did in query_doc_pairs[qid]:
            scores_dict[qid][did] = float(hit_scores.get(did, 0.0))

    # Save partial results
    with open(partial_path, "w") as f:
        json.dump(scores_dict, f)

# Collect statistics
all_scores = []
missing_count = 0
for qid in unique_qids:
    for did in query_doc_pairs[qid]:
        score = scores_dict[qid][did]
        all_scores.append(score)
        if score == 0.0:
            missing_count += 1

logging.info("Done. {} / {} pairs had no SPLADE score (assigned 0).".format(missing_count, total_pairs))

with open(scores_dict_path, "w+") as fp:
    json.dump(scores_dict, indent=True, fp=fp)
if os.path.exists(partial_path):
    os.remove(partial_path)
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
