"""
Compute SPLADE scores for MS MARCO validation triples.

3-phase approach matching the training script:
  Phase 1: Batch-compute SPLADE reps for all unique documents
  Phase 2: Batch-compute SPLADE reps for all unique queries
  Phase 3: Compute dot-product scores
"""
import gzip
import os
import tarfile
import tqdm
import json
import torch
import numpy as np
from scipy import sparse
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import LoggingHandler, util
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Select device
# Select device: CUDA (also covers AMD ROCm on Linux) > DirectML (AMD on Windows) > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    try:
        import torch_directml
        device = torch_directml.device()
        logging.info("Using DirectML (AMD GPU on Windows)")
    except ImportError:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
logging.info("Using device: {}".format(device))

# Load the SPLADE model
splade_model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(splade_model_name)
try:
    splade_model = AutoModelForMaskedLM.from_pretrained(
        splade_model_name,
        use_safetensors=True,
    ).to(device)
except OSError as err:
    raise RuntimeError(
        "SPLADE checkpoint must provide safetensors weights; convert the model or pin a revision that includes them."
    ) from err
splade_model.eval()

BATCH_SIZE = 64
CACHE_DIR = "score_files"
os.makedirs(CACHE_DIR, exist_ok=True)


def compute_splade_reps_batched(texts, ids, batch_size=BATCH_SIZE, max_length=256, desc="encoding"):
    """Compute SPLADE sparse representations in batches."""
    reps = {}
    for i in tqdm.trange(0, len(texts), batch_size, desc=desc):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        tokens = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)
        use_autocast = isinstance(device, str) and device in ("cuda", "mps")
        autocast_device = device if isinstance(device, str) else device.type
        with torch.no_grad(), torch.amp.autocast(device_type=autocast_device, enabled=use_autocast):
            output = splade_model(**tokens)
        vecs = torch.max(
            torch.log(1 + torch.relu(output.logits))
            * tokens["attention_mask"].unsqueeze(-1),
            dim=1,
        )[0].cpu().float().numpy()
        for j, id_ in enumerate(batch_ids):
            reps[id_] = sparse.csr_matrix(vecs[j])
    return reps


# ============================================================
# Load MS MARCO data
# ============================================================
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)

corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage

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
# Parse validation triples to find unique IDs
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
unique_dids = list(set(did for dids in query_doc_pairs.values() for did in dids))
logging.info("Unique queries: {}  |  Unique documents: {}".format(len(unique_qids), len(unique_dids)))

# ============================================================
# Phase 1: Compute document representations
# ============================================================
# Reuse doc reps from training script if available
doc_reps_path = os.path.join(CACHE_DIR, "splade_doc_reps")
if os.path.exists(doc_reps_path + ".data.npz"):
    logging.info("Phase 1: Loading cached document reps from training phase...")
    data = sparse.load_npz(doc_reps_path + ".data.npz")
    with open(doc_reps_path + ".ids.json", "r") as f:
        cached_ids = json.load(f)
    doc_reps = {id_: data[i] for i, id_ in enumerate(cached_ids)}
    # Check if all needed docs are cached
    missing_dids = [did for did in unique_dids if did not in doc_reps]
    if missing_dids:
        logging.info("Computing {} missing document reps...".format(len(missing_dids)))
        missing_texts = [corpus[did] for did in missing_dids]
        extra_reps = compute_splade_reps_batched(missing_texts, missing_dids, desc="extra docs")
        doc_reps.update(extra_reps)
else:
    logging.info("Phase 1: Computing SPLADE reps for {} documents...".format(len(unique_dids)))
    doc_texts = [corpus[did] for did in unique_dids]
    doc_reps = compute_splade_reps_batched(doc_texts, unique_dids, max_length=256, desc="Phase 1: docs")

# ============================================================
# Phase 2: Compute query representations
# ============================================================
logging.info("Phase 2: Computing SPLADE reps for {} queries...".format(len(unique_qids)))
query_texts = [queries[qid] for qid in unique_qids]
query_reps = compute_splade_reps_batched(query_texts, unique_qids, max_length=64, desc="Phase 2: queries")

# ============================================================
# Phase 3: Compute dot-product scores
# ============================================================
logging.info("Phase 3: Computing dot-product scores...")
scores = {}
for qid in tqdm.tqdm(unique_qids, desc="Phase 3: dot products"):
    q_rep = query_reps[qid]
    scores[qid] = {}
    for did in query_doc_pairs[qid]:
        d_rep = doc_reps[did]
        score = float((q_rep.multiply(d_rep)).sum())
        scores[qid][did] = score

scores_dict_path = os.path.join(CACHE_DIR, "5_splade_scores_train-eval_triples.json")
with open(scores_dict_path, "w+") as fp:
    json.dump(scores, indent=True, fp=fp)
logging.info("Saved scores to {}".format(scores_dict_path))
