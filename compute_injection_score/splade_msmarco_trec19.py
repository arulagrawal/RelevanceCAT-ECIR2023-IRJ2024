"""
Compute SPLADE scores for TREC DL'19 top-1000 candidates.

3-phase approach matching the training script:
  Phase 1: Batch-compute SPLADE reps for all unique documents
  Phase 2: Batch-compute SPLADE reps for all unique queries
  Phase 3: Compute dot-product scores
"""
import json, tqdm, os
import torch
import numpy as np
from scipy import sparse
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import LoggingHandler
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
# Parse top-1000 file to find unique queries and documents
# ============================================================
data_folder = 'msmarco-data'
filename = "msmarco-passagetest2019-top1000.tsv"
top1000_filepath = os.path.join(data_folder, filename)

logging.info("Parsing top-1000 file...")
unique_queries = {}   # {qid: query_text}
unique_docs = {}      # {pid: passage_text}
query_doc_pairs = {}  # {qid: set(pids)}

with open(top1000_filepath) as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True, desc="parsing top-1000"):
        qid, pid, query, passage = line.strip().split("\t")
        unique_queries[qid] = query
        unique_docs[pid] = passage
        if qid not in query_doc_pairs:
            query_doc_pairs[qid] = set()
        query_doc_pairs[qid].add(pid)

logging.info("Unique queries: {}  |  Unique documents: {}".format(
    len(unique_queries), len(unique_docs)))

# ============================================================
# Phase 1: Compute document representations
# ============================================================
doc_ids = list(unique_docs.keys())
doc_texts = [unique_docs[pid] for pid in doc_ids]
logging.info("Phase 1: Computing SPLADE reps for {} documents...".format(len(doc_ids)))
doc_reps = compute_splade_reps_batched(doc_texts, doc_ids, max_length=256, desc="Phase 1: docs")

# ============================================================
# Phase 2: Compute query representations
# ============================================================
query_ids = list(unique_queries.keys())
query_texts = [unique_queries[qid] for qid in query_ids]
logging.info("Phase 2: Computing SPLADE reps for {} queries...".format(len(query_ids)))
query_reps = compute_splade_reps_batched(query_texts, query_ids, max_length=64, desc="Phase 2: queries")

# ============================================================
# Phase 3: Compute dot-product scores
# ============================================================
logging.info("Phase 3: Computing dot-product scores...")
scores = {}
for qid in tqdm.tqdm(query_ids, desc="Phase 3: dot products"):
    q_rep = query_reps[qid]
    scores[qid] = {}
    for pid in query_doc_pairs[qid]:
        d_rep = doc_reps[pid]
        score = float((q_rep.multiply(d_rep)).sum())
        scores[qid][pid] = score

scores_dict_path = os.path.join(CACHE_DIR, "3_trec19_splade_scores.json")
with open(scores_dict_path, "w+") as fp:
    json.dump(scores, indent=True, fp=fp)
logging.info("Saved scores to {}".format(scores_dict_path))
