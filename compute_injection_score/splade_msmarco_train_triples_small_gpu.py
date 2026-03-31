"""
Compute SPLADE scores for MS MARCO training triples.

Optimized 3-phase approach:
  Phase 1: Batch-compute SPLADE reps for all unique documents (saved in chunks to disk)
  Phase 2: Batch-compute SPLADE reps for all unique queries (save to disk)
  Phase 3: Compute dot-product scores (pure math, very fast)

Each phase is resumable — if interrupted, re-running skips completed chunks/phases.
"""
import torch
import os, tarfile, tqdm, json, numpy as np
from scipy import sparse
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import LoggingHandler, util
import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

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

# Tunable parameters
BATCH_SIZE = 128       # larger batches = better GPU/MPS utilization
DOC_MAX_LENGTH = 128   # MS MARCO passages avg ~55 words; 128 tokens covers >95%
QUERY_MAX_LENGTH = 64
CHUNK_SIZE = 500_000   # save doc reps to disk every 500K docs for resumability
CACHE_DIR = "score_files"
os.makedirs(CACHE_DIR, exist_ok=True)


def compute_splade_reps_batched(texts, ids, batch_size=BATCH_SIZE, max_length=128, desc="encoding"):
    """Compute SPLADE sparse representations for a list of texts.
    Returns a dict of {id: scipy.sparse.csr_matrix}."""
    all_reps = {}
    for i in tqdm.trange(0, len(texts), batch_size, desc=desc):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        # autocast only supported on cuda and mps
        use_autocast = isinstance(device, str) and device in ("cuda", "mps")
        with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=use_autocast):
            output = splade_model(**tokens)
        reps = torch.max(
            torch.log(1 + torch.relu(output.logits))
            * tokens["attention_mask"].unsqueeze(-1),
            dim=1,
        )[0].cpu().float().numpy()
        for j, id_ in enumerate(batch_ids):
            all_reps[id_] = sparse.csr_matrix(reps[j])
    return all_reps


def save_sparse_chunk(reps_dict, path):
    """Save a chunk of sparse representations to disk."""
    ids = list(reps_dict.keys())
    data = sparse.vstack([reps_dict[id_] for id_ in ids])
    sparse.save_npz(path + ".data.npz", data)
    with open(path + ".ids.json", "w") as f:
        json.dump(ids, f)
    logging.info("Saved chunk with {} reps to {}".format(len(ids), path))


def load_sparse_chunk(path):
    """Load a chunk of sparse representations from disk."""
    data = sparse.load_npz(path + ".data.npz")
    with open(path + ".ids.json", "r") as f:
        ids = json.load(f)
    return {id_: data[i] for i, id_ in enumerate(ids)}


def load_all_doc_chunks(chunk_dir):
    """Load all saved doc rep chunks and merge into one dict."""
    reps = {}
    chunk_idx = 0
    while True:
        path = os.path.join(chunk_dir, "chunk_{}".format(chunk_idx))
        if not os.path.exists(path + ".data.npz"):
            break
        chunk = load_sparse_chunk(path)
        reps.update(chunk)
        chunk_idx += 1
    logging.info("Loaded {} total doc reps from {} chunks".format(len(reps), chunk_idx))
    return reps


# ============================================================
# Load MS MARCO data
# ============================================================
data_folder = "msmarco-data"
os.makedirs(data_folder, exist_ok=True)

corpus = {}
collection_filepath = os.path.join(data_folder, "collection.tsv")
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, "collection.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
            tar_filepath,
        )
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)
with open(collection_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage

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
    data_folder, "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1"
)
if not os.path.exists(train_filepath):
    logging.info("Download teacher scores")
    util.http_get(
        "https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1",
        train_filepath,
    )

# Parse training triples to find all unique query and document IDs
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
unique_dids = list(set(did for dids in query_doc_pairs.values() for did in dids))
logging.info("Unique queries: {}  |  Unique documents: {}".format(len(unique_qids), len(unique_dids)))

# ============================================================
# Phase 1: Compute document representations (chunked + resumable)
# ============================================================
doc_chunks_dir = os.path.join(CACHE_DIR, "splade_doc_chunks")
os.makedirs(doc_chunks_dir, exist_ok=True)

# Figure out how many chunks are already done
completed_ids = set()
completed_chunks = 0
while os.path.exists(os.path.join(doc_chunks_dir, "chunk_{}.ids.json".format(completed_chunks))):
    with open(os.path.join(doc_chunks_dir, "chunk_{}.ids.json".format(completed_chunks))) as f:
        completed_ids.update(json.load(f))
    completed_chunks += 1

remaining_dids = [did for did in unique_dids if did not in completed_ids]

if remaining_dids:
    logging.info("Phase 1: {} docs already done in {} chunks, {} remaining...".format(
        len(completed_ids), completed_chunks, len(remaining_dids)))

    # Process remaining docs in chunks
    for chunk_start in range(0, len(remaining_dids), CHUNK_SIZE):
        chunk_idx = completed_chunks + (chunk_start // CHUNK_SIZE)
        chunk_dids = remaining_dids[chunk_start : chunk_start + CHUNK_SIZE]
        chunk_texts = [corpus[did] for did in chunk_dids]

        logging.info("Phase 1: Processing chunk {} ({} docs)...".format(chunk_idx, len(chunk_dids)))
        chunk_reps = compute_splade_reps_batched(
            chunk_texts, chunk_dids, max_length=DOC_MAX_LENGTH,
            desc="Phase 1 chunk {}".format(chunk_idx),
        )
        save_sparse_chunk(chunk_reps, os.path.join(doc_chunks_dir, "chunk_{}".format(chunk_idx)))
        del chunk_reps  # free memory before next chunk
else:
    logging.info("Phase 1 SKIP: all {} document reps already computed".format(len(unique_dids)))

# Load all doc reps for Phase 3
logging.info("Loading all document reps...")
doc_reps = load_all_doc_chunks(doc_chunks_dir)

# ============================================================
# Phase 2: Compute query representations
# ============================================================
query_reps_path = os.path.join(CACHE_DIR, "splade_query_reps")
if os.path.exists(query_reps_path + ".data.npz"):
    logging.info("Phase 2 SKIP: query reps already computed")
    query_reps = load_sparse_chunk(query_reps_path)
else:
    logging.info("Phase 2: Computing SPLADE reps for {} unique queries...".format(len(unique_qids)))
    query_texts = [queries[qid] for qid in unique_qids]
    query_reps = compute_splade_reps_batched(query_texts, unique_qids, max_length=QUERY_MAX_LENGTH, desc="Phase 2: queries")
    save_sparse_chunk(query_reps, query_reps_path)

# ============================================================
# Phase 3: Compute dot-product scores
# ============================================================
scores_dict_path = os.path.join(CACHE_DIR, "1_splade_scores_train_triples_small_gpu.json")
logging.info("Phase 3: Computing dot-product scores...")

scores_dict = {}
all_scores = []
for qid in tqdm.tqdm(unique_qids, desc="Phase 3: dot products"):
    q_rep = query_reps[qid]
    scores_dict[qid] = {}
    for did in query_doc_pairs[qid]:
        d_rep = doc_reps[did]
        score = float((q_rep.multiply(d_rep)).sum())
        scores_dict[qid][did] = score
        all_scores.append(score)

with open(scores_dict_path, "w+") as fp:
    json.dump(scores_dict, indent=True, fp=fp)
logging.info("Saved scores to {}".format(scores_dict_path))

# Print score statistics for normalization calibration
all_scores = np.array(all_scores)
print("\n=== SPLADE Score Statistics (for normalization) ===")
print("Count: {}".format(len(all_scores)))
print("Min:   {:.4f}".format(np.min(all_scores)))
print("Max:   {:.4f}".format(np.max(all_scores)))
print("Mean:  {:.4f}".format(np.mean(all_scores)))
print("Median:{:.4f}".format(np.median(all_scores)))
print("P1:    {:.4f}".format(np.percentile(all_scores, 1)))
print("P5:    {:.4f}".format(np.percentile(all_scores, 5)))
print("P95:   {:.4f}".format(np.percentile(all_scores, 95)))
print("P99:   {:.4f}".format(np.percentile(all_scores, 99)))
print("=== Use these to set global_min_splade and global_max_splade ===")
