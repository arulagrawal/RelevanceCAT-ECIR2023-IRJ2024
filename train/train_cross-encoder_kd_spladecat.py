"""
This script trains a Cross-Encoder for the MS Marco dataset using knowledge distillation
with SPLADE score injection (SPLADECAT).

Based on train_cross-encoder_kd_bm25cat.py, but uses SPLADE scores instead of BM25.
SPLADE scores are injected as text prefix to the query: "{score} [SEP] {query}".

Running this script:
python train_cross-encoder_kd_spladecat.py
"""
import os
# Bypass torch.load vulnerability check for PyTorch <2.6 (e.g. torch-directml pinned to 2.4)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from CERerankingEvaluator_bm25cat import CERerankingEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import torch
import json

# SPLADE normalization constants (from empirical score distribution)
global_min_splade = 0
global_max_splade = 125807  # P99 of training scores

scores_path = "score_files/1_splade_scores_train_triples_small_gpu.json"
scores = json.loads(open(scores_path, "r").read())
for qid in tqdm.tqdm(scores.keys(), desc="reading scores...{}".format(scores_path)):
    for did, score in scores[qid].items():
        normalized_score = (score - global_min_splade) / (global_max_splade - global_min_splade)
        normalized_score = int(normalized_score * 100)
        scores[qid][did] = normalized_score

validation_scores_path = "score_files/5_splade_scores_train-eval_triples.json"
scores_validation = json.loads(open(validation_scores_path, "r").read())
for qid in tqdm.tqdm(scores_validation.keys(), desc="reading validation scores...{}".format(validation_scores_path)):
    if qid not in scores:
        scores[qid] = {}
    for did, score in scores_validation[qid].items():
        normalized_score = (score - global_min_splade) / (global_max_splade - global_min_splade)
        normalized_score = int(normalized_score * 100)
        scores[qid][did] = normalized_score

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


# Select device: CUDA > DirectML (AMD on Windows) > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    try:
        import torch_directml
        device = str(torch_directml.device())
    except ImportError:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
logging.info("Using device: {}".format(device))

# First, we define the transformer model we want to fine-tune
model_name = 'microsoft/MiniLM-L12-H384-uncased'
train_batch_size = 32
num_epochs = 1
model_save_path = 'finetuned_CEs/train-cross-encoder-kd-spladecat-' + model_name.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# We set num_labels=1 and set the activation function to Identity, so that we get the raw logits
model = CrossEncoder(model_name, num_labels=1, max_length=512, default_activation_function=torch.nn.Identity(), device=device)


### Now we read the MS Marco dataset
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)


#### Read the corpus files, that contain all the passages. Store them in the corpus dict
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


### Read the train queries, store in queries dict
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


### Now we create our dev data
train_samples = []
dev_samples = {}

# We use 200 random queries from the train set for evaluation during training
# Each query has at least one relevant and up to 200 irrelevant (negative) passages
num_dev_queries = 200
num_max_dev_negatives = 200

train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    logging.info("Download " + os.path.basename(train_eval_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': list(), 'positive': list(), 'negative': list()}

        if qid in dev_samples:
            dev_samples[qid]['positive'].append(corpus[pos_id])
            dev_samples[qid]['query'].append("{} [SEP] {}".format(scores[qid][pos_id], queries[qid]))

            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].append(corpus[neg_id])
                dev_samples[qid]['query'].append("{} [SEP] {}".format(scores[qid][neg_id], queries[qid]))


dev_qids = set(dev_samples.keys())

# Read our training file
# As input examples, we provide the (query, passage) pair together with the logits score from the teacher ensemble
teacher_logits_filepath = os.path.join(data_folder, 'bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv')
train_samples = []
if not os.path.exists(teacher_logits_filepath):
    util.http_get('https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1', teacher_logits_filepath)

with open(teacher_logits_filepath) as fIn:
    for line in fIn:
        pos_score, neg_score, qid, pid1, pid2 = line.strip().split("\t")

        if qid in dev_qids:  # Skip queries in our dev dataset
            continue

        train_samples.append(InputExample(texts=["{} [SEP] {}".format(scores[qid][pid1], queries[qid]), corpus[pid1]], label=float(pos_score)))
        train_samples.append(InputExample(texts=["{} [SEP] {}".format(scores[qid][pid2], queries[qid]), corpus[pid2]], label=float(neg_score)))

# Free large intermediate data structures before model training
import gc
del corpus, queries, scores
gc.collect()
logging.info("Freed corpus/queries/scores from memory. train_samples: {}".format(len(train_samples)))

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)

# We add an evaluator, which evaluates the performance during training
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

# Configure the training
warmup_steps = 5000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          loss_fct=torch.nn.MSELoss(),
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr': 7e-6},
          use_amp=True)

# Save latest model
model.save(model_save_path + '-latest')
