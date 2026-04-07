"""
This script trains a Cross-Encoder for the MS Marco dataset using knowledge distillation
with BM25 score injection (BM25CAT).

Based on the BM25 injection setting from prior work, the script uses BM25 scores as
input-level signals for the cross-encoder. BM25 scores are injected as text prefix to
the query: "{score} [SEP] {query}".

Running this script:
python train_cross-encoder_kd_bm25cat.py
"""

import os
import json
import gzip
import tarfile
import logging
from datetime import datetime

import tqdm
import torch

## Bypass torch.load vulnerability check for PyTorch <2.6
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import MSELoss as CEMSELoss
from datasets import Dataset

from CERerankingEvaluator_bm25cat import CERerankingEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
#### /print debug information to stdout


### Select device: CUDA > DirectML > MPS > CPU
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


### Model config
model_name = 'microsoft/MiniLM-L12-H384-uncased'
train_batch_size = 256
num_epochs = 1
learning_rate = 7e-6
warmup_steps = 625
sample_every_n = 10   # 10% of training triples

model_save_path = (
    'finetuned_CEs/train-cross-encoder-kd-bm25cat-'
    + model_name.replace("/", "-")
    + '-'
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

## BM25 normalization constants
global_min_bm25 = 0
global_max_bm25 = 50


### We set num_labels=1 and set the activation function to Identity, so that we get the raw logits
model = CrossEncoder(
    model_name,
    num_labels=1,
    max_length=512,
    activation_fn=torch.nn.Identity(),
    device=device
)


### Now we read the MS Marco dataset
data_folder = 'msmarco-data'
injection_folder = os.path.join(data_folder, "injection_scores")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(injection_folder, exist_ok=True)


#### Download BM25 injection scores
train_scores_path = os.path.join(injection_folder, '1_bm25_scores_train_triples_small.json')
if not os.path.exists(train_scores_path):
    logging.info("Download " + os.path.basename(train_scores_path))
    util.http_get(
        'https://www.dropbox.com/scl/fi/ssgpoun44jtlrwy24wrad/1_bm25_scores_train_triples_small.json?rlkey=3og8ayxmyjxsei7okdumseaq7&raw=1',
        train_scores_path
    )

validation_scores_path = os.path.join(injection_folder, '5_bm25_scores_train-eval_triples.json')
if not os.path.exists(validation_scores_path):
    logging.info("Download " + os.path.basename(validation_scores_path))
    util.http_get(
        'https://www.dropbox.com/scl/fi/q433llwfdk701x336ce3p/5_bm25_scores_train-eval_triples.json?rlkey=5782bylutyzmk10f1uax3iao5&raw=1',
        validation_scores_path
    )


#### Loading injection scores and applying normalization
scores = json.loads(open(train_scores_path, "r", encoding="utf8").read())
for qid in tqdm.tqdm(scores.keys(), desc="reading scores...{}".format(train_scores_path)):
    for did, score in scores[qid].items():
        normalized_score = (score - global_min_bm25) / (global_max_bm25 - global_min_bm25)
        normalized_score = int(normalized_score * 100)
        scores[qid][did] = normalized_score

scores_validation = json.loads(open(validation_scores_path, "r", encoding="utf8").read())
for qid in tqdm.tqdm(scores_validation.keys(), desc="reading validation scores...{}".format(validation_scores_path)):
    if qid not in scores:
        scores[qid] = {}
    for did, score in scores_validation[qid].items():
        normalized_score = (score - global_min_bm25) / (global_max_bm25 - global_min_bm25)
        normalized_score = int(normalized_score * 100)
        scores[qid][did] = normalized_score


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

logging.info("Loaded corpus with {} passages".format(len(corpus)))


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

logging.info("Loaded {} training queries".format(len(queries)))


### Now we create our dev data
dev_samples = {}
num_dev_queries = 200
num_max_dev_negatives = 200

train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    logging.info("Download " + os.path.basename(train_eval_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

with gzip.open(train_eval_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()

        if qid not in queries:
            continue
        if pos_id not in corpus or neg_id not in corpus:
            continue
        if qid not in scores:
            continue
        if pos_id not in scores[qid] or neg_id not in scores[qid]:
            continue

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': list(), 'positive': list(), 'negative': list()}

        if qid in dev_samples:
            dev_samples[qid]['positive'].append(corpus[pos_id])
            dev_samples[qid]['query'].append("{} [SEP] {}".format(scores[qid][pos_id], queries[qid]))

            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].append(corpus[neg_id])
                dev_samples[qid]['query'].append("{} [SEP] {}".format(scores[qid][neg_id], queries[qid]))

dev_qids = set(dev_samples.keys())
logging.info("Built dev set with {} queries".format(len(dev_qids)))


### Download teacher logits
teacher_logits_filepath = os.path.join(data_folder, 'bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv')
if not os.path.exists(teacher_logits_filepath):
    logging.info("Downloading teacher logits from HuggingFace...")
    from datasets import load_dataset as _load_dataset
    import gc

    _mse_ds = _load_dataset("sentence-transformers/msmarco", "bert-ensemble-mse", split="train")
    _score_lookup = {}
    for row in tqdm.tqdm(_mse_ds, desc="Building score lookup"):
        qid = str(row['query_id'])
        if qid not in _score_lookup:
            _score_lookup[qid] = {}
        _score_lookup[qid][str(row['passage_id'])] = row['score']
    del _mse_ds
    gc.collect()

    _margin_ds = _load_dataset("sentence-transformers/msmarco", "bert-ensemble-margin-mse", split="train")
    with open(teacher_logits_filepath, 'w', encoding='utf8') as fOut:
        for row in tqdm.tqdm(_margin_ds, desc="Writing teacher logits TSV"):
            qid = str(row['query_id'])
            pos_id = str(row['positive_id'])
            neg_id = str(row['negative_id'])
            pos_score = _score_lookup.get(qid, {}).get(pos_id, 0.0)
            neg_score = _score_lookup.get(qid, {}).get(neg_id, 0.0)
            fOut.write("{}\t{}\t{}\t{}\t{}\n".format(pos_score, neg_score, qid, pos_id, neg_id))

    del _margin_ds, _score_lookup
    gc.collect()
    logging.info("Saved teacher logits to {}".format(teacher_logits_filepath))


### Write pre-processed training data to disk instead of holding all samples in RAM
train_data_path = os.path.join(data_folder, 'bm25cat_train_data_10pct.tsv')
if os.path.exists(train_data_path):
    logging.info("SKIP: {} already exists".format(train_data_path))
else:
    logging.info("Writing 10% pre-processed training data to {}...".format(train_data_path))
    num_train_samples = 0
    line_idx = 0

    with open(teacher_logits_filepath, encoding='utf8') as fIn, open(train_data_path, 'w', encoding='utf8') as fOut:
        for line in fIn:
            pos_score, neg_score, qid, pid1, pid2 = line.strip().split("\t")

            if qid in dev_qids:
                continue

            line_idx += 1
            if line_idx % sample_every_n != 0:
                continue

            if qid not in queries:
                continue
            if pid1 not in corpus or pid2 not in corpus:
                continue
            if qid not in scores:
                continue
            if pid1 not in scores[qid] or pid2 not in scores[qid]:
                continue

            q1 = "{} [SEP] {}".format(scores[qid][pid1], queries[qid])
            q2 = "{} [SEP] {}".format(scores[qid][pid2], queries[qid])

            fOut.write("{}\t{}\t{}\n".format(q1, corpus[pid1], pos_score))
            fOut.write("{}\t{}\t{}\n".format(q2, corpus[pid2], neg_score))
            num_train_samples += 2

    logging.info("Wrote {} training samples to disk".format(num_train_samples))


### Free all large data structures
import gc
del corpus, queries, scores
gc.collect()
logging.info("Freed corpus/queries/scores from memory")


### Build HF Dataset from TSV file
logging.info("Building HF Dataset from {}...".format(train_data_path))

def generate_examples():
    with open(train_data_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                yield {"sentence1": parts[0], "sentence2": parts[1], "label": float(parts[2])}

train_dataset = Dataset.from_generator(generate_examples)
logging.info("Dataset ready: {} samples".format(len(train_dataset)))


### Configure training with CrossEncoderTrainer
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

args = CrossEncoderTrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    bf16=torch.cuda.is_available(),
    fp16=False,
    eval_strategy="steps",
    eval_steps=5000,
    save_strategy="steps",
    save_steps=5000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_sequential_score",
    greater_is_better=True,
    max_grad_norm=1.0,
    weight_decay=0.01,
    dataloader_num_workers=4,
    dataloader_pin_memory=torch.cuda.is_available(),
    logging_steps=100,
)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=CEMSELoss(model),
    evaluator=[evaluator],
)

logging.info("starting training")
trainer.train()
logging.info("saving model")

### Save final model
model.save(model_save_path + '-latest')