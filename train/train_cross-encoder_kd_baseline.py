"""
Baseline Cross-Encoder training with knowledge distillation (NO score injection).

This is the control experiment: same setup as SPLADECAT but without injecting
any retrieval scores into the query. Used to measure whether SPLADE injection
actually improves re-ranking.

Running this script:
python train_cross-encoder_kd_baseline.py
"""
import os
# Bypass torch.load vulnerability check for PyTorch <2.6
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import MSELoss as CEMSELoss
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from datasets import Dataset
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import torch

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Select device
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

# Model config
model_name = 'microsoft/MiniLM-L12-H384-uncased'
train_batch_size = 256
num_epochs = 1
model_save_path = 'finetuned_CEs/train-cross-encoder-kd-baseline-' + model_name.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model = CrossEncoder(model_name, num_labels=1, max_length=512, default_activation_function=torch.nn.Identity(), device=device)

### Now we read the MS Marco dataset
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)

#### Read the corpus files
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

### Read the train queries
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

### Dev data (no injection — plain queries)
dev_samples = {}
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
            dev_samples[qid] = {'query': queries[qid], 'positive': list(), 'negative': list()}
        if qid in dev_samples:
            dev_samples[qid]['positive'].append(corpus[pos_id])
            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].append(corpus[neg_id])

dev_qids = set(dev_samples.keys())

# Download teacher logits
teacher_logits_filepath = os.path.join(data_folder, 'bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv')
if not os.path.exists(teacher_logits_filepath):
    logging.info("Downloading teacher logits from HuggingFace...")
    from datasets import load_dataset as _load_dataset
    _mse_ds = _load_dataset("sentence-transformers/msmarco", "bert-ensemble-mse", split="train")
    _score_lookup = {}
    for row in tqdm.tqdm(_mse_ds, desc="Building score lookup"):
        qid = str(row['query_id'])
        if qid not in _score_lookup:
            _score_lookup[qid] = {}
        _score_lookup[qid][str(row['passage_id'])] = row['score']
    del _mse_ds
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
    import gc; gc.collect()

# Write 10% pre-processed training data (NO injection — plain queries)
SAMPLE_EVERY_N = 10
train_data_path = os.path.join(data_folder, 'baseline_train_data_10pct.tsv')
if os.path.exists(train_data_path):
    logging.info("SKIP: {} already exists".format(train_data_path))
else:
    logging.info("Writing 10% baseline training data to {}...".format(train_data_path))
    num_train_samples = 0
    line_idx = 0
    with open(teacher_logits_filepath, encoding='utf8') as fIn, open(train_data_path, 'w', encoding='utf8') as fOut:
        for line in fIn:
            pos_score, neg_score, qid, pid1, pid2 = line.strip().split("\t")
            if qid in dev_qids:
                continue
            line_idx += 1
            if line_idx % SAMPLE_EVERY_N != 0:
                continue
            # No injection — plain query text
            fOut.write("{}\t{}\t{}\n".format(queries[qid], corpus[pid1], pos_score))
            fOut.write("{}\t{}\t{}\n".format(queries[qid], corpus[pid2], neg_score))
            num_train_samples += 2
    logging.info("Wrote {} training samples to disk".format(num_train_samples))

# Free large data
import gc
del corpus, queries
gc.collect()
logging.info("Freed corpus/queries from memory")

# Build HF Dataset
logging.info("Building HF Dataset from {}...".format(train_data_path))


def generate_examples():
    with open(train_data_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                yield {"sentence1": parts[0], "sentence2": parts[1], "label": float(parts[2])}


train_dataset = Dataset.from_generator(generate_examples)
logging.info("Dataset ready: {} samples".format(len(train_dataset)))

# Training
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

args = CrossEncoderTrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    warmup_steps=625,
    learning_rate=7e-6,
    bf16=True,
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
    dataloader_pin_memory=True,
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
model.save(model_save_path + '-latest')
