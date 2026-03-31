import sys, math, tqdm, json, pytrec_eval, gzip, os, tarfile, logging
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers import InputExample
from transformers import AutoTokenizer
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
"""# Initializing variables
"""
model_name = "finetuned_CEs/train-cross-encoder-kd-spladecat-microsoft-MiniLM-L12-H384-uncased/"  # TODO: update with actual path after training
fine_tuned_model_path = model_name
ranking_output_path = model_name + "trec19_spladecat.ranking"
base_write_path = ""
base_path = base_write_path + "msmarco-data/"
injection_scores_path = "compute_injection_score/score_files/"
qrel_path = base_path + "2019qrels-pass.txt"
top100_run_path = base_path + "msmarco-passagetest2019-top1000.tsv"
queries_path = base_path + "msmarco-test2019-queries.tsv"
scores_path = injection_scores_path + "3_trec19_splade_scores.json"

# SPLADE normalization constants - must match training script!
global_min_splade = 0
global_max_splade = 200  # TODO: update based on empirical analysis

pos_neg_ratio = 4
max_train_samples = 0 # full train set
valid_max_queries = 0 # full validation set
valid_max_negatives_per_query = 0 # full negatives per query
corpus_path = base_path + "collection_truncated.tsv"
triples_train_path = base_path + "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1"
triples_validation_path = base_path + "msmarco-qidpidtriples.rnd-shuf.train-eval.tsv"
max_length_query = 30
max_length_passage = 200
model_max_length = 230 + 3 + 3 # 3:[cls]query[sep]doc[sep]. 3 extra tokens for injection: score normally takes two tokens, plus one [sep]

print("fine_tuned_model_path {} | model_max_length {} | queries_path {} | ranking_output_path {} ".format(fine_tuned_model_path, model_max_length, queries_path, ranking_output_path))

scores = json.loads(open(scores_path, "r").read())
for qid in tqdm.tqdm(scores.keys(), desc = "reading scores...{}".format(scores_path)):
  for did, score in scores[qid].items():
    normalized_score = (score - global_min_splade) / (global_max_splade - global_min_splade)
    normalized_score = int(normalized_score * 100)
    scores[qid][did] = normalized_score

"""# CrossEncoder Class

"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator
class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                 default_activation_function = None):
        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config, ignore_mismatched_sizes = True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        input_was_string = False
        if isinstance(sentences[0], str):
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm.tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def save(self, path):
        if path is None:
            return
        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        return self.save(path)



"""# Evaluator Class"""

import numpy as np
import os
import csv
import pytrec_eval
import tqdm
from sentence_transformers import LoggingHandler, util
class CERerankingEvaluatorTest:
    def __init__(self, samples, qrel, all_metrics: set = {"recall.10"}, ranking_output_path: str = '', batch_size: int = 16):
        self.samples = samples
        self.all_metrics = all_metrics
        self.qrel = qrel
        self.ranking_output_path = ranking_output_path
        self.batch_size = batch_size
        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())
        self.mean_metrics = {}

    def rank(self, model) -> float:
        print("len: # queries: " + str(len(self.samples)))
        cnt = 0
        try:
            run = {}
            f_w = open(self.ranking_output_path, "w+")
            for instance in tqdm.tqdm(self.samples):
                cnt += 1
                print("cnt: ", cnt)
                qid = instance['qid']
                if qid not in run:
                  run[qid] = {}
                queries = instance['queries']
                docs = list(instance['docs'])
                ids = list(instance['docs_ids'])
                model_input = [[query, doc] for query, doc in zip(queries, docs)]
                if model.config.num_labels > 1:
                    pred_scores = model.predict(model_input, apply_softmax=True, batch_size = self.batch_size)[:, 1].tolist()
                else:
                    pred_scores = model.predict(model_input, batch_size = self.batch_size).tolist()
                for pred_score, did in zip(list(pred_scores), ids):
                    line = "{query_id} Q0 {document_id} {rank} {score} STANDARD\n".format(query_id=qid,
                                                                                          document_id=did,
                                                                                          rank="-10",
                                                                                          score=str(pred_score))
                    f_w.write(line)
                    run[qid][did] = float(pred_score)
            f_w.close()
            if self.qrel is not None and self.qrel != "None":
              evaluator = pytrec_eval.RelevanceEvaluator(self.qrel, self.all_metrics)
              scores = evaluator.evaluate(run)
              self.mean_metrics = {}
              metrics_string = ""
              for metric in list(self.all_metrics):
                  self.mean_metrics[metric] = np.mean([ele[metric.replace(".","_")] for ele in scores.values()])
                  metrics_string = metrics_string +  "{}: {} | ".format(metric, self.mean_metrics[metric])
              print("metrics eval: ", metrics_string)
        except Exception as e:
            logger.error("error: ", e)
        self.__fix_rank_filed()
        return self.mean_metrics

    def __fix_rank_filed(self, ranking_path = "", splittor = " ", return_lines = False, ranking_lines = None, remove_last_query = None):
        splittor = splittor
        if ranking_path == "":
          ranking_path = self.ranking_output_path
        else:
          ranking_path = ranking_path
        return_lines = return_lines
        ranking_lines = ranking_lines
        remove_last_query = remove_last_query

        re_rank_dict = {}
        if ranking_lines is None:
            ranking_lines = open(ranking_path, "r").readlines()
        for line in ranking_lines:
            line = line.strip()
            if len(line.split(splittor))<4: continue
            qid, q0, did, rank, score, runname = line.split(splittor)
            if qid not in re_rank_dict:
                re_rank_dict[qid]= {}
            re_rank_dict[qid][did] = float(score)

        json_query_docs_score = re_rank_dict
        list_of_tupple = []
        for query, docs_dict in json_query_docs_score.items():
            for doc_id, score in docs_dict.items():
                list_of_tupple.append((query, doc_id, float(score)))

        new_list = sorted(list_of_tupple, key=lambda element: (element[0], element[2]), reverse=True)
        out_lines = ""
        rank = 1
        cur_q_id =""
        find_duplicate = []
        for query, doc_id, score in new_list:
            dup = query+"_"+doc_id
            if query == doc_id:continue
            if cur_q_id == "":
                cur_q_id = query
            if cur_q_id != query:
                rank = 1
                cur_q_id = query
            line = "{query_id} Q0 {document_id} {rank} {score} STANDARD\n".format(query_id=query,
                                                                                        document_id=doc_id, rank=str(rank), score= str(score))
            rank += 1
            out_lines += line
        if return_lines==True:
            return out_lines
        else:
            f_qrel = open(ranking_path+"_rankfield_fixed", "w")
            f_qrel.write(out_lines)
            f_qrel.close()

"""#Data

## utils

### read collections
"""

def read_collection(f_path):
  corpus = {}
  with open(f_path, "r") as fp:
    for line in tqdm.tqdm(fp, desc="reading {}".format(f_path)):
      did, dtext = line.strip().split("\t")
      corpus[did] = dtext
  return corpus
from glob import glob

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side = "right")

def get_truncated_dict(id_content_dict, tokenizer, max_length):
  for id_, content, in tqdm.tqdm(id_content_dict.items()):
    truncated_content = tokenizer.batch_decode(tokenizer(content, padding=True, truncation=True, return_tensors="pt", max_length=max_length)['input_ids'], skip_special_tokens=True)[0]
    id_content_dict[id_] = truncated_content
  return id_content_dict

"""### reading top1000: utils"""

def read_top1000_run(f_path, corpus, queries, separator = " ", scores = {}, allowed_qids = None):
  samples = {}
  with open(f_path, "r") as fp:
    for line in tqdm.tqdm(fp, desc="reading {}".format(f_path)):
      qid, did, query, passage = line.strip().split("\t")
      if qid not in queries: continue
      if allowed_qids is not None and qid not in allowed_qids: continue
      query = queries[qid]
      if qid not in samples:
        samples[qid] = {'qid': qid , 'queries': list(), 'docs': list(), 'docs_ids': list()}
      samples[qid]['queries'].append("{} [SEP] {}".format(scores[qid][did], query))
      samples[qid]['docs'].append(corpus[did])
      samples[qid]['docs_ids'].append(did)
  return samples

"""## Reading data


"""### reading qrel"""

with open(qrel_path, 'r') as f_qrel:
    qrel = pytrec_eval.parse_qrel(f_qrel)

### reading corpus and queries and truncate it

queries = read_collection(queries_path)
corpus =  read_collection(corpus_path)

queries = get_truncated_dict(queries, tokenizer, max_length_query)


"""### reading top1000: main"""
allowed_qids = set({'19335', '47923', '87181', '87452', '104861', '130510', '131843', '146187', '148538', '156493', '168216', '182539', '183378', '207786', '264014', '359349', '405717', '443396', '451602', '489204', '490595', '527433', '573724', '833860', '855410', '915593', '962179', '1037798', '1063750', '1103812', '1106007', '1110199', '1112341', '1113437', '1114646', '1114819', '1115776', '1117099', '1121402', '1121709', '1124210', '1129237', '1133167'}) # judged qids need to be reranked!
test_samples = read_top1000_run(top100_run_path, corpus, queries, separator = " ", scores = scores, allowed_qids = allowed_qids)



"""# Evaluating"""

model = CrossEncoder(fine_tuned_model_path, num_labels=1, max_length=model_max_length)
batch_size = 64

evaluator = CERerankingEvaluatorTest(
    test_samples,
    qrel,
    all_metrics = {"ndcg_cut.10", "map_cut.1000", "recall.10"},
    ranking_output_path = ranking_output_path,
    batch_size = batch_size
)
measures_results = evaluator.rank(model)

print("measures_results: ", measures_results)
