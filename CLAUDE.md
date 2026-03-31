# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RelevanceCAT is a research project (ECIR 2023 / IRJ 2024) that improves Cross-Encoder re-rankers for information retrieval by injecting relevance scores (BM25 and/or DPR) into the model input. It demonstrates that prepending normalized retrieval scores to query text significantly improves MiniLM-based cross-encoder re-ranking on MS MARCO.

## Running the Pipeline

This is a research codebase with no package manager or build system. Scripts are run directly with Python. Dependencies include `torch`, `transformers`, `sentence-transformers`, `pyserini`, `pytrec_eval`, `numpy`, and `tqdm`.

### 1. Compute Injection Scores
```bash
python compute_injection_score/bm25_msmarco_train_triples_small.py
python compute_injection_score/bm25_msmarco_validation_set.py
python compute_injection_score/dpr_msmarco_train_triples_small_gpu.py  # requires GPU
python compute_injection_score/bm25_msmarco_DEV.py
```

### 2. Train Models
```bash
python train/train_ms-marco-MiniLM-L-12_v3_bm25.py        # v3: BM25 input injection (recommended)
python train/train_ms-marco-MiniLM-L-12-v2_1_bm25added.py  # v2.1: BM25 in loss function
python train/train_cross-encoder_kd_bm25cat.py              # Knowledge distillation with BM25CAT
python train/train_cross-encoder_kd.py                      # Standard KD baseline
```

### 3. Evaluate
```bash
python evaluation/eval_trec19-MiniLM-L-12-v3.py             # v3 on TREC DL'19
python evaluation/eval_trec19-MiniLM-L-12-v2_1_bm25added.py # v2.1 on TREC DL'19
python evaluation/eval_trec19-MiniLM-L-12-v2.py             # baseline on TREC DL'19
```

## Architecture

The pipeline has three phases: **score computation → training → evaluation**.

**Score Injection** — the core innovation has two approaches:
- **Input-level (v3):** Normalized score is prepended to query text: `"{score} [SEP] {query}"`. The transformer learns to use the score token as a feature.
- **Loss-level (v2.1):** BM25 scores are blended with teacher ensemble logits in the loss function, keeping input unchanged.

**CrossEncoder** — training scripts contain a custom CrossEncoder implementation (not the sentence-transformers one) built on `transformers`. It handles tokenization, training with BCE loss against teacher logits (knowledge distillation), AMP, gradient accumulation, and checkpoint saving.

**Score normalization:**
- BM25 (Pyserini, k1=0.82, b=0.68): normalized to 0–100 integer scale for v3, 0–10 float scale for v2.1
- DPR (msmarco-distilbert-base-tas-b): dot-product similarity between embeddings

**Score file format:** JSON dicts keyed by query ID → passage ID → score: `{qid: {pid: score, ...}, ...}`

**Model:** `microsoft/MiniLM-L12-H384-uncased` with max 30 query tokens + 200 passage tokens. Training uses AdamW with warmup + linear decay, evaluated by MRR@10 every 5000 steps.

**Evaluation** uses `CERerankingEvaluator_bm25cat.py` for MRR@K and outputs TREC-formatted run files to `run_files/`.

## Data Dependencies

Datasets are auto-downloaded during training (MS MARCO collection, queries, triples, teacher logits from Zenodo). TREC DL'19/'20 qrels are needed for evaluation.
