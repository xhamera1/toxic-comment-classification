# Task 3 Plan: Optimization of Models Operating on Sequential Data

## Project Context

This project implements a multi-label text classification pipeline for toxic comment detection using the Jigsaw Toxic Comment Classification Challenge dataset. The objective is to compare and optimize classical sequential neural architectures (RNN, LSTM, GRU) and a modern Transformer-based model (BERT), then justify design decisions using robust evaluation metrics suitable for multi-label tasks.

## Deliverables

- Main artifact: `GGSN_Patryk_Chamera_Zad3.ipynb` (complete, self-contained notebook for submission).
- Supporting artifacts:
  - `models/` for model checkpoints and best weights.
  - `plots/` for training curves and comparison figures.
  - `PLAN.md` for implementation strategy and milestones.

## 1) Data Exploration (EDA) and Cleaning

### 1.1 Data Loading and Schema Validation

- Load:
  - `data/train.csv`
  - `data/test.csv`
  - `data/test_labels.csv`
  - `data/sample_submission.csv`
- Verify expected columns:
  - Text feature: `comment_text`
  - Labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Confirm data types, row counts, and missing value statistics.

### 1.2 Label Distribution and Multi-Label Characteristics

- Compute per-label prevalence and imbalance ratios.
- Analyze label co-occurrence matrix and frequent label combinations.
- Estimate the proportion of:
  - clean comments (all labels = 0),
  - single-label toxic comments,
  - multi-label toxic comments.

### 1.3 Text-Level Diagnostics

- Analyze comment length distributions (characters, words, tokens).
- Detect noisy artifacts (URLs, user mentions, repeated punctuation, rare symbols).
- Review representative examples for each label and major label combinations.

### 1.4 Cleaning Strategy

- Preserve semantics relevant to toxicity while reducing noise.
- Candidate transformations:
  - Unicode normalization,
  - lowercasing (for non-cased pipelines),
  - whitespace cleanup,
  - optional URL and username masking.
- Keep cleaning conservative for BERT to avoid harming pretrained tokenization behavior.

## 2) Text Preprocessing and Tokenization

### 2.1 Common Experimental Setup

- Define reproducible train/validation split (iterative stratification if available for multi-label stability).
- Set random seeds and deterministic behavior where possible.
- Prepare reusable helper functions for preprocessing, batching, and metrics.

### 2.2 Sequential Models (RNN/LSTM/GRU) Pipeline

- Tokenization options:
  - Keras `Tokenizer` with vocabulary cap (`num_words`),
  - optional OOV token handling.
- Sequence preparation:
  - pad/truncate to fixed maximum sequence length.
- Embeddings:
  - baseline: trainable embedding layer initialized randomly,
  - optional extension: pretrained embeddings (if time allows).

### 2.3 Transformer (BERT) Pipeline

- Use Hugging Face tokenizer (e.g., `bert-base-uncased`).
- Prepare attention masks and token type ids (if model requires).
- Dynamic or fixed-length padding strategy.
- Use `BCEWithLogitsLoss` for multi-label output heads.

## 3) Baseline Model Implementation

### 3.1 Shared Multi-Label Classification Design

- Output layer size = 6 (one neuron per toxicity category).
- Sigmoid activation for probability outputs.
- Loss function:
  - TensorFlow path: Binary Cross-Entropy,
  - PyTorch path: `BCEWithLogitsLoss` (preferred with logits).

### 3.2 RNN Baseline

- Architecture: Embedding -> SimpleRNN -> Dense layers -> 6 logits/probabilities.
- Include dropout and recurrent dropout where supported.

### 3.3 LSTM Baseline

- Architecture: Embedding -> LSTM -> Dense layers -> 6 outputs.
- Compare unidirectional vs bidirectional variant if compute allows.

### 3.4 GRU Baseline

- Architecture: Embedding -> GRU -> Dense layers -> 6 outputs.
- Benchmark speed/performance trade-offs relative to LSTM.

## 4) Transformer-Based Model (BERT)

### 4.1 Model Setup

- Fine-tune a pretrained BERT encoder with classification head for 6 labels.
- Start with frozen-then-unfrozen or full fine-tuning strategy depending on stability.

### 4.2 Training Configuration

- Optimizer: AdamW.
- Learning rate schedule: warmup + linear decay (optional but recommended).
- Mixed precision (if available) for efficiency.
- Gradient clipping to stabilize training.

## 5) Optimization Strategy

### 5.1 Hyperparameter Tuning Scope

- Learning rate grid/range for sequential and BERT models.
- Hidden units and number of recurrent layers.
- Embedding dimension for sequential models.
- Batch size and max sequence length.
- Dropout rates and weight decay.

### 5.2 Regularization and Generalization

- Dropout, recurrent dropout (RNN family), L2 weight decay.
- Early stopping on validation macro ROC-AUC or macro F1.
- Class imbalance handling:
  - optional positive class weighting,
  - threshold calibration per label.

### 5.3 Activation and Architectural Variants

- Evaluate dense activation options (ReLU/GELU in heads where applicable).
- Optional pooling variants:
  - last hidden state,
  - mean/max pooling for sequence encoders.

## 6) Evaluation Protocol

### 6.1 Core Metrics for Multi-Label Classification

- ROC-AUC:
  - macro-averaged across 6 labels,
  - per-label ROC-AUC.
- F1-score:
  - macro F1 and micro F1,
  - per-label F1 with calibrated thresholds.
- Additional diagnostics:
  - precision, recall per label,
  - Hamming loss (optional).

### 6.2 Validation and Comparison

- Keep a consistent validation split across all models.
- Track training and validation curves (loss, key metrics) in `plots/`.
- Compare:
  - predictive quality,
  - training time,
  - memory/computational cost.

### 6.3 Error Analysis

- Inspect high-confidence false positives and false negatives.
- Identify labels with weakest recall or precision.
- Discuss model behavior on short vs long comments.

## 7) Notebook Structure (Implementation Order)

1. Title and project metadata.
2. Imports, configuration, seeds, and environment checks.
3. Data loading and EDA.
4. Cleaning and preprocessing utilities.
5. Data split and dataloaders.
6. Baseline models: RNN, LSTM, GRU.
7. Transformer model: BERT fine-tuning.
8. Hyperparameter tuning experiments.
9. Final evaluation and model comparison.
10. Conclusions and recommended model for deployment.

## 8) Reproducibility and Engineering Practices

- Fix random seeds and log package versions.
- Save best checkpoints under `models/`.
- Persist plots under `plots/`.
- Keep notebook cells modular and executable top-to-bottom.

## 9) Expected Outcomes

- A clear experimental transition from classical sequential architectures to Transformers.
- Quantitative evidence of optimization impact (hyperparameters, regularization, thresholds).
- A final model recommendation supported by multi-label metrics and error analysis.
