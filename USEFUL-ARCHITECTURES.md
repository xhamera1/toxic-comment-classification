# Useful Architectures and Optimization Patterns for Task 3

## Purpose of This Document

This file extracts practical ideas from the lecture notebooks in `sequential-notebooks-lecture/` and translates them into an implementation strategy for the Jigsaw Toxic Comment **multi-label** task.

It is intentionally not a 1:1 copy of lecture code. Instead, it captures reusable design patterns, expected hyperparameter ranges, and required adaptations.

---

## Source Notebooks Reviewed

1. `AH-ML-7-Sequential_Models v4.ipynb`
2. `AH-ML-7-Sequential_Models_ECG v4.ipynb`
3. `AH-Lecture-LAB12-BERTonTextIMDB v2.ipynb`
4. `AH-ML-8-BERTonTextIMDB v6.ipynb`

---

## 1) Sequential Models (RNN / LSTM / GRU): What Is Reusable

## Core Architectural Pattern

Lecture pattern:

- `Tokenizer` -> integer sequences -> `pad_sequences`
- `Embedding(input_dim=vocab_size, output_dim=8 or 16)`
- recurrent layer (`SimpleRNN`, `GRU`, or `LSTM`)
- final `Dense(..., activation='sigmoid')` for binary tasks

How to adapt for Jigsaw multi-label:

- Keep the same backbone idea (`Embedding + recurrent encoder`).
- Replace final head with `Dense(6, activation='sigmoid')` for six independent labels.
- Use `binary_crossentropy` (TensorFlow) or `BCEWithLogitsLoss` (PyTorch) for multi-label objective.

## Strong Baseline Templates

- **RNN baseline**: `Embedding -> SimpleRNN(64-128) -> Dropout -> Dense(6, sigmoid)`
- **GRU baseline**: `Embedding -> GRU(64-128) -> Dropout -> Dense(6, sigmoid)`
- **LSTM baseline**: `Embedding -> LSTM(64-128) -> Dropout -> Dense(6, sigmoid)`
- **Improved variants**:
  - Bidirectional recurrent layers (`Bidirectional(LSTM(...))`)
  - Stacked recurrent blocks with `return_sequences=True` in lower layers

The ECG notebook confirms that GRU/LSTM/BiLSTM variants are natural progression paths when sequence complexity grows.

## Preprocessing Signals to Reuse

From sequential text notebook:

- Use Keras tokenization and sequence padding as first-stage pipeline.
- Compute and control `maxlen` rather than using unconstrained sequence lengths.
- Keep vocabulary capped (`num_words`) to control memory and overfitting.

Recommended adaptation:

- Set `maxlen` from distribution quantiles (e.g., 95th percentile of tokenized comment length) instead of pure max.
- Use an OOV token and truncation policy consistent across all RNN/LSTM/GRU experiments.

## Optimization Patterns Observed

Lecture notebooks consistently use:

- `optimizer='adam'`
- multi-epoch training
- simple architecture sweeps (RNN vs GRU vs LSTM)

Recommended extension for Task 3:

- Tune:
  - embedding dim: `[64, 128, 256]`
  - hidden units: `[64, 128, 256]`
  - dropout/recurrent dropout: `[0.1, 0.2, 0.3, 0.5]`
  - learning rate: `[1e-3, 5e-4, 1e-4]`
  - batch size: `[32, 64, 128]`
  - `maxlen`: `[128, 192, 256, 320]`
- Add callbacks:
  - `EarlyStopping(monitor='val_roc_auc_macro', patience=2-3, mode='max', restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_roc_auc_macro', factor=0.5, patience=1-2, mode='max')`

---

## 2) BERT/Transformer Pattern: What Is Reusable

## Core Lecture Pattern

From both BERT notebooks:

- `BertTokenizer.from_pretrained("bert-base-uncased")`
- tokenization with `padding=True, truncation=True` and optional `max_length=512`
- `BertForSequenceClassification.from_pretrained(...)`
- Hugging Face `Trainer` + `TrainingArguments`
- metrics through `compute_metrics`

## Required Multi-Label Adaptation (Critical)

Lecture notebooks focus on binary sentiment setups. For Jigsaw:

- Set `num_labels=6`
- Configure **multi-label** output behavior:
  - logits shape `[batch_size, 6]`
  - sigmoid on logits for probabilities
- Use multi-label loss:
  - if using native HF model, ensure problem type is set for multi-label classification
  - alternatively use custom Trainer loss (`BCEWithLogitsLoss`)
- Convert labels to float tensors with shape `[N, 6]`
- Use thresholding per label (default 0.5 baseline, then calibrate)

## Recommended BERT TrainingArguments Starting Point

- `learning_rate=2e-5`
- `weight_decay=0.01`
- `per_device_train_batch_size=8 or 16` (based on VRAM)
- `per_device_eval_batch_size=8 or 16`
- `num_train_epochs=3 to 5` for first sweep
- `eval_strategy='epoch'` and `save_strategy='epoch'`
- `load_best_model_at_end=True`
- consider `fp16=True` if GPU supports it

These defaults are consistent with lecture settings and align with common fine-tuning practice.

## Hyperparameter Tuning Patterns from Lectures

The BERT notebooks explicitly explore:

- learning rate sweeps (`2e-5`, `5e-5`)
- batch size sweeps (`8`, `16`, `32`)
- epoch changes
- Optuna-based search over `learning_rate`, `batch_size`, and `num_epochs`

Recommended Jigsaw extension:

- add search over:
  - threshold strategy (global 0.5 vs per-label tuned thresholds)
  - `max_length` (`128`, `192`, `256`)
  - warmup ratio (`0.0`, `0.06`, `0.1`)

---

## 3) Metrics and Evaluation: What Must Change for Task 3

Lecture metrics are mainly binary classification oriented (accuracy, precision, recall, F1).  
For multi-label toxicity, use:

- **Primary**:
  - ROC-AUC macro (across 6 labels)
  - F1 macro and F1 micro
- **Secondary**:
  - per-label ROC-AUC
  - per-label precision/recall/F1
  - Hamming loss (optional)
- **Calibration**:
  - tune threshold per label on validation set to maximize F1 or Youden J statistic

Avoid relying on plain accuracy as a main KPI due to severe class imbalance and sparse positive labels.

---

## 4) Practical Architecture Stack for `GGSN_Patryk_Chamera_Zad3.ipynb`

Use this progression:

1. **Baseline 1**: `Embedding + SimpleRNN`
2. **Baseline 2**: `Embedding + GRU`
3. **Baseline 3**: `Embedding + LSTM`
4. **Enhanced sequential**: `Embedding + BiLSTM` (optional if time allows)
5. **Transformer**: `bert-base-uncased` multi-label fine-tuning

This gives a clear and academic transition from classic sequential models to Transformer methods.

---

## 5) Common Pitfalls to Avoid (Based on Lecture-to-Task Gap)

- Do not keep binary output heads (`Dense(1, sigmoid)`) for Jigsaw.
- Do not use softmax over 6 labels (labels are not mutually exclusive).
- Do not report only accuracy.
- Do not compare models trained with different train/validation splits.
- Do not use max sequence length blindly at 512 for all experiments; it can waste compute.

---

## 6) Suggested Minimal Experimental Grid (Submission-Friendly)

## Sequential Models

- Models: `RNN`, `GRU`, `LSTM`
- Common fixed setup:
  - `maxlen=256`
  - `vocab_size=30_000`
  - `embedding_dim=128`
- Tune:
  - hidden units: `64 vs 128`
  - dropout: `0.2 vs 0.5`
  - learning rate: `1e-3 vs 5e-4`

## BERT

- Base model: `bert-base-uncased`, `num_labels=6`
- Tune:
  - LR: `2e-5 vs 3e-5`
  - batch size: `8 vs 16`
  - epochs: `3 vs 4`
  - max length: `192 vs 256`

This grid is small enough for coursework constraints but strong enough to show optimization methodology.

---

## 7) Final Recommendation for Your Notebook Narrative

When writing the final notebook, frame the story as:

1. Start with reproducible preprocessing for multi-label text.
2. Build interpretable sequential baselines.
3. Apply systematic optimization (not ad-hoc trial and error).
4. Transition to BERT and show why contextual embeddings improve performance.
5. Use multi-label metrics and threshold calibration to justify final model choice.

This narrative directly satisfies Task 3 goals and demonstrates engineering + research maturity.
