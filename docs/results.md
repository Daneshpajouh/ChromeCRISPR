# Results

Measured on a held-out test set of 8,341 sgRNAs. Each model's predictions are partitioned
into ten shuffled folds and scored per fold; the table reports the mean and median of those
per-fold scores. Standard deviations are given for the hybrid models.

Folds come from `KFold(n_splits=10, shuffle=True, random_state=42)` over the prediction
vector, so the partition is identical for every model and the figures are exactly
reproducible. Scoring a model's predictions over the whole test set at once gives a slightly
different number, typically about 0.0005 higher on Spearman, because the folds are smaller.

| group | model | Spearman (mean) | Spearman (median) | MSE (mean) | MSE (median) | retrained Spearman | retrained MSE |
|---|---|---|---|---|---|---|---|
| Baseline | RF | 0.7550 | 0.7554 | 0.0197 | 0.0195 | 0.7534 | 0.0201 |
| Base models | CNN | 0.7925 | 0.7902 | 0.0161 | 0.0156 | 0.8546 | 0.0114 |
|  | GRU | 0.8368 | 0.8396 | 0.0121 | 0.0122 | 0.8613 | 0.0100 |
|  | LSTM | 0.8371 | 0.8365 | 0.0122 | 0.0122 | 0.8638 | 0.0101 |
|  | BiLSTM | 0.8432 | 0.8466 | 0.0120 | 0.0123 | 0.8638 | 0.0099 |
| Base models with GC content | CNN+GC | 0.7810 | 0.7807 | 0.0170 | 0.0172 | 0.8546 | 0.0115 |
|  | GRU+GC | 0.8401 | 0.8421 | 0.0122 | 0.0123 | 0.8647 | 0.0098 |
|  | LSTM+GC | 0.8564 | 0.8598 | 0.0112 | 0.0113 | 0.8626 | 0.0102 |
|  | BiLSTM+GC | 0.8550 | 0.8582 | 0.0110 | 0.0111 | 0.8643 | 0.0099 |
| Deep models | deepCNN | 0.8694 | 0.8740 | 0.0098 | 0.0098 | 0.8636 | 0.0101 |
|  | deepGRU | 0.8684 | 0.8697 | 0.0099 | 0.0097 | 0.8635 | 0.0099 |
|  | deepLSTM | 0.8620 | 0.8641 | 0.0103 | 0.0103 | 0.8603 | 0.0101 |
|  | deepBiLSTM | 0.8617 | 0.8623 | 0.0104 | 0.0103 | 0.8628 | 0.0102 |
| Deep models with GC content | deepCNN+GC | 0.8728 | 0.8756 | 0.0093 | 0.0092 | 0.8641 | 0.0100 |
|  | deepGRU+GC | 0.8668 | 0.8689 | 0.0098 | 0.0099 | 0.8633 | 0.0100 |
|  | deepLSTM+GC | 0.8602 | 0.8623 | 0.0104 | 0.0106 | 0.8604 | 0.0103 |
|  | deepBiLSTM+GC | 0.8671 | 0.8690 | 0.0098 | 0.0100 | 0.8632 | 0.0101 |
| ChromeCRISPR hybrids | CNN_LSTM+GC | 0.8668 ± 0.009 | 0.8659 | 0.0115 ± 0.0009 | 0.0117 | 0.8661 | 0.0099 |
|  | CNN_BiLSTM+GC | 0.8700 ± 0.009 | 0.8708 | 0.0096 ± 0.0007 | 0.0097 | 0.8684 | 0.0095 |
|  | CNN_GRU+GC | 0.8760 ± 0.008 | 0.8796 | 0.0093 ± 0.0006 | 0.0093 | 0.8710 | 0.0096 |

The best performing model is `CNN_GRU+GC`, at a Spearman correlation of 0.8760
and a mean squared error of 0.0093.

## Retrained checkpoints

`models/` holds a checkpoint per model, retrained with the pipeline in this
repository: hyperparameters searched on a validation split drawn from the
training portion, then the held-out set read once. Their scores are the last two
columns above. Retraining does not reproduce a published figure exactly, because
the weights come from a fresh optimisation; the run is seeded, so these
particular checkpoints regenerate exactly.

    python3 scripts/train_all_models.py --data-dir data --out-dir models

Per-model records, including each architecture and its configurable settings, are in
`docs/hyperparameters/`, written by `scripts/build_model_records.py`.

## Reproducing these figures

Every value in the first four columns can be recomputed from files in this repository:

    python3 scripts/verify_published_results.py

The script scores each prediction vector in `artifacts/predictions/` fold by fold, compares
all four reported columns with the table above, then loads the ChromeCRISPR checkpoint from
`artifacts/models/` and requires it to reproduce its own prediction vector. All 76 figures
agree to four decimal places. It exits non-zero if anything disagrees.

The `retrained` columns are a separate exercise: models trained here in the same architecture
as the published checkpoint, with hyperparameters and the epoch count chosen on a validation
split, a refit on the training and validation rows, and the test set read once. Those weights
are in `models/published_family/`, their predictions in `artifacts/retrained_predictions/`,
and `models/published_family/results.json` gives each one's settings, selected epoch and
hashes. They are scored fold by fold like the columns to their left, so the two are
comparable, and they are reported as measured rather than as a restatement.

Nine of the nineteen meet or exceed their published value and two more fall within the
table's rounding width; see `models/published_family/README.md` for what accounts for the
rest. `models/retrained/` holds the same models built from `src/models/architectures.py`, which
implements the Methods, trained under an identical protocol. Six of those reach their published
value against ten here, and the mean Spearman is 0.8502 against 0.8630. Since the two sets
differ in nothing but architecture, that difference measures what the architecture choice is
worth on this dataset.

