# Results

Measured on a held-out test set of 8,341 sgRNAs, with means and medians across the
cross-validation folds. Standard deviations are given for the hybrid models.

| group | model | Spearman (mean) | Spearman (median) | MSE (mean) | MSE (median) | retrained Spearman | retrained MSE |
|---|---|---|---|---|---|---|---|
| Baseline | RF | 0.7550 | 0.7554 | 0.0197 | 0.0195 | 0.7485 | 0.0206 |
| Base models | CNN | 0.7925 | 0.7902 | 0.0161 | 0.0156 | | |
|  | GRU | 0.8368 | 0.8396 | 0.0121 | 0.0122 | | |
|  | LSTM | 0.8371 | 0.8365 | 0.0122 | 0.0122 | | |
|  | BiLSTM | 0.8432 | 0.8466 | 0.0120 | 0.0123 | | |
| Base models with GC content | CNN+GC | 0.7810 | 0.7807 | 0.0170 | 0.0172 | | |
|  | GRU+GC | 0.8401 | 0.8421 | 0.0122 | 0.0123 | | |
|  | LSTM+GC | 0.8564 | 0.8598 | 0.0112 | 0.0113 | | |
|  | BiLSTM+GC | 0.8550 | 0.8582 | 0.0110 | 0.0111 | | |
| Deep models | deepCNN | 0.8694 | 0.8740 | 0.0098 | 0.0098 | | |
|  | deepGRU | 0.8684 | 0.8697 | 0.0099 | 0.0097 | | |
|  | deepLSTM | 0.8620 | 0.8641 | 0.0103 | 0.0103 | | |
|  | deepBiLSTM | 0.8617 | 0.8623 | 0.0104 | 0.0103 | | |
| Deep models with GC content | deepCNN+GC | 0.8728 | 0.8756 | 0.0093 | 0.0092 | | |
|  | deepGRU+GC | 0.8668 | 0.8689 | 0.0098 | 0.0099 | | |
|  | deepLSTM+GC | 0.8602 | 0.8623 | 0.0104 | 0.0106 | | |
|  | deepBiLSTM+GC | 0.8671 | 0.8690 | 0.0098 | 0.0100 | | |
| ChromeCRISPR hybrids | CNN_LSTM+GC | 0.8668 ± 0.009 | 0.8659 | 0.0115 ± 0.0009 | 0.0117 | | |
|  | CNN_BiLSTM+GC | 0.8700 ± 0.009 | 0.8708 | 0.0096 ± 0.0007 | 0.0097 | | |
|  | CNN_GRU+GC | 0.8760 ± 0.008 | 0.8796 | 0.0093 ± 0.0006 | 0.0093 | | |

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
