# Source

| module | contents |
|---|---|
| `models/architectures.py` | every model architecture |
| `models/cnn_model.py`, `models/rnn_models.py`, `models/hybrid_models.py` | named factories |
| `training/trainer.py` | training loop and hyperparameter search |
| `evaluation/metrics.py` | Spearman correlation, mean squared error and comparison utilities |

`tests/` checks each architecture's layer counts, widths, kernel sizes and fusion widths.
