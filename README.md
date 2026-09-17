# ChromeCRISPR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17058361.svg)](https://doi.org/10.5281/zenodo.17058361)

Hybrid CNN-RNN models for predicting CRISPR/Cas9 on-target activity from sgRNA sequence.

## Quick start

    pip install -r requirements.txt

```python
from src.models.architectures import MODELS

model = MODELS["CNN_GRU+GC"]()      # the best performing model
```

Models take a one-hot encoded sequence of shape `(batch, 21, 4)`. The GC variants take GC
content as a second argument of shape `(batch,)`.

## Results

`CNN_GRU+GC` reaches a Spearman correlation of **0.8760** and a mean squared error of
**0.0093** on a held-out test set of 8,341 sgRNAs. Full table for all twenty models in
[`docs/results.md`](docs/results.md).

![CNN_GRU+GC](docs/architectures/CNN_GRU_plus_GC.svg)

## Models

Twenty models in five groups: a Random Forest baseline; CNN, GRU, LSTM and BiLSTM base models;
the same four with GC content; deep variants of each; and the three ChromeCRISPR hybrids. A
Transformer is also included. See [`docs/architectures.md`](docs/architectures.md).

## Layout

| path | contents |
|---|---|
| `src/models/architectures.py` | every model architecture |
| `src/training/`, `src/evaluation/` | training loop, hyperparameter search and metrics |
| `docs/architectures.md` | architectures and their configurable settings |
| `docs/results.md` | results for all twenty models |
| `docs/hyperparameters/` | one machine-readable record per model |
| `docs/architectures/` | one diagram per model |
| `scripts/` | training, and generators for the records and diagrams |
| `tests/` | test suite |

## Training

```python
from src.training.trainer import ChromeCRISPRTrainer
from src.models.hybrid_models import create_cnn_gru_model

trainer = ChromeCRISPRTrainer(create_cnn_gru_model, {})
results = trainer.train_model(sequences, targets,
                              {"batch_size": 64, "learning_rate": 1e-3}, epochs=100)
```

Hyperparameters are selected on a validation split drawn from the training portion. GC content
is derived from the sequence, so it does not need to be supplied.

## Tests

    pip install -r requirements-dev.txt
    pytest

## Regenerating the derived files

    python3 scripts/build_model_records.py
    python3 scripts/build_architecture_diagrams.py

## Data

From the DeepHF study, available from the NCBI Sequence Read Archive under
[PRJNA522677](https://www.ncbi.nlm.nih.gov/bioproject/522677/). See
[`DATASET_REFERENCE.md`](DATASET_REFERENCE.md).

## Citation

See [`CITATIONS.md`](CITATIONS.md).

## License

See [`LICENSE`](LICENSE).
