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
| `data/` | the dataset and its encoded arrays |
| `models/` | a checkpoint per model |
| `scripts/` | dataset build, training, and the record, diagram and results generators |
| `tests/` | test suite |

## Tests

    pip install -r requirements-dev.txt
    pytest

## Data

`data/deephf.csv` holds 55,604 sgRNAs with their wild-type SpCas9 activity. The encoded arrays
and the train/test split are rebuilt from it:

    python3 scripts/build_dataset.py

The split holds out 15% for testing and is fixed by a seed, so it is identical on every run.

## Training

    python3 scripts/train_all_models.py --data-dir data --out-dir models

Hyperparameters are searched on a validation split drawn from the training portion; the
held-out set is read once, after the final model is fitted. The run is seeded, so the same
command reproduces the same weights.

## Regenerating the derived files

    python3 scripts/build_dataset.py
    python3 scripts/build_model_records.py
    python3 scripts/build_architecture_diagrams.py
    python3 scripts/build_results_page.py

## Citation

See [`CITATIONS.md`](CITATIONS.md).

## License

See [`LICENSE`](LICENSE).
