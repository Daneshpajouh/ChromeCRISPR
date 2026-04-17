# ChromeCRISPR Preprocessing Manifest

Generated at: `2026-04-17T01:37:30Z`

## Dataset Scope

- Dataset: `DeepHF public benchmark bundle`
- Sequence length: `21`
- Prediction target: `sgRNA on-target activity / indel frequency`
- Boundary note: The public repo documents the study setup but does not bundle the full raw training dataset.

## Preprocessing Pipeline

| Step | Name | Output |
|---:|---|---|
| 1 | sequence_validation | validated 21-mer sequence |
| 2 | one_hot_encoding | 84-dimensional sequence representation |
| 3 | embedding_layer | 128-dimensional latent sequence embedding |
| 4 | gc_content_calculation | scalar GC content in [0, 1] |
| 5 | feature_normalization | normalized features |
| 6 | split_and_validation | train, validation, and test partitions |

## Training Protocol

- Optimizer: `Adam`
- Loss: `Mean Squared Error`
- HPO: `Optuna Bayesian optimization` with `100` trials/model
- Cross-validation: `5-fold`
- Early stopping patience: `10`
- LR schedule: `ReduceLROnPlateau`
- Reported hardware: `NVIDIA V100 Volta, 32GB HBM2`

## Best Model Snapshot

- Model: `CNN_GRU+GC`
- Spearman: `0.876`
- MSE: `0.0093`
- Learning rate: `0.000209`
- Batch size: `64`
- Dropout: `0.142`
- Epochs: `84`

## Public Repo Boundary

- Canonical checkpoint root: `models/`
- Workflow focus: artifact verification and documentation clarity
- Not bundled: `raw training data, full production retraining pipeline`
