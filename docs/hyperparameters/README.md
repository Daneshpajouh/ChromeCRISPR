# ChromeCRISPR Hyperparameters

This directory contains per-model hyperparameter records for the public ChromeCRISPR release.

## Files

Each model is documented in two forms:
- `*_hyperparameters.json`: machine-readable record
- `*_README.md`: model-specific summary and notes

## Categories

| Category | Models |
|---|---:|
| Base models | 5 |
| Base models + GC | 4 |
| Deep models | 4 |
| Deep models + GC | 4 |
| Hybrid models | 3 |

## Best model

| Item | Value |
|---|---|
| Model | `CNN_GRU+GC` |
| Spearman | `0.876` |
| MSE | `0.0093` |
| JSON | `CNN_GRU+GC_hyperparameters.json` |
| Summary | `CNN_GRU+GC_README.md` |

## Model index

### Base models
- `CNN`
- `GRU`
- `LSTM`
- `BiLSTM`
- `RF`

### Base models + GC
- `CNN+GC`
- `GRU+GC`
- `LSTM+GC`
- `BiLSTM+GC`

### Deep models
- `deepCNN`
- `deepGRU`
- `deepLSTM`
- `deepBiLSTM`

### Deep models + GC
- `deepCNN+GC`
- `deepGRU+GC`
- `deepLSTM+GC`
- `deepBiLSTM+GC`

### Hybrid models
- `CNN_GRU+GC`
- `CNN_LSTM+GC`
- `CNN_BiLSTM+GC`

## Table of reported performance

| Model | Spearman | MSE |
|---|---:|---:|
| `CNN_GRU+GC` | `0.876` | `0.0093` |
| `deepCNN+GC` | `0.873` | `0.0093` |
| `CNN_BiLSTM+GC` | `0.870` | `0.0096` |
| `deepCNN` | `0.869` | `0.0098` |
| `deepGRU` | `0.868` | `0.0099` |
| `CNN_LSTM+GC` | `0.867` | `0.0115` |
| `deepBiLSTM+GC` | `0.867` | `0.0098` |
| `deepGRU+GC` | `0.867` | `0.0098` |
| `deepLSTM` | `0.862` | `0.0103` |
| `deepBiLSTM` | `0.862` | `0.0104` |
| `deepLSTM+GC` | `0.860` | `0.0104` |
| `LSTM+GC` | `0.856` | `0.0112` |
| `BiLSTM+GC` | `0.855` | `0.0110` |
| `BiLSTM` | `0.843` | `0.0120` |
| `GRU+GC` | `0.840` | `0.0122` |
| `GRU` | `0.837` | `0.0121` |
| `LSTM` | `0.837` | `0.0122` |
| `CNN` | `0.793` | `0.0161` |
| `RF` | `0.789` | `0.0161` |
| `CNN+GC` | `0.781` | `0.0170` |

## Common training setup

Across the documented runs:
- optimization framework: Optuna
- selection procedure: 5-fold cross-validation
- optimizer: Adam for neural models
- primary ranking metric: Spearman correlation
- batch sizes tested: `32`, `64`, `128`
- learning-rate search range: `1e-5` to `1e-2`

## Notes

- The JSON files are the best source for exact parameter values.
- The companion READMEs are easier to read but are still summaries of the released study artifacts.
- For the current repository check status, use the files under `../../reports/workflow/` rather than this directory.
