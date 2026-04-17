# ChromeCRISPR Canonical Checkpoints

This directory is the canonical location of the published ChromeCRISPR model artifacts.

## Inventory

| Category | Count |
|---|---:|
| Base models | 5 |
| Base models + GC | 4 |
| Deep models | 4 |
| Deep models + GC | 4 |
| ChromeCRISPR hybrid models | 3 |

## Best Model

- Model: `CNN_GRU+GC`
- Checkpoint: `chromecrispr_hybrid_models/CNN_GRU+GC.pth`
- Published performance: `0.876` Spearman, `0.0093` MSE
- Hyperparameters: `../docs/hyperparameters/CNN_GRU+GC_hyperparameters.json`

## Directory Guide

- `base_models/`: RF, CNN, GRU, LSTM, BiLSTM
- `base_models_with_gc/`: GC-content variants of the base neural models
- `deep_models/`: deeper CNN/GRU/LSTM/BiLSTM checkpoints
- `deep_models_with_gc/`: deep variants with GC-content integration
- `chromecrispr_hybrid_models/`: the ChromeCRISPR hybrid architectures

## Repo-Local Loading Example

```python
import torch
from src.models import create_cnn_gru_model

model = create_cnn_gru_model()
state_dict = torch.load("models/chromecrispr_hybrid_models/CNN_GRU+GC.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()
```

## Workflow Integration

The Snakemake workflow treats this directory as the canonical artifact root when building the public model registry and integrity reports:

```bash
bash scripts/run_snakemake.sh inventory
```
