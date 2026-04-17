# ChromeCRISPR Checkpoints

This directory contains the published model artifacts.

## Categories

| Category | Count |
|---|---:|
| Base models | 5 |
| Base models + GC | 4 |
| Deep models | 4 |
| Deep models + GC | 4 |
| ChromeCRISPR hybrid models | 3 |

## Best model

| Item | Value |
|---|---|
| Model | `CNN_GRU+GC` |
| Checkpoint | `chromecrispr_hybrid_models/CNN_GRU+GC.pth` |
| Spearman | `0.876` |
| MSE | `0.0093` |
| Hyperparameters | `../docs/hyperparameters/CNN_GRU+GC_hyperparameters.json` |

## Directory list

- `base_models/`: RF, CNN, GRU, LSTM, BiLSTM
- `base_models_with_gc/`: GC variants of the neural base models
- `deep_models/`: deep CNN/GRU/LSTM/BiLSTM
- `deep_models_with_gc/`: deep GC variants
- `chromecrispr_hybrid_models/`: hybrid CNN-RNN models

## Loading example

```python
import torch
from src.models import create_model

model = create_model("cnn_gru_gc")
state_dict = torch.load(
    "models/chromecrispr_hybrid_models/CNN_GRU+GC.pth",
    map_location="cpu",
)
model.load_state_dict(state_dict, strict=False)
model.eval()
```

## Checks

The workflow treats `models/` as the canonical artifact root. To rebuild the registry and integrity reports:

```bash
bash scripts/run_snakemake.sh inventory
bash scripts/run_snakemake.sh integrity
```
