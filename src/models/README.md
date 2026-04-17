# ChromeCRISPR Model Code

This directory contains the model definitions used by the public Python interface.

## Boundary

- `../../models/` contains the published checkpoint artifacts
- `src/models/` contains the corresponding Python model code

## Files

| File | Contents |
|---|---|
| `cnn_model.py` | CNN and deep CNN variants |
| `rnn_models.py` | GRU, LSTM, and BiLSTM variants |
| `hybrid_models.py` | hybrid CNN-GRU/LSTM/BiLSTM variants |
| `__init__.py` | exported factory functions |

## Factory examples

```python
from src.models import create_model

cnn = create_model("cnn")
gru_gc = create_model("gru_gc")
hybrid = create_model("cnn_gru_gc")
```

## Loading a published artifact

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
