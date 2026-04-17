# ChromeCRISPR Model Source Code

This directory now contains the repo-local model code only.

## Canonical Boundary

- `models/` is the canonical root for the published checkpoint artifacts.
- `src/models/` is the canonical root for the Python model definitions and factory exports.

This separation is intentional and enforced by the workflow integrity checks so the public repo does not ship duplicated checkpoint trees under both locations.

## Available Source Files

- `cnn_model.py`: CNN and deep CNN model definitions
- `rnn_models.py`: GRU, LSTM, and BiLSTM model definitions
- `hybrid_models.py`: ChromeCRISPR hybrid CNN-RNN definitions
- `__init__.py`: public factory exports for repo-local examples

## Public Factory Surface

```python
from src.models import create_model, create_cnn_gru_model

best_model = create_cnn_gru_model()
base_model = create_model("gru")
```

## Checkpoints

Published checkpoints should be loaded from `models/`, for example:

```python
import torch
from src.models import create_cnn_gru_model

model = create_cnn_gru_model()
state_dict = torch.load("models/chromecrispr_hybrid_models/CNN_GRU+GC.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
```
