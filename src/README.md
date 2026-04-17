# ChromeCRISPR Source Tree

This directory contains the repo-local Python code used to describe, instantiate, and inspect the ChromeCRISPR models.

## Layout

```text
src/
├── evaluation/   # Metrics and compatibility-oriented validation helpers
├── models/       # Model class definitions and public factory exports
└── training/     # Training utilities retained from the study codebase
```

## Canonical Boundary

- `models/` at the repository root is the canonical checkpoint/artifact location.
- `src/models/` is code only.
- The workflow integrity checks treat checkpoint files under `src/models/` as a structural error.

## Public Surface

The most stable repo-local entry point is `src.models`.

```python
from src.models import create_cnn_gru_model, create_model

model = create_cnn_gru_model()
other = create_model("lstm")
```

Available exported factories:
- `create_cnn_model`
- `create_deep_cnn_model`
- `create_gru_model`
- `create_lstm_model`
- `create_bilstm_model`
- `create_cnn_gru_model`
- `create_cnn_lstm_model`
- `create_cnn_bilstm_model`
- `create_model`

## Evaluation Utilities

`src.evaluation` currently exposes metric helpers and a compatibility-oriented validation script. The validation script is a smoke-test style utility around the published checkpoints; it is not a full reproduction of the manuscript evaluation pipeline.

## Training Utilities

`src.training` retains the study’s training helper scaffolding. These utilities assume repo-local/preprocessed inputs and are better treated as research code than as a polished public SDK.
