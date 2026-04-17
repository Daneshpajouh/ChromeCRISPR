# ChromeCRISPR Source Tree

This directory contains the Python code included with the public release.

## Layout

```text
src/
├── evaluation/   # metrics and checkpoint validation helpers
├── models/       # model classes and factory functions
└── training/     # retained training utilities
```

## Boundary

- published artifacts live in `../models/`
- code lives in `src/`
- checkpoint files under `src/models/` are treated as an integrity error by the workflow

## Main entry points

Typical imports are in `src.models` and `src.evaluation`.

```python
from src.models import create_model
from src.evaluation import ChromeCRISPRMetrics
```

Available factory names include:
- `cnn`
- `cnn_gc`
- `deep_cnn`
- `deep_cnn_gc`
- `gru`
- `gru_gc`
- `lstm`
- `lstm_gc`
- `bilstm`
- `bilstm_gc`
- `cnn_gru_gc`
- `cnn_lstm_gc`
- `cnn_bilstm_gc`

## Notes

- `src.evaluation` contains metrics and the checkpoint smoke validator.
- `src.training` contains retained study code and is closer to research code than a packaged training API.
