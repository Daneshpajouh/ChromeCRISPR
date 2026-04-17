# ChromeCRISPR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17058362.svg)](https://doi.org/10.5281/zenodo.17058362)

ChromeCRISPR is a public artifact repository for the ChromeCRISPR study on CRISPR/Cas9 on-target activity prediction. It packages the published model checkpoints, hyperparameter records, architecture documentation, and a Snakemake workflow that verifies the public artifact set.

## What This Repository Is

This repository is designed to support:
- access to the published ChromeCRISPR model checkpoints
- inspection of per-model hyperparameters and architecture notes
- lightweight repo-level reproducibility checks through Snakemake
- repo-local Python usage examples for model construction and checkpoint loading

## What This Repository Is Not

This repository does not currently provide a fully turnkey raw-data retraining pipeline for the original manuscript experiments. The public workflow focuses on published artifact integrity and documentation consistency around the released model set.

## Model Collection

The canonical published checkpoints live under `models/`.

| Category | Count | Notes |
|---|---:|---|
| Base models | 5 | RF, CNN, GRU, LSTM, BiLSTM |
| Base models + GC | 4 | Adds GC-content feature variants |
| Deep models | 4 | Deeper single-architecture models |
| Deep models + GC | 4 | Deep variants with GC-content |
| ChromeCRISPR hybrid models | 3 | CNN-RNN hybrids including the top model |

Best published model:
- `CNN_GRU+GC`
- Spearman correlation: `0.876`
- Mean squared error: `0.0093`
- Checkpoint: `models/chromecrispr_hybrid_models/CNN_GRU+GC.pth`
- Hyperparameters: `docs/hyperparameters/CNN_GRU+GC_hyperparameters.json`

## Snakemake Workflow

This repo now includes a real Snakemake workflow for public verification and organization.

```bash
bash scripts/run_snakemake.sh report
```

Useful targets:
- `inventory`: build a canonical model registry from the released checkpoints and hyperparameter JSON files
- `integrity`: verify the public artifact set and workflow/package surface
- `audit`: check public markdown links and import examples
- `report`: build the full public summary under `reports/workflow/`

Workflow docs: [workflow/README.md](workflow/README.md)

## Installation

```bash
git clone https://github.com/Daneshpajouh/ChromeCRISPR.git
cd ChromeCRISPR
pip install -r requirements.txt
```

For the Snakemake workflow, use the bootstrap wrapper instead of installing Snakemake into the main environment manually:

```bash
bash scripts/run_snakemake.sh report
```

## Python Usage

The public Python surface is intentionally small and repo-local.

### Create a model instance

```python
from src.models import create_cnn_gru_model, create_model

best_model = create_cnn_gru_model()
base_gru = create_model("gru")
```

### Load a published checkpoint

```python
import torch
from src.models import create_cnn_gru_model

model = create_cnn_gru_model()
state_dict = torch.load(
    "models/chromecrispr_hybrid_models/CNN_GRU+GC.pth",
    map_location="cpu",
)
model.load_state_dict(state_dict, strict=False)
model.eval()
```

### Repo-local evaluation utilities

```python
from src.evaluation import ChromeCRISPRMetrics

metrics = ChromeCRISPRMetrics()
```

## Repository Layout

```text
ChromeCRISPR/
├── docs/                    # Hyperparameters, architectures, and study documentation
├── models/                  # Canonical published checkpoints
├── reports/workflow/        # Generated workflow outputs
├── scripts/                 # Public registry, integrity, and workflow helpers
├── src/                     # Repo-local model, evaluation, and training code
└── workflow/                # Snakemake workflow definition
```

## Documentation

- [docs/README.md](docs/README.md): documentation index
- [models/README.md](models/README.md): canonical checkpoint inventory
- [docs/hyperparameters/README.md](docs/hyperparameters/README.md): per-model hyperparameter records
- [docs/MODEL_ARCHITECTURES.md](docs/MODEL_ARCHITECTURES.md): architecture overview
- [docs/COMPREHENSIVE_MODEL_DOCUMENTATION.md](docs/COMPREHENSIVE_MODEL_DOCUMENTATION.md): manuscript-facing performance discussion

## Citation

If you use ChromeCRISPR in research, please cite the associated study and Zenodo record referenced in this repository.
