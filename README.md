# ChromeCRISPR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17058362.svg)](https://doi.org/10.5281/zenodo.17058362)

This repository contains the public ChromeCRISPR release for CRISPR/Cas9 on-target activity prediction. It includes the published checkpoints, model documentation, hyperparameter records, preprocessing notes, and a Snakemake workflow for repository checks.

## Included

- published model artifacts in `models/`
- model and training documentation in `docs/`
- Python model definitions and helpers in `src/`
- workflow rules and wrappers in `workflow/` and `scripts/`
- generated reports in `reports/workflow/`

## Current checks

Latest workflow outputs report:
- `20` published artifacts tracked
- integrity checks passed
- documentation/example audit passed
- checkpoint smoke test passed for `20 / 20` artifacts

See:
- `reports/workflow/public_repo_summary.md`
- `reports/workflow/public_repo_full_summary.md`
- `reports/workflow/checkpoint_smoke.md`
- `reports/workflow/checkpoint_validator_report.json`

## Scope

This repository is for inspection, loading, and verification of the released artifacts. It does not provide a full raw-data retraining pipeline for the original study.

## Quick start

### Install

```bash
git clone https://github.com/Daneshpajouh/ChromeCRISPR.git
cd ChromeCRISPR
pip install -r requirements.txt
```

### Run checks

| Command | Purpose |
|---|---|
| `bash scripts/run_snakemake.sh report` | inventory, integrity, docs audit, preprocessing report |
| `bash scripts/run_snakemake.sh smoke` | checkpoint smoke test |
| `bash scripts/run_snakemake.sh full` | report + smoke |

### Load a checkpoint

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

## Layout

| Path | Contents |
|---|---|
| `models/` | published checkpoints |
| `docs/` | architecture notes, hyperparameters, training notes |
| `src/` | Python model code and helpers |
| `workflow/` | Snakemake rules and workflow docs |
| `reports/workflow/` | generated report files |

## Best published model

| Item | Value |
|---|---|
| Model | `CNN_GRU+GC` |
| Category | `ChromeCRISPR hybrid models` |
| Spearman | `0.876` |
| MSE | `0.0093` |
| Checkpoint | `models/chromecrispr_hybrid_models/CNN_GRU+GC.pth` |
| Hyperparameters | `docs/hyperparameters/CNN_GRU+GC_hyperparameters.json` |

## Preprocessing summary

The documented preprocessing path is:
1. validate 21-mer sgRNA sequences
2. encode sequences into the 84-feature representation
3. project them through learned embeddings in the model
4. compute GC content for GC-aware variants
5. normalize numerical features where used
6. apply the documented split and 5-fold model-selection procedure

To regenerate the preprocessing report:

```bash
bash scripts/run_snakemake.sh preprocessing
```

Main output:
- `reports/workflow/preprocessing_manifest.md`

## Python entry points

Typical imports:

```python
from src.models import create_model
from src.evaluation import ChromeCRISPRMetrics
```

Examples:
- `create_model("cnn")`
- `create_model("gru_gc")`
- `create_model("cnn_gru_gc")`

## Documentation

- [docs/README.md](docs/README.md)
- [models/README.md](models/README.md)
- [src/README.md](src/README.md)
- [docs/training_procedures/README.md](docs/training_procedures/README.md)
- [docs/hyperparameters/README.md](docs/hyperparameters/README.md)
- [workflow/README.md](workflow/README.md)

## Citation

If you use ChromeCRISPR in research, cite the associated study and the Zenodo record referenced in this repository.
