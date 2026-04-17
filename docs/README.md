# ChromeCRISPR Documentation

This directory contains the documentation bundled with the public release.

## Contents

| Path | Description |
|---|---|
| `hyperparameters/` | per-model JSON records and companion READMEs |
| `training_procedures/` | preprocessing and training notes |
| `MODEL_ARCHITECTURES.md` | architecture overview |
| `COMPREHENSIVE_MODEL_DOCUMENTATION.md` | model comparison notes |
| `model_architectures/` | LaTeX/TikZ architecture assets |

## Related locations

- checkpoints: `../models/`
- Python code: `../src/`
- generated reports: `../reports/workflow/`

## Suggested reading order

1. top-level `README.md`
2. `training_procedures/README.md`
3. `hyperparameters/README.md`
4. per-model files under `hyperparameters/`

## Best model reference

| Item | Value |
|---|---|
| Model | `CNN_GRU+GC` |
| Checkpoint | `../models/chromecrispr_hybrid_models/CNN_GRU+GC.pth` |
| Hyperparameters | `hyperparameters/CNN_GRU+GC_hyperparameters.json` |
| Spearman | `0.876` |
| MSE | `0.0093` |

## Generated reports

Regenerate the documentation-side report set with:

```bash
bash scripts/run_snakemake.sh report
```

Key outputs:
- `../reports/workflow/model_registry.md`
- `../reports/workflow/repo_integrity.md`
- `../reports/workflow/public_examples_audit.md`
- `../reports/workflow/preprocessing_manifest.md`
- `../reports/workflow/public_repo_summary.md`
- `../reports/workflow/public_repo_full_summary.md`

## Scope note

The documentation describes the released artifacts and the documented study setup. It is not a substitute for a full raw-data retraining package.
