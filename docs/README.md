# ChromeCRISPR Documentation

This directory is the documentation index for the public ChromeCRISPR release.

## Documentation Roles

- `hyperparameters/`: per-model JSON records and human-readable summaries
- `training_procedures/`: documented preprocessing and training assumptions from the study
- `MODEL_ARCHITECTURES.md`: architecture-level overview
- `COMPREHENSIVE_MODEL_DOCUMENTATION.md`: manuscript-facing comparative discussion

## Canonical Repo Boundary

- Canonical checkpoints live in `../models/`
- Repo-local code lives in `../src/`
- Workflow-generated verification and preprocessing reports live in `../reports/workflow/`

## Best Model Quick Reference

- Model: `CNN_GRU+GC`
- Checkpoint: `../models/chromecrispr_hybrid_models/CNN_GRU+GC.pth`
- Hyperparameters: `hyperparameters/CNN_GRU+GC_hyperparameters.json`
- Published test performance: `0.876` Spearman, `0.0093` MSE

## Preprocessing and Workflow Reports

Rebuild the public docs-side reports with:

```bash
bash scripts/run_snakemake.sh report
```

Key generated artifacts:
- `../reports/workflow/model_registry.md`
- `../reports/workflow/repo_integrity.md`
- `../reports/workflow/public_examples_audit.md`
- `../reports/workflow/preprocessing_manifest.md`
- `../reports/workflow/public_repo_summary.md`

## Notes on Public Usage

The public repo supports inspection, verification, and repo-local checkpoint loading. It does not currently claim to expose a fully packaged raw-data retraining API.
