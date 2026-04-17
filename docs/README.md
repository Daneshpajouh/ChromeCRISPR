# ChromeCRISPR Documentation

This directory is the documentation index for the public ChromeCRISPR release.

## Scope

The documentation set covers:
- per-model hyperparameter records
- architecture descriptions
- training-procedure notes from the study
- manuscript-facing performance summaries

It should be read together with the canonical checkpoint collection in `models/` and the public verification workflow in `workflow/`.

## Primary Entry Points

- [hyperparameters/](hyperparameters/): per-model JSON records and README summaries
- [training_procedures/README.md](training_procedures/README.md): training and preprocessing notes
- [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md): architecture-level overview
- [COMPREHENSIVE_MODEL_DOCUMENTATION.md](COMPREHENSIVE_MODEL_DOCUMENTATION.md): comparative study write-up

## Best Model Quick Reference

- Model: `CNN_GRU+GC`
- Checkpoint: `../models/chromecrispr_hybrid_models/CNN_GRU+GC.pth`
- Hyperparameters: `hyperparameters/CNN_GRU+GC_hyperparameters.json`
- Published test performance: `0.876` Spearman, `0.0093` MSE

## Public Verification Workflow

To rebuild the repo-level public reports:

```bash
bash scripts/run_snakemake.sh report
```

Generated outputs are written to `../reports/workflow/` and include:
- model registry
- integrity report
- markdown example/link audit
- public summary report

## Notes on Usage Examples

The public repo supports lightweight repo-local Python usage such as constructing model classes and loading released checkpoints. It does not currently expose a complete raw-data retraining API with a stable packaged interface.
