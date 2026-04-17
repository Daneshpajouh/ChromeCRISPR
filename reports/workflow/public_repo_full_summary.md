# ChromeCRISPR Public Repo Full Summary

Generated from registry timestamp: `2026-04-17T01:37:30Z`

## Status

- Canonical models tracked: `20`
- Repo integrity passed: `true`
- Public examples audit passed: `true`
- Duplicate checkpoints under `src/models/`: `0`
- Orphan checkpoints: `0`
- Orphan hyperparameter files: `0`

## Top Models

| Model | Category | Spearman | MSE |
|---|---|---:|---:|
| CNN_GRU+GC | ChromeCRISPR hybrid models | 0.876 | 0.0093 |
| deepCNN+GC | Deep models + GC | 0.873 | 0.0093 |
| CNN_BiLSTM+GC | ChromeCRISPR hybrid models | 0.870 | 0.0096 |
| deepCNN | Deep models | 0.869 | 0.0098 |
| deepGRU | Deep models | 0.868 | 0.0099 |

## Smoke Lane

- Smoke status: `executed`
- Artifacts checked: `20`
- Smoke-passed artifacts: `3`
- Smoke-failed artifacts: `17`
- Heuristic benchmark-shape matches: `0`
- Load warnings: `2`
- Compatibility note: some published checkpoints still have architecture mismatches against the current repo-local model definitions; see `reports/workflow/checkpoint_validator_report.json` for exact failures.

## Preprocessing Snapshot

- Dataset scope: `DeepHF public benchmark bundle`
- Sequence length: `21`
- HPO framework: `Optuna Bayesian optimization`
- Cross-validation: `5-fold`
- Best model batch size: `64`
- Best model learning rate: `0.000209`

## Workflow

```bash
bash scripts/run_snakemake.sh report
bash scripts/run_snakemake.sh smoke
bash scripts/run_snakemake.sh full
```

The default workflow is intentionally scoped to public-repo reproducibility: canonical model inventory, documentation integrity, markdown example/link auditing, and a structured preprocessing manifest around the published artifacts.
