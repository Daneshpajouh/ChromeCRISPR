# ChromeCRISPR Public Repo Summary

Generated from registry timestamp: `2026-04-17T01:27:38Z`

## Status

- Canonical models tracked: `20`
- Repo integrity passed: `true`
- Public examples audit passed: `true`
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

## Workflow

```bash
bash scripts/run_snakemake.sh report
```

The workflow is intentionally scoped to public-repo reproducibility: canonical model inventory, documentation integrity, and markdown example/link auditing around the published artifacts.
