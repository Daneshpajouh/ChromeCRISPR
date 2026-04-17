# ChromeCRISPR Workflow

This directory contains the Snakemake workflow used for repository checks and report generation.

## Targets

| Target | Purpose |
|---|---|
| `inventory` | build the model registry |
| `integrity` | check repository structure and required assets |
| `audit` | check markdown links and example imports |
| `preprocessing` | build the preprocessing manifest |
| `report` | run the report set without the smoke test |
| `smoke` | run the checkpoint smoke test |
| `full` | run `report` and `smoke` |

Run from the repository root:

```bash
bash scripts/run_snakemake.sh report
bash scripts/run_snakemake.sh smoke
bash scripts/run_snakemake.sh full
```

## Files

```text
workflow/
├── Snakefile
├── config/chromecrispr.yaml
├── requirements-snakemake.txt
├── requirements-smoke.txt
└── rules/
    ├── audit.smk
    ├── integrity.smk
    ├── preprocessing.smk
    ├── reporting.smk
    └── smoke.smk
```

## Outputs

The workflow writes to `reports/workflow/`:
- `model_registry.json` / `model_registry.md`
- `repo_integrity.json` / `repo_integrity.md`
- `public_examples_audit.json` / `public_examples_audit.md`
- `preprocessing_manifest.json` / `preprocessing_manifest.md`
- `public_repo_summary.md`
- `public_repo_full_summary.md`
- `checkpoint_smoke.json` / `checkpoint_smoke.md`
- `checkpoint_validator_report.json`

## Smoke test

The smoke target uses a separate environment in `.smoke-venv/` built from `workflow/requirements-smoke.txt`.

The smoke test checks whether the published artifacts can be loaded and executed with the Python code included in this repository. It does not attempt to reproduce the manuscript benchmark values from raw data.
