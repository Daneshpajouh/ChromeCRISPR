# ChromeCRISPR Snakemake Workflow

This workflow is the public reproducibility layer for the ChromeCRISPR repository.

It is intentionally scoped to what the repository can verify directly from the published artifacts:
- canonical model inventory
- documentation and artifact integrity checks
- markdown link and import-example auditing
- a compact summary report for public inspection

## Layout

```text
workflow/
├── Snakefile
├── config/chromecrispr.yaml
├── requirements-snakemake.txt
└── rules/
    ├── audit.smk
    ├── integrity.smk
    └── reporting.smk
```

## Usage

From the repository root:

```bash
bash scripts/run_snakemake.sh report
```

Useful targets:
- `inventory`: build the canonical model registry
- `integrity`: verify checkpoints, documentation, and workflow assets
- `audit`: verify public markdown links and import examples
- `report`: build the full public summary

## Outputs

The workflow writes deterministic outputs to `reports/workflow/`:
- `model_registry.json`
- `model_registry.md`
- `repo_integrity.json`
- `repo_integrity.md`
- `public_examples_audit.json`
- `public_examples_audit.md`
- `public_repo_summary.md`
