PREPROCESSING_JSON = f"{REPORTS_DIR}/preprocessing_manifest.json"
PREPROCESSING_MD = f"{REPORTS_DIR}/preprocessing_manifest.md"
BEST_MODEL_HPARAMS = "docs/hyperparameters/CNN_GRU+GC_hyperparameters.json"

rule build_preprocessing_manifest:
    input:
        BEST_MODEL_HPARAMS,
        "docs/training_procedures/README.md",
        "scripts/build_preprocessing_manifest.py",
        "scripts/chromecrispr_repo.py",
    output:
        json=PREPROCESSING_JSON,
        md=PREPROCESSING_MD,
    shell:
        (
            "python3 scripts/build_preprocessing_manifest.py "
            "--best-model {input[0]} --json-out {output.json} --md-out {output.md}"
        )
