rule build_model_registry:
    input:
        CANONICAL_MODEL_INPUTS + HYPERPARAMETER_INPUTS + [
            "scripts/build_model_registry.py",
            "scripts/chromecrispr_repo.py",
            "models/README.md",
            "docs/README.md",
        ],
    output:
        json=REGISTRY_JSON,
        md=REGISTRY_MD,
    shell:
        "python3 scripts/build_model_registry.py --json-out {output.json} --md-out {output.md}"


rule build_public_repo_summary:
    input:
        registry=REGISTRY_JSON,
        integrity=INTEGRITY_JSON,
        audit=AUDIT_JSON,
        preprocessing=PREPROCESSING_JSON,
        script="scripts/build_public_repo_summary.py",
        helper="scripts/chromecrispr_repo.py",
    output:
        md=SUMMARY_MD,
    shell:
        (
            "python3 scripts/build_public_repo_summary.py "
            "--registry {input.registry} --integrity {input.integrity} "
            "--audit {input.audit} --preprocessing {input.preprocessing} "
            "--md-out {output.md}"
        )
