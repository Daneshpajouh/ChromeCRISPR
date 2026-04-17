rule check_repo_integrity:
    input:
        [
            REGISTRY_JSON,
            "scripts/check_repo_integrity.py",
            "scripts/chromecrispr_repo.py",
        ] + PUBLIC_DOC_INPUTS + WORKFLOW_SOURCE_INPUTS + SOURCE_CODE_INPUTS,
    output:
        json=INTEGRITY_JSON,
        md=INTEGRITY_MD,
    shell:
        "python3 scripts/check_repo_integrity.py --json-out {output.json} --md-out {output.md}"
