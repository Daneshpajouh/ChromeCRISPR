rule check_repo_integrity:
    input:
        REGISTRY_JSON,
    output:
        json=INTEGRITY_JSON,
        md=INTEGRITY_MD,
    shell:
        "python3 scripts/check_repo_integrity.py --json-out {output.json} --md-out {output.md}"
