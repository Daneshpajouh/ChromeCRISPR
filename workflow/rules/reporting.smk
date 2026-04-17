rule build_model_registry:
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
    output:
        md=SUMMARY_MD,
    shell:
        (
            "python3 scripts/build_public_repo_summary.py "
            "--registry {input.registry} --integrity {input.integrity} "
            "--audit {input.audit} --md-out {output.md}"
        )
