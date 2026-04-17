DOCS_TO_AUDIT = " ".join(config["docs_to_audit"])

rule audit_public_examples:
    output:
        json=AUDIT_JSON,
        md=AUDIT_MD,
    params:
        docs=DOCS_TO_AUDIT,
    shell:
        "python3 scripts/audit_public_examples.py --json-out {output.json} --md-out {output.md} --docs {params.docs}"
