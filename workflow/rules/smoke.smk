SMOKE_JSON = f"{REPORTS_DIR}/checkpoint_smoke.json"
SMOKE_MD = f"{REPORTS_DIR}/checkpoint_smoke.md"
VALIDATOR_JSON = f"{REPORTS_DIR}/checkpoint_validator_report.json"

rule run_checkpoint_smoke:
    input:
        [
            "scripts/run_checkpoint_smoke.py",
            "scripts/run_smoke_env.sh",
            "src/evaluation/validate_all_20_models.py",
            "src/evaluation/metrics.py",
            "src/models/__init__.py",
            "src/models/cnn_model.py",
            "src/models/rnn_models.py",
            "src/models/hybrid_models.py",
            "workflow/requirements-smoke.txt",
        ] + CANONICAL_MODEL_INPUTS,
    output:
        json=SMOKE_JSON,
        md=SMOKE_MD,
        validator=VALIDATOR_JSON,
    shell:
        (
            "bash scripts/run_smoke_env.sh "
            "--json-out {output.json} "
            "--md-out {output.md} "
            "--validator-report {output.validator}"
        )
