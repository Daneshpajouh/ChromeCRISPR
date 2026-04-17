SMOKE_JSON = f"{REPORTS_DIR}/checkpoint_smoke.json"
SMOKE_MD = f"{REPORTS_DIR}/checkpoint_smoke.md"

rule run_checkpoint_smoke:
    input:
        [
            "scripts/run_checkpoint_smoke.py",
            "src/evaluation/validate_all_20_models.py",
            "src/evaluation/metrics.py",
            "src/models/__init__.py",
            "src/models/cnn_model.py",
            "src/models/rnn_models.py",
            "src/models/hybrid_models.py",
        ] + CANONICAL_MODEL_INPUTS,
    output:
        json=SMOKE_JSON,
        md=SMOKE_MD,
    shell:
        "python3 scripts/run_checkpoint_smoke.py --json-out {output.json} --md-out {output.md}"
