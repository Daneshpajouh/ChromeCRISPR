#!/usr/bin/env python3
"""Check public repo integrity for models, docs, and workflow surface."""

from __future__ import annotations

import argparse
from pathlib import Path

from chromecrispr_repo import EXPECTED_CATEGORY_COUNTS, PACKAGE_READMES, collect_registry, repo_root, write_json, write_text


def build_report() -> dict:
    root = repo_root()
    registry = collect_registry()
    issues = []

    if len(registry["models"]) != registry["expected_model_count"]:
        issues.append(
            {
                "code": "model_count_mismatch",
                "message": f"Expected {registry['expected_model_count']} canonical models but found {len(registry['models'])}.",
            }
        )

    for category, expected in EXPECTED_CATEGORY_COUNTS.items():
        observed = registry["category_counts"].get(category, 0)
        if observed != expected:
            issues.append(
                {
                    "code": "category_count_mismatch",
                    "message": f"Category {category} expected {expected} models but found {observed}.",
                }
            )

    for model in registry["models"]:
        if not model["checkpoint_exists"]:
            issues.append({"code": "missing_checkpoint", "message": f"{model['model_name']} is missing a canonical checkpoint."})
        if not model["hyperparameter_exists"]:
            issues.append({"code": "missing_hyperparameters", "message": f"{model['model_name']} is missing hyperparameter JSON."})
        if not model["model_readme_exists"]:
            issues.append({"code": "missing_category_readme", "message": f"{model['model_name']} is missing its category README."})
        if not model["hyperparameter_readme_exists"]:
            issues.append({"code": "missing_hyperparameter_readme", "message": f"{model['model_name']} is missing its hyperparameter README."})

    for readme in PACKAGE_READMES:
        if not (root / readme).exists():
            issues.append({"code": "missing_public_readme", "message": f"Required public README is missing: {readme}."})

    for required in [
        "workflow/Snakefile",
        "workflow/config/chromecrispr.yaml",
        "workflow/README.md",
        "workflow/requirements-snakemake.txt",
        "workflow/requirements-smoke.txt",
        "scripts/run_snakemake.sh",
        "scripts/run_smoke_env.sh",
        "scripts/build_preprocessing_manifest.py",
        "scripts/run_checkpoint_smoke.py",
        "src/models/__init__.py",
        "src/evaluation/__init__.py",
        "src/training/__init__.py",
        "src/models/README.md",
    ]:
        if not (root / required).exists():
            issues.append({"code": "missing_workflow_asset", "message": f"Missing workflow/package asset: {required}."})

    duplicate_checkpoints = sorted((root / "src/models").glob("**/*.pth")) + sorted((root / "src/models").glob("**/*.joblib"))
    if duplicate_checkpoints:
        issues.append(
            {
                "code": "noncanonical_duplicate_checkpoints",
                "message": f"Found {len(duplicate_checkpoints)} non-canonical checkpoint files under src/models/. Published artifacts must live only under models/.",
            }
        )

    return {
        "generated_at": registry["generated_at"],
        "integrity_passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "summary": {
            "canonical_model_count": len(registry["models"]),
            "expected_model_count": registry["expected_model_count"],
            "category_counts": registry["category_counts"],
            "duplicate_checkpoint_count_under_src_models": len(duplicate_checkpoints),
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# ChromeCRISPR Repo Integrity",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"- Integrity passed: `{str(report['integrity_passed']).lower()}`",
        f"- Issue count: `{report['issue_count']}`",
        f"- Duplicate checkpoints under `src/models/`: `{report['summary']['duplicate_checkpoint_count_under_src_models']}`",
        "",
    ]
    if report["issues"]:
        lines.extend(["## Issues", ""])
        for issue in report["issues"]:
            lines.append(f"- `{issue['code']}`: {issue['message']}")
    else:
        lines.extend(["## Status", "", "- All canonical model, documentation, workflow, and code/artifact boundary checks passed."])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    report = build_report()
    write_json(Path(args.json_out), report)
    write_text(Path(args.md_out), render_markdown(report))


if __name__ == "__main__":
    main()
