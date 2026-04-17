#!/usr/bin/env python3
"""Optional checkpoint smoke test for the public ChromeCRISPR repo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_markdown(report: dict) -> str:
    lines = [
        "# ChromeCRISPR Checkpoint Smoke Test",
        "",
        f"- Status: `{report['status']}`",
        f"- Executed: `{str(report['executed']).lower()}`",
        f"- Scope: `{report['scope']}`",
        "",
    ]
    if report.get("missing_dependencies"):
        lines.append(f"- Missing dependencies: `{', '.join(report['missing_dependencies'])}`")
    if report.get("total_models") is not None:
        lines.append(f"- Total artifacts checked: `{report['total_models']}`")
        lines.append(f"- Smoke-passed artifacts: `{report['smoke_passed_models']}`")
        lines.append(f"- Smoke-failed artifacts: `{report['smoke_failed_models']}`")
        lines.append(f"- Heuristic benchmark-shape matches: `{report['heuristic_spec_matches']}`")
        lines.append(f"- Load warnings: `{report['models_with_load_warnings']}`")
    lines.append("")
    if report.get("note"):
        lines.append(report["note"])
        lines.append("")
    if report.get("report_path"):
        lines.append(f"- Validator output: `{report['report_path']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument("--validator-report", default="reports/workflow/checkpoint_validator_report.json")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    missing = []
    for dependency in ("numpy", "torch", "scipy", "sklearn", "joblib"):
        try:
            __import__(dependency)
        except Exception:
            missing.append(dependency)

    if missing:
        report = {
            "status": "skipped_missing_dependencies",
            "executed": False,
            "scope": "full_20_model_compatibility_smoke",
            "missing_dependencies": missing,
            "note": "The optional smoke test was not executed because the dedicated smoke environment was not available. Run the workflow `smoke` or `full` target to let the repo bootstrap its own isolated smoke environment automatically.",
            "report_path": None,
            "total_models": None,
            "smoke_passed_models": None,
            "smoke_failed_models": None,
            "heuristic_spec_matches": None,
            "models_with_load_warnings": None,
        }
        write_json(Path(args.json_out), report)
        write_text(Path(args.md_out), render_markdown(report))
        return

    from src.evaluation.validate_all_20_models import ChromeCRISPRValidator

    validator = ChromeCRISPRValidator(models_base_path="models")
    validator.validate_all_models()
    validator.save_validation_report(args.validator_report)
    validator_report = json.loads(Path(args.validator_report).read_text(encoding="utf-8"))

    report = {
        "status": "executed",
        "executed": True,
        "scope": "full_20_model_compatibility_smoke",
        "missing_dependencies": [],
        "note": "The smoke lane verifies that published artifacts load and execute with repo-local code. Synthetic metric deltas versus manuscript values are retained only as heuristic compatibility context.",
        "report_path": args.validator_report,
        "total_models": validator_report["total_models"],
        "smoke_passed_models": validator_report["smoke_passed_models"],
        "smoke_failed_models": validator_report["smoke_failed_models"],
        "heuristic_spec_matches": validator_report["heuristic_spec_matches"],
        "models_with_load_warnings": validator_report["models_with_load_warnings"],
    }
    write_json(Path(args.json_out), report)
    write_text(Path(args.md_out), render_markdown(report))


if __name__ == "__main__":
    main()
