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
    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append("numpy")
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")
    try:
        import scipy  # noqa: F401
    except Exception:
        missing.append("scipy")

    if missing:
        report = {
            "status": "skipped_missing_dependencies",
            "executed": False,
            "scope": "full_20_model_compatibility_smoke",
            "missing_dependencies": missing,
            "note": "The optional smoke test was not executed because the current local Python environment does not include the ML dependencies needed by the compatibility validator. Install the project requirements and rerun the `smoke` target if you want this report to execute locally.",
            "report_path": None,
        }
        write_json(Path(args.json_out), report)
        write_text(Path(args.md_out), render_markdown(report))
        return

    from src.evaluation.validate_all_20_models import ChromeCRISPRValidator

    validator = ChromeCRISPRValidator(models_base_path="models")
    validator.validate_all_models()
    validator.save_validation_report(args.validator_report)

    report = {
        "status": "executed",
        "executed": True,
        "scope": "full_20_model_compatibility_smoke",
        "missing_dependencies": [],
        "note": "This is a compatibility-oriented smoke test using synthetic inputs. It is not a reproduction of the manuscript benchmark dataset.",
        "report_path": args.validator_report,
    }
    write_json(Path(args.json_out), report)
    write_text(Path(args.md_out), render_markdown(report))


if __name__ == "__main__":
    main()
