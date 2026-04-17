#!/usr/bin/env python3
"""Render a concise public workflow summary for ChromeCRISPR."""

from __future__ import annotations

import argparse
from pathlib import Path

from chromecrispr_repo import read_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--integrity", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    registry = read_json(Path(args.registry))
    integrity = read_json(Path(args.integrity))
    audit = read_json(Path(args.audit))
    preprocessing = read_json(Path(args.preprocessing))
    top_models = registry["models"][:5]

    lines = [
        "# ChromeCRISPR Public Repo Summary",
        "",
        f"Generated from registry timestamp: `{registry['generated_at']}`",
        "",
        "## Status",
        "",
        f"- Canonical models tracked: `{len(registry['models'])}`",
        f"- Repo integrity passed: `{str(integrity['integrity_passed']).lower()}`",
        f"- Public examples audit passed: `{str(audit['audit_passed']).lower()}`",
        f"- Duplicate checkpoints under `src/models/`: `{integrity['summary']['duplicate_checkpoint_count_under_src_models']}`",
        f"- Orphan checkpoints: `{len(registry['orphan_checkpoints'])}`",
        f"- Orphan hyperparameter files: `{len(registry['orphan_hyperparameters'])}`",
        "",
        "## Top Models",
        "",
        "| Model | Category | Spearman | MSE |",
        "|---|---|---:|---:|",
    ]
    for model in top_models:
        lines.append(
            f"| {model['model_name']} | {model['category_label']} | {model['spearman_correlation']:.3f} | {model['mean_squared_error']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Preprocessing Snapshot",
            "",
            f"- Dataset scope: `{preprocessing['dataset_scope']['dataset']}`",
            f"- Sequence length: `{preprocessing['dataset_scope']['sequence_length']}`",
            f"- HPO framework: `{preprocessing['training_protocol']['optimization_framework']}`",
            f"- Cross-validation: `{preprocessing['training_protocol']['cross_validation']}`",
            f"- Best model batch size: `{preprocessing['best_model_snapshot']['batch_size']}`",
            f"- Best model learning rate: `{preprocessing['best_model_snapshot']['learning_rate']}`",
            "",
            "## Workflow",
            "",
            "```bash",
            "bash scripts/run_snakemake.sh report",
            "bash scripts/run_snakemake.sh smoke",
            "```",
            "",
            "The default workflow is intentionally scoped to public-repo reproducibility: canonical model inventory, documentation integrity, markdown example/link auditing, and a structured preprocessing manifest around the published artifacts.",
            "",
        ]
    )

    if integrity["issues"] or audit["issues"]:
        lines.extend(["## Open Issues", ""])
        for issue in integrity["issues"][:10]:
            lines.append(f"- Integrity: {issue['message']}")
        for issue in audit["issues"][:10]:
            lines.append(f"- Audit: {issue['message']}")
        lines.append("")

    write_text(Path(args.md_out), "\n".join(lines))


if __name__ == "__main__":
    main()
