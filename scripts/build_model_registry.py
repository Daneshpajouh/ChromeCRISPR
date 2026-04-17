#!/usr/bin/env python3
"""Build a canonical model registry for the public ChromeCRISPR repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from chromecrispr_repo import (
    CATEGORY_ORDER,
    CATEGORY_LABELS,
    EXPECTED_CATEGORY_COUNTS,
    collect_registry,
    write_json,
    write_text,
)


def render_markdown(registry: dict) -> str:
    lines = [
        "# ChromeCRISPR Model Registry",
        "",
        f"Generated at: `{registry['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- Canonical model root: `{registry['canonical_model_root']}`",
        f"- Hyperparameter root: `{registry['hyperparameter_root']}`",
        f"- Canonical model count: `{len(registry['models'])}`",
        "",
        "## Category Counts",
        "",
        "| Category | Count | Expected |",
        "|---|---:|---:|",
    ]
    for category in CATEGORY_ORDER:
        lines.append(
            f"| {CATEGORY_LABELS[category]} | {registry['category_counts'].get(category, 0)} | {EXPECTED_CATEGORY_COUNTS[category]} |"
        )
    lines.extend(
        [
            "",
            "## Models",
            "",
            "| Model | Category | Spearman | MSE | Checkpoint | Hyperparameters |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for model in registry["models"]:
        spearman = "" if model["spearman_correlation"] is None else f"{model['spearman_correlation']:.3f}"
        mse = "" if model["mean_squared_error"] is None else f"{model['mean_squared_error']:.4f}"
        lines.append(
            f"| {model['model_name']} | {model['category_label'] or 'Missing'} | {spearman} | {mse} | "
            f"`{model['checkpoint_path'] or 'missing'}` | `{model['hyperparameter_path'] or 'missing'}` |"
        )
    if registry["orphan_checkpoints"] or registry["orphan_hyperparameters"]:
        lines.extend(["", "## Orphans", ""])
        if registry["orphan_checkpoints"]:
            lines.append(f"- Checkpoints without hyperparameter JSON: `{', '.join(registry['orphan_checkpoints'])}`")
        if registry["orphan_hyperparameters"]:
            lines.append(f"- Hyperparameter JSON without checkpoint: `{', '.join(registry['orphan_hyperparameters'])}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    registry = collect_registry()
    write_json(Path(args.json_out), registry)
    write_text(Path(args.md_out), render_markdown(registry))


if __name__ == "__main__":
    main()
