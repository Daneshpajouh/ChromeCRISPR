#!/usr/bin/env python3
"""Shared repository helpers for ChromeCRISPR workflow scripts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

CATEGORY_ORDER = [
    "base_models",
    "base_models_with_gc",
    "deep_models",
    "deep_models_with_gc",
    "chromecrispr_hybrid_models",
]

CATEGORY_LABELS = {
    "base_models": "Base models",
    "base_models_with_gc": "Base models + GC",
    "deep_models": "Deep models",
    "deep_models_with_gc": "Deep models + GC",
    "chromecrispr_hybrid_models": "ChromeCRISPR hybrid models",
}

EXPECTED_CATEGORY_COUNTS = {
    "base_models": 5,
    "base_models_with_gc": 4,
    "deep_models": 4,
    "deep_models_with_gc": 4,
    "chromecrispr_hybrid_models": 3,
}

PACKAGE_READMES = [
    "README.md",
    "docs/README.md",
    "models/README.md",
    "src/README.md",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _checkpoint_suffix(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def collect_registry() -> Dict[str, Any]:
    root = repo_root()
    model_root = root / "models"
    hyperparameter_root = root / "docs" / "hyperparameters"

    checkpoint_index: Dict[str, Dict[str, Any]] = {}
    category_counts: Counter[str] = Counter()

    for category in CATEGORY_ORDER:
        category_dir = model_root / category
        if not category_dir.exists():
            continue
        for artifact in sorted(category_dir.iterdir()):
            if artifact.suffix.lower() not in {".pth", ".joblib"}:
                continue
            model_name = artifact.stem
            checkpoint_index[model_name] = {
                "checkpoint_path": artifact.relative_to(root).as_posix(),
                "checkpoint_format": _checkpoint_suffix(artifact),
                "category": category,
                "category_label": CATEGORY_LABELS.get(category, category),
                "model_readme_path": (category_dir / "README.md").relative_to(root).as_posix(),
            }
            category_counts[category] += 1

    hyperparameter_index: Dict[str, Dict[str, Any]] = {}
    for json_path in sorted(hyperparameter_root.glob("*_hyperparameters.json")):
        payload = read_json(json_path)
        model_name = payload.get("model_name", json_path.stem.replace("_hyperparameters", ""))
        performance = payload.get("performance", {})
        training = payload.get("training_details", {})
        params = payload.get("parameter_counts", {})
        hyperparameter_index[model_name] = {
            "hyperparameter_path": json_path.relative_to(root).as_posix(),
            "hyperparameter_readme_path": (json_path.parent / f"{model_name}_README.md").relative_to(root).as_posix(),
            "architecture": payload.get("architecture"),
            "model_type": payload.get("model_type"),
            "performance": {
                "spearman_correlation": performance.get("spearman_correlation"),
                "mean_squared_error": performance.get("mean_squared_error"),
            },
            "parameter_count": params.get("total_parameters"),
            "training_time": training.get("training_time"),
        }

    all_model_names = sorted(set(checkpoint_index) | set(hyperparameter_index))
    models: List[Dict[str, Any]] = []
    orphan_checkpoints = sorted(set(checkpoint_index) - set(hyperparameter_index))
    orphan_hyperparameters = sorted(set(hyperparameter_index) - set(checkpoint_index))

    for model_name in all_model_names:
        checkpoint = checkpoint_index.get(model_name, {})
        hyperparams = hyperparameter_index.get(model_name, {})
        spearman = hyperparams.get("performance", {}).get("spearman_correlation")
        mse = hyperparams.get("performance", {}).get("mean_squared_error")
        models.append(
            {
                "model_name": model_name,
                "category": checkpoint.get("category"),
                "category_label": checkpoint.get("category_label"),
                "checkpoint_path": checkpoint.get("checkpoint_path"),
                "checkpoint_format": checkpoint.get("checkpoint_format"),
                "checkpoint_exists": bool(checkpoint),
                "model_readme_path": checkpoint.get("model_readme_path"),
                "model_readme_exists": bool(checkpoint.get("model_readme_path") and (root / checkpoint["model_readme_path"]).exists()),
                "hyperparameter_path": hyperparams.get("hyperparameter_path"),
                "hyperparameter_exists": bool(hyperparams),
                "hyperparameter_readme_path": hyperparams.get("hyperparameter_readme_path"),
                "hyperparameter_readme_exists": bool(
                    hyperparams.get("hyperparameter_readme_path")
                    and (root / hyperparams["hyperparameter_readme_path"]).exists()
                ),
                "architecture": hyperparams.get("architecture"),
                "model_type": hyperparams.get("model_type"),
                "spearman_correlation": spearman,
                "mean_squared_error": mse,
                "parameter_count": hyperparams.get("parameter_count"),
                "training_time": hyperparams.get("training_time"),
            }
        )

    models.sort(
        key=lambda row: (
            row["spearman_correlation"] is None,
            -(row["spearman_correlation"] or 0.0),
            row["model_name"],
        )
    )

    return {
        "generated_at": utc_timestamp(),
        "repo_root": str(root),
        "canonical_model_root": "models",
        "hyperparameter_root": "docs/hyperparameters",
        "expected_model_count": 20,
        "category_counts": {category: category_counts.get(category, 0) for category in CATEGORY_ORDER},
        "category_labels": CATEGORY_LABELS,
        "models": models,
        "orphan_checkpoints": orphan_checkpoints,
        "orphan_hyperparameters": orphan_hyperparameters,
    }
