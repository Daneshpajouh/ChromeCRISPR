#!/usr/bin/env python3
"""Build a structured preprocessing and training manifest for the public repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from chromecrispr_repo import read_json, utc_timestamp, write_json, write_text


def build_manifest(best_model_path: Path) -> dict:
    best = read_json(best_model_path)
    return {
        "generated_at": utc_timestamp(),
        "source_documents": {
            "training_procedures": "docs/training_procedures/README.md",
            "best_model_hyperparameters": best_model_path.as_posix(),
        },
        "dataset_scope": {
            "dataset": "DeepHF public benchmark bundle",
            "sequence_length": 21,
            "target": "sgRNA on-target activity / indel frequency",
            "notes": "The public repo documents the study setup but does not bundle the full raw training dataset.",
        },
        "preprocessing_pipeline": [
            {
                "step": 1,
                "name": "sequence_validation",
                "description": "Validate 21-mer sgRNA sequences before encoding.",
                "inputs": ["DNA sequence"],
                "outputs": ["validated 21-mer sequence"],
            },
            {
                "step": 2,
                "name": "one_hot_encoding",
                "description": "Encode 21-mer sequences as 84 input features (21 × 4 nucleotides).",
                "inputs": ["validated 21-mer sequence"],
                "outputs": ["84-dimensional sequence representation"],
            },
            {
                "step": 3,
                "name": "embedding_layer",
                "description": "Project sequence representation into a learned 128-dimensional embedding inside the neural models.",
                "inputs": ["84-dimensional sequence representation"],
                "outputs": ["128-dimensional latent sequence embedding"],
            },
            {
                "step": 4,
                "name": "gc_content_calculation",
                "description": "Compute GC content as (G + C) / sequence length for the GC-aware variants.",
                "inputs": ["validated 21-mer sequence"],
                "outputs": ["scalar GC content in [0, 1]"],
            },
            {
                "step": 5,
                "name": "feature_normalization",
                "description": "Optional StandardScaler normalization for numerical features where used by the study pipeline.",
                "inputs": ["numerical features"],
                "outputs": ["normalized features"],
            },
            {
                "step": 6,
                "name": "split_and_validation",
                "description": "Apply the documented 70/15/15 train/validation/test split with 5-fold cross-validation inside model selection.",
                "inputs": ["preprocessed feature matrix", "activity labels"],
                "outputs": ["train, validation, and test partitions"],
            },
        ],
        "training_protocol": {
            "optimizer": "Adam",
            "loss": "Mean Squared Error",
            "optimization_framework": "Optuna Bayesian optimization",
            "optimization_target": "maximize Spearman correlation",
            "trials_per_model": 100,
            "cross_validation": "5-fold",
            "early_stopping_patience": 10,
            "learning_rate_schedule": "ReduceLROnPlateau",
            "reported_hardware": "NVIDIA V100 Volta, 32GB HBM2",
        },
        "best_model_snapshot": {
            "model_name": best.get("model_name"),
            "model_type": best.get("model_type"),
            "spearman_correlation": best.get("performance", {}).get("spearman_correlation"),
            "mean_squared_error": best.get("performance", {}).get("mean_squared_error"),
            "learning_rate": best.get("hyperparameters", {}).get("learning_rate", {}).get("best_value"),
            "batch_size": best.get("hyperparameters", {}).get("batch_size", {}).get("best_value"),
            "dropout_rate": best.get("hyperparameters", {}).get("dropout_rate", {}).get("best_value"),
            "epochs": best.get("hyperparameters", {}).get("epochs", {}).get("best_value"),
        },
        "public_repo_boundary": {
            "canonical_checkpoint_root": "models/",
            "workflow_focus": "artifact verification and documentation clarity",
            "not_bundled": ["raw training data", "full production retraining pipeline"],
        },
    }


def render_markdown(manifest: dict) -> str:
    lines = [
        "# ChromeCRISPR Preprocessing Manifest",
        "",
        f"Generated at: `{manifest['generated_at']}`",
        "",
        "## Dataset Scope",
        "",
        f"- Dataset: `{manifest['dataset_scope']['dataset']}`",
        f"- Sequence length: `{manifest['dataset_scope']['sequence_length']}`",
        f"- Prediction target: `{manifest['dataset_scope']['target']}`",
        f"- Boundary note: {manifest['dataset_scope']['notes']}",
        "",
        "## Preprocessing Pipeline",
        "",
        "| Step | Name | Output |",
        "|---:|---|---|",
    ]
    for step in manifest["preprocessing_pipeline"]:
        lines.append(f"| {step['step']} | {step['name']} | {', '.join(step['outputs'])} |")
    lines.extend([
        "",
        "## Training Protocol",
        "",
        f"- Optimizer: `{manifest['training_protocol']['optimizer']}`",
        f"- Loss: `{manifest['training_protocol']['loss']}`",
        f"- HPO: `{manifest['training_protocol']['optimization_framework']}` with `{manifest['training_protocol']['trials_per_model']}` trials/model",
        f"- Cross-validation: `{manifest['training_protocol']['cross_validation']}`",
        f"- Early stopping patience: `{manifest['training_protocol']['early_stopping_patience']}`",
        f"- LR schedule: `{manifest['training_protocol']['learning_rate_schedule']}`",
        f"- Reported hardware: `{manifest['training_protocol']['reported_hardware']}`",
        "",
        "## Best Model Snapshot",
        "",
        f"- Model: `{manifest['best_model_snapshot']['model_name']}`",
        f"- Spearman: `{manifest['best_model_snapshot']['spearman_correlation']}`",
        f"- MSE: `{manifest['best_model_snapshot']['mean_squared_error']}`",
        f"- Learning rate: `{manifest['best_model_snapshot']['learning_rate']}`",
        f"- Batch size: `{manifest['best_model_snapshot']['batch_size']}`",
        f"- Dropout: `{manifest['best_model_snapshot']['dropout_rate']}`",
        f"- Epochs: `{manifest['best_model_snapshot']['epochs']}`",
        "",
        "## Public Repo Boundary",
        "",
        f"- Canonical checkpoint root: `{manifest['public_repo_boundary']['canonical_checkpoint_root']}`",
        f"- Workflow focus: {manifest['public_repo_boundary']['workflow_focus']}",
        f"- Not bundled: `{', '.join(manifest['public_repo_boundary']['not_bundled'])}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-model", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    manifest = build_manifest(Path(args.best_model))
    write_json(Path(args.json_out), manifest)
    write_text(Path(args.md_out), render_markdown(manifest))


if __name__ == "__main__":
    main()
