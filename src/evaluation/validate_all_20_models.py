#!/usr/bin/env python3
"""
ChromeCRISPR checkpoint compatibility validator.

This script is a compatibility-oriented smoke-test utility around the published
model files. It verifies that released artifacts can be loaded and exercised in
repo-local code without pretending to reproduce manuscript benchmark metrics.
Synthetic regression metrics are retained only as a heuristic side signal so
load/forward execution remains the primary pass criterion.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import joblib
except Exception:  # pragma: no cover - joblib is expected in the smoke env
    joblib = None

try:
    from src.evaluation.metrics import ChromeCRISPRMetrics
    from src.models.cnn_model import create_cnn_model, create_deep_cnn_model
    from src.models.hybrid_models import create_cnn_bilstm_model, create_cnn_gru_model, create_cnn_lstm_model
    from src.models.rnn_models import create_bilstm_model, create_gru_model, create_lstm_model
except ImportError:  # pragma: no cover - repo-local fallback
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.evaluation.metrics import ChromeCRISPRMetrics
    from src.models.cnn_model import create_cnn_model, create_deep_cnn_model
    from src.models.hybrid_models import create_cnn_bilstm_model, create_cnn_gru_model, create_cnn_lstm_model
    from src.models.rnn_models import create_bilstm_model, create_gru_model, create_lstm_model


@dataclass
class ValidationResult:
    """Compatibility smoke result for one published artifact."""

    model_name: str
    artifact_type: str
    artifact_present: bool
    load_passed: bool
    forward_pass_passed: bool
    smoke_passed: bool
    heuristic_spec_match: Optional[bool]
    missing_key_count: int
    unexpected_key_count: int
    spearman_correlation: float
    mse: float
    rmse: float
    mae: float
    r2: float
    pearson_correlation: float
    error_message: Optional[str] = None
    note: Optional[str] = None


class ChromeCRISPRValidator:
    """Compatibility-oriented validator for the published ChromeCRISPR artifacts."""

    def __init__(self, models_base_path: str = "../models"):
        self.models_base_path = Path(models_base_path)
        self.metrics_calculator = ChromeCRISPRMetrics()
        self.performance_specs = {
            "CNN_GRU+GC": {"spearman": 0.8760, "mse": 0.0093},
            "deepCNN+GC": {"spearman": 0.8730, "mse": 0.0093},
            "CNN_BiLSTM+GC": {"spearman": 0.8700, "mse": 0.0096},
            "CNN_LSTM+GC": {"spearman": 0.8670, "mse": 0.0115},
            "deepBiLSTM+GC": {"spearman": 0.8670, "mse": 0.0098},
            "deepGRU+GC": {"spearman": 0.8670, "mse": 0.0098},
            "deepGRU": {"spearman": 0.8680, "mse": 0.0099},
            "deepCNN": {"spearman": 0.8690, "mse": 0.0098},
            "deepLSTM": {"spearman": 0.8620, "mse": 0.0103},
            "deepBiLSTM": {"spearman": 0.8620, "mse": 0.0104},
            "deepLSTM+GC": {"spearman": 0.8600, "mse": 0.0104},
            "LSTM+GC": {"spearman": 0.8560, "mse": 0.0112},
            "BiLSTM+GC": {"spearman": 0.8550, "mse": 0.0110},
            "BiLSTM": {"spearman": 0.8430, "mse": 0.0120},
            "GRU+GC": {"spearman": 0.8400, "mse": 0.0122},
            "GRU": {"spearman": 0.8370, "mse": 0.0121},
            "LSTM": {"spearman": 0.8370, "mse": 0.0122},
            "CNN": {"spearman": 0.7930, "mse": 0.0161},
            "CNN+GC": {"spearman": 0.7810, "mse": 0.0170},
            "RF": {"spearman": 0.7890, "mse": 0.0161},
        }
        self.tolerance = 0.001
        self.results: List[ValidationResult] = []
        self.rng = np.random.default_rng(42)

    def _load_state_dict(self, model: nn.Module, model_path: Path):
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch fallback
            state_dict = torch.load(model_path, map_location="cpu")
        return model.load_state_dict(state_dict, strict=False)

    def _create_torch_model(self, model_path: Path) -> Optional[nn.Module]:
        name = str(model_path)
        if "CNN_GRU" in name:
            return create_cnn_gru_model()
        if "CNN_LSTM" in name:
            return create_cnn_lstm_model()
        if "CNN_BiLSTM" in name:
            return create_cnn_bilstm_model()
        if "BiLSTM" in name and "CNN" not in name:
            return create_bilstm_model(num_layers=2)
        if "GRU" in name and "CNN" not in name:
            return create_gru_model(num_layers=4 if "deep" in name else 2)
        if "LSTM" in name and "CNN" not in name and "BiLSTM" not in name:
            return create_lstm_model(num_layers=4 if "deep" in name else 2)
        if "CNN" in name and "GRU" not in name and "LSTM" not in name:
            return create_deep_cnn_model() if "deep" in name else create_cnn_model()
        return None

    def _synthetic_targets(self, n_samples: int) -> np.ndarray:
        return self.rng.uniform(0.05, 1.0, size=n_samples).astype(np.float32)

    def _synthetic_sequences(self, n_samples: int = 128) -> np.ndarray:
        return self.rng.integers(0, 4, size=(n_samples, 21), dtype=np.int64)

    def _synthetic_gc_content(self, sequences: np.ndarray) -> np.ndarray:
        gc_hits = np.isin(sequences, [1, 2]).sum(axis=1).astype(np.float32) / sequences.shape[1]
        return gc_hits.reshape(-1, 1)

    def _synthetic_rf_features(self, n_samples: int = 128) -> np.ndarray:
        return self.rng.random((n_samples, 84), dtype=np.float32)

    def _heuristic_spec_match(self, model_name: str, spearman: float, mse: float) -> Optional[bool]:
        specs = self.performance_specs.get(model_name)
        if not specs:
            return None
        return abs(spearman - specs["spearman"]) <= self.tolerance and abs(mse - specs["mse"]) <= self.tolerance

    def _torch_result(self, model_path: Path, model_name: str) -> ValidationResult:
        if not model_path.exists():
            return ValidationResult(
                model_name=model_name,
                artifact_type="torch_checkpoint",
                artifact_present=False,
                load_passed=False,
                forward_pass_passed=False,
                smoke_passed=False,
                heuristic_spec_match=None,
                missing_key_count=0,
                unexpected_key_count=0,
                spearman_correlation=0.0,
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2=0.0,
                pearson_correlation=0.0,
                error_message=f"Missing checkpoint: {model_path}",
            )

        model = self._create_torch_model(model_path)
        if model is None:
            return ValidationResult(
                model_name=model_name,
                artifact_type="torch_checkpoint",
                artifact_present=True,
                load_passed=False,
                forward_pass_passed=False,
                smoke_passed=False,
                heuristic_spec_match=None,
                missing_key_count=0,
                unexpected_key_count=0,
                spearman_correlation=0.0,
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2=0.0,
                pearson_correlation=0.0,
                error_message="No repo-local factory is available for this checkpoint type.",
            )

        try:
            load_result = self._load_state_dict(model, model_path)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            model.eval()

            sequences = self._synthetic_sequences()
            targets = self._synthetic_targets(len(sequences))
            with torch.no_grad():
                inputs = torch.from_numpy(sequences).long()
                if any(tag in model_name for tag in ("CNN_GRU+GC", "CNN_LSTM+GC", "CNN_BiLSTM+GC")):
                    gc_content = torch.from_numpy(self._synthetic_gc_content(sequences)).float()
                    predictions = model(inputs, gc_content).detach().cpu().numpy().reshape(-1)
                else:
                    predictions = model(inputs).detach().cpu().numpy().reshape(-1)

            if not np.all(np.isfinite(predictions)):
                raise ValueError("Forward pass produced non-finite predictions.")

            metrics = self.metrics_calculator.calculate_metrics(targets, predictions, model_name)
            warning_bits = []
            if missing_keys:
                warning_bits.append(f"{len(missing_keys)} missing state_dict keys")
            if unexpected_keys:
                warning_bits.append(f"{len(unexpected_keys)} unexpected state_dict keys")
            note = "; ".join(warning_bits) if warning_bits else None

            return ValidationResult(
                model_name=model_name,
                artifact_type="torch_checkpoint",
                artifact_present=True,
                load_passed=True,
                forward_pass_passed=True,
                smoke_passed=True,
                heuristic_spec_match=self._heuristic_spec_match(model_name, metrics["spearman_corr"], metrics["mse"]),
                missing_key_count=len(missing_keys),
                unexpected_key_count=len(unexpected_keys),
                spearman_correlation=metrics["spearman_corr"],
                mse=metrics["mse"],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                r2=metrics["r2"],
                pearson_correlation=metrics["pearson_corr"],
                note=note,
            )
        except Exception as exc:
            return ValidationResult(
                model_name=model_name,
                artifact_type="torch_checkpoint",
                artifact_present=True,
                load_passed=False,
                forward_pass_passed=False,
                smoke_passed=False,
                heuristic_spec_match=None,
                missing_key_count=0,
                unexpected_key_count=0,
                spearman_correlation=0.0,
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2=0.0,
                pearson_correlation=0.0,
                error_message=str(exc),
            )

    def _rf_result(self, model_path: Path, model_name: str) -> ValidationResult:
        if not model_path.exists():
            return ValidationResult(
                model_name=model_name,
                artifact_type="joblib_model",
                artifact_present=False,
                load_passed=False,
                forward_pass_passed=False,
                smoke_passed=False,
                heuristic_spec_match=None,
                missing_key_count=0,
                unexpected_key_count=0,
                spearman_correlation=0.0,
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2=0.0,
                pearson_correlation=0.0,
                error_message=f"Missing artifact: {model_path}",
            )

        if joblib is None:
            return ValidationResult(
                model_name=model_name,
                artifact_type="joblib_model",
                artifact_present=True,
                load_passed=False,
                forward_pass_passed=False,
                smoke_passed=False,
                heuristic_spec_match=None,
                missing_key_count=0,
                unexpected_key_count=0,
                spearman_correlation=0.0,
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2=0.0,
                pearson_correlation=0.0,
                error_message="joblib is not available in the current environment.",
            )

        try:
            model = joblib.load(model_path)
            features = self._synthetic_rf_features()
            targets = self._synthetic_targets(len(features))
            predictions = np.asarray(model.predict(features), dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(predictions)):
                raise ValueError("RF prediction produced non-finite values.")
            metrics = self.metrics_calculator.calculate_metrics(targets, predictions, model_name)
            return ValidationResult(
                model_name=model_name,
                artifact_type="joblib_model",
                artifact_present=True,
                load_passed=True,
                forward_pass_passed=True,
                smoke_passed=True,
                heuristic_spec_match=self._heuristic_spec_match(model_name, metrics["spearman_corr"], metrics["mse"]),
                missing_key_count=0,
                unexpected_key_count=0,
                spearman_correlation=metrics["spearman_corr"],
                mse=metrics["mse"],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                r2=metrics["r2"],
                pearson_correlation=metrics["pearson_corr"],
                note="RandomForestRegressor loaded from the published joblib artifact.",
            )
        except Exception as exc:
            return ValidationResult(
                model_name=model_name,
                artifact_type="joblib_model",
                artifact_present=True,
                load_passed=False,
                forward_pass_passed=False,
                smoke_passed=False,
                heuristic_spec_match=None,
                missing_key_count=0,
                unexpected_key_count=0,
                spearman_correlation=0.0,
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2=0.0,
                pearson_correlation=0.0,
                error_message=str(exc),
            )

    def validate_model(self, model_path: Path, model_name: str) -> ValidationResult:
        if model_path.suffix.lower() == ".joblib":
            return self._rf_result(model_path, model_name)
        return self._torch_result(model_path, model_name)

    def validate_all_models(self) -> List[ValidationResult]:
        print("Starting checkpoint compatibility validation for all 20 ChromeCRISPR models")
        print("=" * 70)

        model_files = {
            "CNN": self.models_base_path / "base_models" / "CNN.pth",
            "GRU": self.models_base_path / "base_models" / "GRU.pth",
            "LSTM": self.models_base_path / "base_models" / "LSTM.pth",
            "BiLSTM": self.models_base_path / "base_models" / "BiLSTM.pth",
            "RF": self.models_base_path / "base_models" / "RF.joblib",
            "CNN+GC": self.models_base_path / "base_models_with_gc" / "CNN+GC.pth",
            "GRU+GC": self.models_base_path / "base_models_with_gc" / "GRU+GC.pth",
            "LSTM+GC": self.models_base_path / "base_models_with_gc" / "LSTM+GC.pth",
            "BiLSTM+GC": self.models_base_path / "base_models_with_gc" / "BiLSTM+GC.pth",
            "deepCNN": self.models_base_path / "deep_models" / "deepCNN.pth",
            "deepGRU": self.models_base_path / "deep_models" / "deepGRU.pth",
            "deepLSTM": self.models_base_path / "deep_models" / "deepLSTM.pth",
            "deepBiLSTM": self.models_base_path / "deep_models" / "deepBiLSTM.pth",
            "deepCNN+GC": self.models_base_path / "deep_models_with_gc" / "deepCNN+GC.pth",
            "deepGRU+GC": self.models_base_path / "deep_models_with_gc" / "deepGRU+GC.pth",
            "deepLSTM+GC": self.models_base_path / "deep_models_with_gc" / "deepLSTM+GC.pth",
            "deepBiLSTM+GC": self.models_base_path / "deep_models_with_gc" / "deepBiLSTM+GC.pth",
            "CNN_GRU+GC": self.models_base_path / "chromecrispr_hybrid_models" / "CNN_GRU+GC.pth",
            "CNN_LSTM+GC": self.models_base_path / "chromecrispr_hybrid_models" / "CNN_LSTM+GC.pth",
            "CNN_BiLSTM+GC": self.models_base_path / "chromecrispr_hybrid_models" / "CNN_BiLSTM+GC.pth",
        }

        results: List[ValidationResult] = []
        smoke_passed = 0
        total_count = len(model_files)

        for model_name, model_path in model_files.items():
            print(f"\nValidating {model_name}...")
            print(f"  File: {model_path}")
            result = self.validate_model(model_path, model_name)
            results.append(result)

            if result.smoke_passed:
                print("  SMOKE PASS")
                smoke_passed += 1
            else:
                print("  SMOKE FAIL")
                if result.error_message:
                    print(f"  Error: {result.error_message}")

            if result.note:
                print(f"  Note: {result.note}")

            print("  Metrics:")
            print(f"    Spearman: {result.spearman_correlation:.4f}")
            print(f"    MSE: {result.mse:.4f}")
            print(f"    Pearson: {result.pearson_correlation:.4f}")
            print(f"    Heuristic spec match: {result.heuristic_spec_match}")

        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total Models: {total_count}")
        print(f"Smoke Passed: {smoke_passed}")
        print(f"Smoke Failed: {total_count - smoke_passed}")
        print(f"Smoke Pass Rate: {(smoke_passed / total_count) * 100:.1f}%")

        self.results = results
        return results

    def save_validation_report(self, output_path: str = "chromecrispr_validation_report.json") -> None:
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_type": "checkpoint_compatibility_smoke_test",
            "pass_criterion": "artifact loads and executes on synthetic inputs using repo-local code",
            "heuristic_metric_note": "Synthetic metric deltas versus published benchmark values are recorded only as a side signal and are not the main smoke-pass criterion.",
            "total_models": len(self.results),
            "smoke_passed_models": sum(1 for r in self.results if r.smoke_passed),
            "smoke_failed_models": sum(1 for r in self.results if not r.smoke_passed),
            "heuristic_spec_matches": sum(1 for r in self.results if r.heuristic_spec_match is True),
            "heuristic_spec_mismatches": sum(1 for r in self.results if r.heuristic_spec_match is False),
            "models_with_load_warnings": sum(1 for r in self.results if r.missing_key_count or r.unexpected_key_count),
            "tolerance": self.tolerance,
            "results": [asdict(result) for result in self.results],
        }

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)

        print(f"\nValidation report saved to: {output_path}")


if __name__ == "__main__":
    print("ChromeCRISPR Checkpoint Compatibility Validator")
    print("==============================================")

    validator = ChromeCRISPRValidator()
    validator.validate_all_models()
    validator.save_validation_report()

    print("\nValidation completed.")
