#!/usr/bin/env python3
"""
ChromeCRISPR checkpoint compatibility validator.

This script is a lightweight smoke-test utility around the published model files.
It attempts to reconstruct model classes, load saved weights with compatibility
fallbacks, and run synthetic forward-pass checks. It does not reproduce the
original manuscript benchmark metrics from the DeepHF dataset and should not be
interpreted as a full experimental rerun.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Import ChromeCRISPR modules
try:
    from models.cnn_model import create_cnn_model
    from models.hybrid_models import create_cnn_gru_model, create_cnn_lstm_model, create_cnn_bilstm_model
    from models.rnn_models import create_gru_model, create_lstm_model, create_bilstm_model
    from evaluation.metrics import ChromeCRISPRMetrics
except ImportError:
    # Fallback for when running as standalone script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.cnn_model import create_cnn_model
    from models.hybrid_models import create_cnn_gru_model, create_cnn_lstm_model, create_cnn_bilstm_model
    from models.rnn_models import create_gru_model, create_lstm_model, create_bilstm_model
    from evaluation.metrics import ChromeCRISPRMetrics

@dataclass
class ValidationResult:
    """Data class for compatibility-oriented validation results."""
    model_name: str
    spearman_correlation: float
    mse: float
    rmse: float
    mae: float
    r2: float
    pearson_correlation: float
    validation_passed: bool
    error_message: Optional[str] = None

class ChromeCRISPRValidator:
    """
    Compatibility-oriented validator for the published ChromeCRISPR checkpoints.

    The validation path uses synthetic inputs so the repo can smoke-test weight
    loading and forward passes without shipping the full original training data.
    """

    def __init__(self, models_base_path: str = "../models"):
        self.models_base_path = Path(models_base_path)
        self.metrics_calculator = ChromeCRISPRMetrics()

        # Published reference values retained for compatibility reporting only.
        self.performance_specs = {
            'CNN_GRU+GC': {'spearman': 0.8760, 'mse': 0.0093},
            'deepCNN+GC': {'spearman': 0.8730, 'mse': 0.0093},
            'CNN_BiLSTM+GC': {'spearman': 0.8700, 'mse': 0.0096},
            'CNN_LSTM+GC': {'spearman': 0.8670, 'mse': 0.0115},
            'deepBiLSTM+GC': {'spearman': 0.8670, 'mse': 0.0098},
            'deepGRU+GC': {'spearman': 0.8670, 'mse': 0.0098},
            'deepGRU': {'spearman': 0.8680, 'mse': 0.0099},
            'deepCNN': {'spearman': 0.8690, 'mse': 0.0098},
            'deepLSTM': {'spearman': 0.8620, 'mse': 0.0103},
            'deepBiLSTM': {'spearman': 0.8620, 'mse': 0.0104},
            'deepLSTM+GC': {'spearman': 0.8600, 'mse': 0.0104},
            'LSTM+GC': {'spearman': 0.8560, 'mse': 0.0112},
            'BiLSTM+GC': {'spearman': 0.8550, 'mse': 0.0110},
            'BiLSTM': {'spearman': 0.8430, 'mse': 0.0120},
            'GRU+GC': {'spearman': 0.8400, 'mse': 0.0122},
            'GRU': {'spearman': 0.8370, 'mse': 0.0121},
            'LSTM': {'spearman': 0.8370, 'mse': 0.0122},
            'CNN': {'spearman': 0.7930, 'mse': 0.0161},
            'CNN+GC': {'spearman': 0.7810, 'mse': 0.0170},
            'RF': {'spearman': 0.7890, 'mse': 0.0161}
        }

        self.tolerance = 0.001
        self.results: List[ValidationResult] = []

    def load_model_from_file(self, model_path: Path, model_type: str) -> Optional[nn.Module]:
        """
        Load a PyTorch model from file with appropriate architecture.

        Args:
            model_path: Path to the model file
            model_type: Type of model (CNN, GRU, LSTM, etc.)

        Returns:
            Loaded PyTorch model or None if loading fails
        """
        try:
            if not model_path.exists():
                return None

            # Create model architecture based on type
            model = None

            if 'CNN_GRU' in str(model_path):
                from models.hybrid_models import create_cnn_gru_model
                model = create_cnn_gru_model()
            elif 'CNN_LSTM' in str(model_path):
                from models.hybrid_models import create_cnn_lstm_model
                model = create_cnn_lstm_model()
            elif 'CNN_BiLSTM' in str(model_path):
                from models.hybrid_models import create_cnn_bilstm_model
                model = create_cnn_bilstm_model()
            elif 'GRU' in str(model_path) and 'CNN' not in str(model_path):
                from models.rnn_models import create_gru_model
                if 'deep' in str(model_path):
                    model = create_gru_model(num_layers=4)
                else:
                    model = create_gru_model(num_layers=2)
            elif 'LSTM' in str(model_path) and 'CNN' not in str(model_path) and 'BiLSTM' not in str(model_path):
                from models.rnn_models import create_lstm_model
                if 'deep' in str(model_path):
                    model = create_lstm_model(num_layers=4)
                else:
                    model = create_lstm_model(num_layers=2)
            elif 'BiLSTM' in str(model_path) and 'CNN' not in str(model_path):
                from models.rnn_models import create_bilstm_model
                model = create_bilstm_model(num_layers=2)
            elif 'CNN' in str(model_path) and 'GRU' not in str(model_path) and 'LSTM' not in str(model_path):
                from models.cnn_model import create_cnn_model, create_deep_cnn_model
                if 'deep' in str(model_path):
                    model = create_deep_cnn_model()
                else:
                    model = create_cnn_model()
            elif 'RF' in str(model_path):
                print(f"    Skipping {model_type}: Random Forest (not PyTorch model)")
                return None
            else:
                from models.cnn_model import create_cnn_model
                model = create_cnn_model()

            if model is None:
                return None

            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            result = model.load_state_dict(state_dict, strict=False)

            if result.missing_keys or result.unexpected_keys:
                print(f"    Model loaded with compatibility issues:")
                if result.missing_keys:
                    print(f"     Missing keys: {len(result.missing_keys)}")
                if result.unexpected_keys:
                    print(f"     Unexpected keys: {len(result.unexpected_keys)}")
                print("     Model may have architecture differences from saved version")

            model.eval()
            return model

        except Exception as e:
            print(f"   Error loading model {model_path}: {e}")
            return None

    def generate_test_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic inputs for checkpoint smoke testing."""
        np.random.seed(42)
        sequences = np.random.randint(0, 4, (n_samples, 21))
        targets = np.random.random(n_samples)
        return sequences, targets

    def validate_model(self, model_path: Path, model_name: str) -> ValidationResult:
        """
        Validate a single model against compatibility expectations.

        Args:
            model_path: Path to the model file
            model_name: Name of the model for reporting

        Returns:
            ValidationResult with smoke-test metrics and pass/fail status
        """
        try:
            model = self.load_model_from_file(model_path, model_name)
            if model is None:
                return ValidationResult(
                    model_name=model_name,
                    spearman_correlation=0.0,
                    mse=0.0,
                    rmse=0.0,
                    mae=0.0,
                    r2=0.0,
                    pearson_correlation=0.0,
                    validation_passed=False,
                    error_message=f"Failed to load model from {model_path}"
                )

            sequences, y_true = self.generate_test_data()

            model.eval()
            with torch.no_grad():
                inputs = torch.LongTensor(sequences)

                if 'CNN_GRU' in str(model_path) or 'CNN_LSTM' in str(model_path) or 'CNN_BiLSTM' in str(model_path):
                    gc_content = torch.rand(inputs.size(0), 1)
                    predictions = model(inputs, gc_content).numpy().flatten()
                else:
                    predictions = model(inputs).numpy().flatten()

            metrics = self.metrics_calculator.calculate_metrics(y_true, predictions, model_name)

            specs = self.performance_specs.get(model_name, {})
            expected_spearman = specs.get('spearman', 0.0)
            expected_mse = specs.get('mse', 0.0)

            spearman_diff = abs(metrics['spearman_corr'] - expected_spearman)
            mse_diff = abs(metrics['mse'] - expected_mse)
            validation_passed = (spearman_diff <= self.tolerance) and (mse_diff <= self.tolerance)

            return ValidationResult(
                model_name=model_name,
                spearman_correlation=metrics['spearman_corr'],
                mse=metrics['mse'],
                rmse=metrics['rmse'],
                mae=metrics['mae'],
                r2=metrics['r2'],
                pearson_correlation=metrics['pearson_corr'],
                validation_passed=validation_passed,
                error_message=None,
            )

        except Exception as e:
            return ValidationResult(
                model_name=model_name,
                spearman_correlation=0.0,
                mse=0.0,
                rmse=0.0,
                mae=0.0,
                r2=0.0,
                pearson_correlation=0.0,
                validation_passed=False,
                error_message=str(e),
            )

    def validate_all_models(self) -> List[ValidationResult]:
        """
        Validate all 20 published models in the repository.

        Returns:
            List of ValidationResult objects for all models
        """
        print("Starting checkpoint compatibility validation for all 20 ChromeCRISPR models")
        print("=" * 70)

        model_files = {
            'CNN': self.models_base_path / "base_models" / "CNN.pth",
            'GRU': self.models_base_path / "base_models" / "GRU.pth",
            'LSTM': self.models_base_path / "base_models" / "LSTM.pth",
            'BiLSTM': self.models_base_path / "base_models" / "BiLSTM.pth",
            'RF': self.models_base_path / "base_models" / "RF.joblib",
            'CNN+GC': self.models_base_path / "base_models_with_gc" / "CNN+GC.pth",
            'GRU+GC': self.models_base_path / "base_models_with_gc" / "GRU+GC.pth",
            'LSTM+GC': self.models_base_path / "base_models_with_gc" / "LSTM+GC.pth",
            'BiLSTM+GC': self.models_base_path / "base_models_with_gc" / "BiLSTM+GC.pth",
            'deepCNN': self.models_base_path / "deep_models" / "deepCNN.pth",
            'deepGRU': self.models_base_path / "deep_models" / "deepGRU.pth",
            'deepLSTM': self.models_base_path / "deep_models" / "deepLSTM.pth",
            'deepBiLSTM': self.models_base_path / "deep_models" / "deepBiLSTM.pth",
            'deepCNN+GC': self.models_base_path / "deep_models_with_gc" / "deepCNN+GC.pth",
            'deepGRU+GC': self.models_base_path / "deep_models_with_gc" / "deepGRU+GC.pth",
            'deepLSTM+GC': self.models_base_path / "deep_models_with_gc" / "deepLSTM+GC.pth",
            'deepBiLSTM+GC': self.models_base_path / "deep_models_with_gc" / "deepBiLSTM+GC.pth",
            'CNN_GRU+GC': self.models_base_path / "chromecrispr_hybrid_models" / "CNN_GRU+GC.pth",
            'CNN_LSTM+GC': self.models_base_path / "chromecrispr_hybrid_models" / "CNN_LSTM+GC.pth",
            'CNN_BiLSTM+GC': self.models_base_path / "chromecrispr_hybrid_models" / "CNN_BiLSTM+GC.pth",
        }

        results: List[ValidationResult] = []
        passed_count = 0
        total_count = len(model_files)

        for model_name, model_path in model_files.items():
            print(f"\nValidating {model_name}...")
            print(f"  File: {model_path}")

            result = self.validate_model(model_path, model_name)
            results.append(result)

            if result.validation_passed:
                print("  PASSED")
                passed_count += 1
            else:
                print("  FAILED")
                if result.error_message:
                    print(f"  Error: {result.error_message}")

            print("  Metrics:")
            print(f"    Spearman: {result.spearman_correlation:.4f}")
            print(f"    MSE: {result.mse:.4f}")
            print(f"    Pearson: {result.pearson_correlation:.4f}")

        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total Models: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")
        print(f"Pass Rate: {(passed_count / total_count) * 100:.1f}%")

        if passed_count == total_count:
            print("All models passed checkpoint compatibility validation.")
        else:
            print("Some models failed compatibility validation. Check the individual results above.")

        self.results = results
        return results

    def save_validation_report(self, output_path: str = "chromecrispr_validation_report.json") -> None:
        """Save validation results to a JSON file."""
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_type': 'checkpoint_compatibility_smoke_test',
            'total_models': len(self.results),
            'passed_models': sum(1 for r in self.results if r.validation_passed),
            'failed_models': sum(1 for r in self.results if not r.validation_passed),
            'tolerance': self.tolerance,
            'results': [asdict(result) for result in self.results],
        }

        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, default=str)

        print(f"\nValidation report saved to: {output_path}")


if __name__ == "__main__":
    print("ChromeCRISPR Checkpoint Compatibility Validator")
    print("==============================================")

    validator = ChromeCRISPRValidator()
    validator.validate_all_models()
    validator.save_validation_report()

    print("\nValidation completed.")
