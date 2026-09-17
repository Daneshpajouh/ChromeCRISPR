"""The per-model records describe the models and carry their results."""

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECORDS = sorted((ROOT / "docs" / "hyperparameters").glob("*_hyperparameters.json"))

EXPECTED_MODELS = {
    "RF", "CNN", "GRU", "LSTM", "BiLSTM",
    "CNN+GC", "GRU+GC", "LSTM+GC", "BiLSTM+GC",
    "deepCNN", "deepGRU", "deepLSTM", "deepBiLSTM",
    "deepCNN+GC", "deepGRU+GC", "deepLSTM+GC", "deepBiLSTM+GC",
    "CNN_GRU+GC", "CNN_LSTM+GC", "CNN_BiLSTM+GC",
}


def test_every_model_has_a_record():
    assert {json.loads(p.read_text())["model_name"] for p in RECORDS} == EXPECTED_MODELS


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_shape(path):
    rec = json.loads(path.read_text())
    perf = rec["performance"]
    for key in ("spearman_correlation", "spearman_correlation_median",
                "mean_squared_error", "mean_squared_error_median"):
        assert isinstance(perf[key], float), f"{key} must be numeric, not a display string"
    assert perf["test_set_size"] == 8341
    assert 0.0 < perf["spearman_correlation"] < 1.0
    assert 0.0 < perf["mean_squared_error"] < 1.0
    assert rec["architecture"], "each record must describe its architecture"
    assert rec["configurable"]["items"], "each record must list its configurable settings"


def test_best_model_results():
    perf = json.loads(
        (ROOT / "docs/hyperparameters/CNN_GRU+GC_hyperparameters.json").read_text()
    )["performance"]
    assert perf["spearman_correlation"] == pytest.approx(0.8760)
    assert perf["mean_squared_error"] == pytest.approx(0.0093)
    assert perf["spearman_stdev"] == pytest.approx(0.008)
    assert perf["mean_squared_error_stdev"] == pytest.approx(0.0006)


def test_records_are_reproducible():
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["python3", "scripts/build_model_records.py", "--out-dir", tmp],
                       cwd=ROOT, check=True, capture_output=True)
        for path in RECORDS:
            assert (pathlib.Path(tmp) / path.name).read_text() == path.read_text(), \
                f"{path.name} differs from a fresh generation"
