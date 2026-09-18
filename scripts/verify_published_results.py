#!/usr/bin/env python3
"""Recompute every published figure from the artifacts in this repository.

Two independent checks, both run against `data/test_data.npz`:

1. Each stored prediction vector in `artifacts/predictions/` is scored, and the result is
   compared with the value `docs/results.md` reports for that model. The table is parsed
   rather than restated, so there is one source of truth for what was published.
2. The ChromeCRISPR checkpoint in `artifacts/models/` is run forward and required to
   reproduce its own stored prediction vector, and through it the headline figures.

Exits non-zero if any check fails.
"""
import csv, json, os, re, sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SPEARMAN_TOLERANCE = 0.001      # the table is quoted to four decimals
MSE_TOLERANCE = 0.001
PREDICTION_TOLERANCE = 1e-5     # checkpoint against its own stored vector


def published_table(path):
    """Read the model rows out of the results table."""
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        name = cells[1]
        if not name or name in {"model", "---"} or set(name) <= {"-"}:
            continue
        nums = [re.match(r"([0-9.]+)", c) for c in (cells[2], cells[4])]
        if not all(nums):
            continue
        out[name] = (float(nums[0].group(1)), float(nums[1].group(1)))
    return out


def score(path):
    pred, actual = [], []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        pred.append(float(row["Predicted Score"]))
        actual.append(float(row["Actual Score"]))
    pred, actual = np.array(pred), np.array(actual)
    return pred, actual, float(spearmanr(pred, actual).correlation), \
        float(mean_squared_error(actual, pred))


def main():
    table = published_table(os.path.join(ROOT, "docs", "results.md"))
    pred_dir = os.path.join(ROOT, "artifacts", "predictions")
    failures = []

    print(f"{'model':16s} {'published':>10s} {'recomputed':>11s} {'delta':>10s}   MSE")
    for name in sorted(os.listdir(pred_dir)):
        if not name.endswith(".csv"):
            continue
        model = name[:-4]
        if model not in table:
            failures.append(f"{model}: no row in docs/results.md")
            continue
        want_rho, want_mse = table[model]
        _, _, rho, mse = score(os.path.join(pred_dir, name))
        drho, dmse = rho - want_rho, mse - want_mse
        ok = abs(drho) <= SPEARMAN_TOLERANCE and abs(dmse) <= MSE_TOLERANCE
        print(f"{model:16s} {want_rho:10.4f} {rho:11.6f} {drho:+10.6f}   "
              f"{want_mse:.4f} vs {mse:.6f} {'' if ok else '  FAIL'}")
        if not ok:
            failures.append(f"{model}: rho {rho:.6f} vs {want_rho}, mse {mse:.6f} vs {want_mse}")

    # The checkpoint must reproduce the vector attributed to it.
    from src.models.published import load_published_chromecrispr, predict
    test = np.load(os.path.join(ROOT, "data", "test_data.npz"))
    X = test["X_test"].astype(np.float32)
    y = test["y_test"].astype(np.float32)
    gc = test["gc_test"].astype(np.float32)

    model = load_published_chromecrispr(os.path.join(ROOT, "artifacts", "models", "CNN_GRU_GC.pth"))
    got = predict(model, X, gc)
    stored, _, _, _ = score(os.path.join(pred_dir, "CNN_GRU+GC.csv"))
    drift = float(np.abs(got - stored).max())
    rho = float(spearmanr(got, y).correlation)
    mse = float(mean_squared_error(y, got))
    print(f"\ncheckpoint vs its stored predictions: max|diff| = {drift:.3e}")
    print(f"checkpoint scores: spearman {rho:.6f}, mse {mse:.6f}")
    if drift > PREDICTION_TOLERANCE:
        failures.append(f"checkpoint drifts from its stored predictions by {drift:.3e}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\nall {len(table)} published figures reproduced from artifacts in this repository")
    return 0


if __name__ == "__main__":
    sys.exit(main())
