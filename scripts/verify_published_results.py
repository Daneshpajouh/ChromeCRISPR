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
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The table reports the mean and median of the per-fold score over ten shuffled partitions
# of the test predictions, not a single score over the whole test set.
N_FOLDS = 10
FOLD_SEED = 42
TOLERANCE = 0.00005             # the table is quoted to four decimals, so this is exact
PREDICTION_TOLERANCE = 1e-5     # checkpoint against its own stored vector


def published_table(path):
    """Read the four reported columns for each model out of the results table."""
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
        nums = [re.match(r"([0-9.]+)", c) for c in cells[2:6]]
        if not all(nums):
            continue
        out[name] = tuple(float(n.group(1)) for n in nums)
    return out


def load(path):
    pred, actual = [], []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        pred.append(float(row["Predicted Score"]))
        actual.append(float(row["Actual Score"]))
    return np.array(pred), np.array(actual)


def fold_scores(pred, actual):
    """Mean and median of the per-fold Spearman and MSE, as the table reports them."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    idx = [i for _, i in kf.split(pred)]
    rho = [float(spearmanr(pred[i], actual[i]).correlation) for i in idx]
    mse = [float(mean_squared_error(actual[i], pred[i])) for i in idx]
    return (float(np.mean(rho)), float(np.median(rho)),
            float(np.mean(mse)), float(np.median(mse)))


def main():
    table = published_table(os.path.join(ROOT, "docs", "results.md"))
    pred_dir = os.path.join(ROOT, "artifacts", "predictions")
    failures = []

    labels = ("Spearman mean", "Spearman median", "MSE mean", "MSE median")
    checked, models = 0, 0
    print(f"{'model':16s} " + " ".join(f"{l:>16s}" for l in labels))
    for name in sorted(os.listdir(pred_dir)):
        if not name.endswith(".csv"):
            continue
        model = name[:-4]
        if model not in table:
            failures.append(f"{model}: no row in docs/results.md")
            continue
        want = table[model]
        models += 1
        got = fold_scores(*load(os.path.join(pred_dir, name)))
        cells, ok = [], True
        for w, g in zip(want, got):
            hit = abs(g - w) <= TOLERANCE
            ok &= hit
            checked += 1
            if not hit:
                failures.append(f"{model}: {w} published, {g:.6f} recomputed")
            cells.append(f"{g:.4f}{'' if hit else '!'}")
        print(f"{model:16s} " + " ".join(f"{c:>16s}" for c in cells))

    # The checkpoint must reproduce the vector attributed to it.
    from src.models.published import load_published_chromecrispr, predict
    test = np.load(os.path.join(ROOT, "data", "test_data.npz"))
    X = test["X_test"].astype(np.float32)
    y = test["y_test"].astype(np.float32)
    gc = test["gc_test"].astype(np.float32)

    model = load_published_chromecrispr(os.path.join(ROOT, "artifacts", "models", "CNN_GRU_GC.pth"))
    got = predict(model, X, gc)
    stored, _ = load(os.path.join(pred_dir, "CNN_GRU+GC.csv"))
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
    print(f"\nall {checked} published figures reproduced exactly from artifacts in this "
          f"repository ({models} models x {len(labels)} reported columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
