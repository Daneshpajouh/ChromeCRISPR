"""The dataset build is reproducible and self-consistent."""

import pathlib
import subprocess
import tempfile

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def test_source_and_derived_files_are_present():
    assert (DATA / "deephf.csv").exists()
    assert (DATA / "train_data.npz").exists()
    assert (DATA / "test_data.npz").exists()


def test_split_sizes():
    train = np.load(DATA / "train_data.npz")
    test = np.load(DATA / "test_data.npz")
    assert len(train["y_train"]) == 47263
    assert len(test["y_test"]) == 8341
    assert len(train["y_train"]) + len(test["y_test"]) == 55604


def test_shapes_and_encoding():
    test = np.load(DATA / "test_data.npz")
    X = test["X_test"]
    assert X.shape == (8341, 21, 4)
    assert np.all(X.sum(axis=2) == 1), "each position is a single one-hot base"


def test_gc_content_is_consistent_with_the_encoding():
    test = np.load(DATA / "test_data.npz")
    X, gc = test["X_test"], test["gc_test"]
    from_encoding = X[:, :, 1:3].sum(axis=(1, 2)) / X.shape[1]     # C and G
    assert np.allclose(from_encoding, gc, atol=1e-6)


def test_train_and_test_do_not_overlap():
    train = np.load(DATA / "train_data.npz")
    test = np.load(DATA / "test_data.npz")
    as_rows = lambda a: {a[i].tobytes() for i in range(len(a))}
    assert not (as_rows(train["X_train"]) & as_rows(test["X_test"])), \
        "no sequence appears in both partitions"


def test_build_is_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["python3", "scripts/build_dataset.py",
                        "--csv", str(DATA / "deephf.csv"), "--out-dir", tmp],
                       cwd=ROOT, check=True, capture_output=True)
        for name in ("train_data.npz", "test_data.npz"):
            fresh, shipped = np.load(pathlib.Path(tmp) / name), np.load(DATA / name)
            for key in shipped.files:
                assert np.array_equal(fresh[key], shipped[key]), f"{name}:{key} differs"
