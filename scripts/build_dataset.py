#!/usr/bin/env python3
"""Build the encoded arrays and the train/test split from data/deephf.csv.

The split holds out 15% for testing and is fixed by a seed, so it is identical on every run
and on every machine.
"""
import argparse, os

import numpy as np

BASES = "ACGT"
SEQ_LEN = 21
TEST_FRACTION = 0.15
SPLIT_SEED = 42


def read_csv(path):
    import csv
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if len(r["21mer"]) == SEQ_LEN and r["Wt_Efficiency"] not in ("", "NA")]
    return rows


def encode(sequences):
    """One-hot encode, giving (n, 21, 4)."""
    index = {b: i for i, b in enumerate(BASES)}
    out = np.zeros((len(sequences), SEQ_LEN, len(BASES)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            out[i, j, index[base]] = 1.0
    return out


def gc_content(sequences):
    """The proportion of G and C in each sequence."""
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in sequences],
                    dtype=np.float32)


def split(n, seed=SPLIT_SEED, test_fraction=TEST_FRACTION):
    rng = np.random.RandomState(seed)
    n_test = int(np.ceil(test_fraction * n))
    order = rng.permutation(n)
    return order[n_test:], order[:n_test]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/deephf.csv")
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()

    rows = read_csv(a.csv)
    sequences = [r["21mer"] for r in rows]
    X = encode(sequences)
    g = gc_content(sequences)
    y = np.array([float(r["Wt_Efficiency"]) for r in rows], dtype=np.float32)

    train_idx, test_idx = split(len(rows))
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.out_dir, "train_data.npz"),
                        X_train=X[train_idx], y_train=y[train_idx], gc_train=g[train_idx])
    np.savez_compressed(os.path.join(a.out_dir, "test_data.npz"),
                        X_test=X[test_idx], y_test=y[test_idx], gc_test=g[test_idx])
    print(f"{len(rows)} sgRNAs -> {len(train_idx)} train, {len(test_idx)} test")


if __name__ == "__main__":
    main()
