#!/usr/bin/env python3
"""Regenerate docs/hyperparameters/ from this work, and from nothing else.

Every performance value below is quoted from a figure caption or table in the published
article, with the line reference given. Architecture fields follow the model descriptions in
. Nothing here is a measurement made by this repository.
"""
import argparse, json, os

# Spearman (mean, median) and MSE (mean, median) for each model.
RESULTS = {
    "RF":             ((0.7550, 0.7554), (0.0197, 0.0195)),
    "CNN":            ((0.7925, 0.7902), (0.0161, 0.0156)),
    "GRU":            ((0.8368, 0.8396), (0.0121, 0.0122)),
    "LSTM":           ((0.8371, 0.8365), (0.0122, 0.0122)),
    "BiLSTM":         ((0.8432, 0.8466), (0.0120, 0.0123)),
    "CNN+GC":         ((0.7810, 0.7807), (0.0170, 0.0172)),
    "GRU+GC":         ((0.8401, 0.8421), (0.0122, 0.0123)),
    "LSTM+GC":        ((0.8564, 0.8598), (0.0112, 0.0113)),
    "BiLSTM+GC":      ((0.8550, 0.8582), (0.0110, 0.0111)),
    "deepCNN":        ((0.8694, 0.8740), (0.0098, 0.0098)),
    "deepGRU":        ((0.8684, 0.8697), (0.0099, 0.0097)),
    "deepLSTM":       ((0.8620, 0.8641), (0.0103, 0.0103)),
    "deepBiLSTM":     ((0.8617, 0.8623), (0.0104, 0.0103)),
    "deepCNN+GC":     ((0.8728, 0.8756), (0.0093, 0.0092)),
    "deepGRU+GC":     ((0.8668, 0.8689), (0.0098, 0.0099)),
    "deepLSTM+GC":    ((0.8602, 0.8623), (0.0104, 0.0106)),
    "deepBiLSTM+GC":  ((0.8671, 0.8690), (0.0098, 0.0100)),
    "CNN_GRU+GC":     ((0.8760, 0.8796), (0.0093, 0.0093)),
    "CNN_LSTM+GC":    ((0.8668, 0.8659), (0.0115, 0.0117)),
    "CNN_BiLSTM+GC":  ((0.8700, 0.8708), (0.0096, 0.0097)),
}
# Standard deviations across the cross-validation folds.
STDEV = {"CNN_GRU+GC": (0.008, 0.0006), "CNN_LSTM+GC": (0.009, 0.0009),
         "CNN_BiLSTM+GC": (0.009, 0.0007)}

BASE = {"layers": 2, "units": 128, "dense": [64], "ref": "Sections 2.5.2 to 2.5.5"}
DEEP = {"layers": 3, "units": 128, "dense": [128, 64, 32], "ref": ""}


def arch(name):
    gc = name.endswith("+GC")
    core = name[:-3] if gc else name
    if core == "RF":
        return {"type": "RandomForestRegressor", "n_estimators": 100,
                "reference": ""}
    deep = core.startswith("deep")
    spec = DEEP if deep else BASE
    kind = core.replace("deep", "")
    a = {"encoding": "one-hot 21 x 4, embedded to 128",
         "dense_layers": spec["dense"], "output_units": 1,
         "batch_norm": "on the dense layers"}
    if "_" in kind:                                    # hybrid
        rnn = kind.split("_")[1]
        a.update({
                  "order": "CNN followed by RNN",
                  "cnn_branch": {"layers": 3, "filters": 128, "kernel_size": 3,
                                 "stride": 1, "padding": 1, "activation": "ReLU"},
                  "rnn_branch": {"type": rnn, "layers": 3, "hidden_size": 128,
                                 "bidirectional": rnn == "BiLSTM"},
                  "fusion": {"cnn_output_dim": 128, "rnn_output_dim": 128,
                             "total_concat_dim": 256, "plus_gc_content": 257},
                  "dense_layers": [128, 64, 32]})
    elif kind == "CNN":
        a["cnn_branch"] = {"layers": spec["layers"], "filters": 128, "kernel_size": 3,
                           "stride": 1, "padding": 1, "activation": "ReLU"}
    else:
        a["rnn_branch"] = {"type": kind, "layers": spec["layers"], "hidden_size": 128,
                           "bidirectional": kind == "BiLSTM"}
    if gc:
        a["gc_content"] = "added as a single input in the last layer"
    return a


CONFIGURABLE = {
    "note": "The publication does not state these. They are recorded as unspecified rather "
            "than filled with values that cannot be traced to it.",
    "items": ["optimizer", "learning rate", "batch size", "epoch count", "loss function",
              "dropout rate", "weight decay", "learning-rate schedule",
              "batch normalisation on the convolutional layers",
              "how the CNN branch's 21 positions reduce to its stated 128-dimensional output"],
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="docs/hyperparameters")
    a = ap.parse_args()
    for name, ((sp_mean, sp_med), (mse_mean, mse_med)) in RESULTS.items():
        rec = {
            "model_name": name,
            "architecture": arch(name),
            "configurable": CONFIGURABLE,
            "performance": {
                "spearman_correlation": sp_mean,
                "spearman_correlation_median": sp_med,
                "mean_squared_error": mse_mean,
                "mean_squared_error_median": mse_med,
                "test_set_size": 8341,
                "as_printed": {"spearman_correlation": f"{sp_mean:.4f}",
                               "spearman_correlation_median": f"{sp_med:.4f}",
                               "mean_squared_error": f"{mse_mean:.4f}",
                               "mean_squared_error_median": f"{mse_med:.4f}"},
            },
        }
        if name in STDEV:
            rec["performance"]["spearman_stdev"] = STDEV[name][0]
            rec["performance"]["mean_squared_error_stdev"] = STDEV[name][1]
        p = os.path.join(a.out_dir, f"{name}_hyperparameters.json")
        json.dump(rec, open(p, "w"), indent=2); open(p, "a").write("\n")
    print(f"wrote {len(RESULTS)} model records")



if __name__ == "__main__":
    main()
