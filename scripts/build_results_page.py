#!/usr/bin/env python3
"""Write docs/results.md from the model records, plus the retrained checkpoints if present."""
import argparse, json, os, glob

GROUPS = [
    ("Baseline", ["RF"]),
    ("Base models", ["CNN", "GRU", "LSTM", "BiLSTM"]),
    ("Base models with GC content", ["CNN+GC", "GRU+GC", "LSTM+GC", "BiLSTM+GC"]),
    ("Deep models", ["deepCNN", "deepGRU", "deepLSTM", "deepBiLSTM"]),
    ("Deep models with GC content",
     ["deepCNN+GC", "deepGRU+GC", "deepLSTM+GC", "deepBiLSTM+GC"]),
    ("ChromeCRISPR hybrids", ["CNN_LSTM+GC", "CNN_BiLSTM+GC", "CNN_GRU+GC"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="docs/hyperparameters")
    ap.add_argument("--retrained", default="docs/training_results.json")
    ap.add_argument("--out", default="docs/results.md")
    a = ap.parse_args()

    recs = {json.load(open(f))["model_name"]: json.load(open(f))
            for f in glob.glob(os.path.join(a.records, "*_hyperparameters.json"))}
    retrained = json.load(open(a.retrained)) if os.path.exists(a.retrained) else {}

    L = ["# Results", "",
         "Measured on a held-out test set of 8,341 sgRNAs, with means and medians across the",
         "cross-validation folds. Standard deviations are given for the hybrid models.", ""]
    head = "| group | model | Spearman (mean) | Spearman (median) | MSE (mean) | MSE (median) |"
    sep = "|---|---|---|---|---|---|"
    if retrained:
        head += " retrained Spearman | retrained MSE |"
        sep += "---|---|"
    L += [head, sep]
    for group, names in GROUPS:
        for i, name in enumerate(names):
            p = recs[name]["performance"]
            printed = p["as_printed"]
            sd = f" ± {p['spearman_stdev']}" if "spearman_stdev" in p else ""
            msd = f" ± {p['mean_squared_error_stdev']}" if "mean_squared_error_stdev" in p else ""
            row = (f"| {group if i == 0 else ''} | {name} | "
                   f"{printed['spearman_correlation']}{sd} | "
                   f"{printed['spearman_correlation_median']} | "
                   f"{printed['mean_squared_error']}{msd} | "
                   f"{printed['mean_squared_error_median']} |")
            if retrained:
                r = retrained.get(name)
                row += (f" {r['spearman_correlation']:.4f} | {r['mean_squared_error']:.4f} |"
                        if r else " | |")
            L.append(row)

    L += ["", "The best performing model is `CNN_GRU+GC`, at a Spearman correlation of 0.8760",
          "and a mean squared error of 0.0093.", ""]
    if retrained:
        L += ["## Retrained checkpoints", "",
              "`models/` holds a checkpoint per model, retrained with the pipeline in this",
              "repository: hyperparameters searched on a validation split drawn from the",
              "training portion, then the held-out set read once. Their scores are the last two",
              "columns above. Retraining does not reproduce a published figure exactly, because",
              "the weights come from a fresh optimisation; the run is seeded, so these",
              "particular checkpoints regenerate exactly.", "",
              "    python3 scripts/train_all_models.py --data-dir data --out-dir models", ""]
    L += ["Per-model records, including each architecture and its configurable settings, are in",
          "`docs/hyperparameters/`, written by `scripts/build_model_records.py`.", ""]
    open(a.out, "w").write("\n".join(L))
    print(f"wrote {a.out}" + (f" with {len(retrained)} retrained models" if retrained else ""))


if __name__ == "__main__":
    main()
