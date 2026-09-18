# Artifacts behind the reported results

## `predictions/`

One file per model, 8,341 rows each, aligned to `data/test_data.npz`: the guide sequence,
the model's predicted efficiency, and the measured efficiency. Scoring these vectors
reproduces the Spearman and MSE values in `docs/results.md`.

`MANIFEST.json` records each file's original name, row count and SHA-256.

## `models/`

`CNN_GRU_GC.pth` is the ChromeCRISPR checkpoint. Run forward on the test set it reproduces
`predictions/CNN_GRU+GC.csv` to within 1.5e-07, giving Spearman 0.876257 and MSE 0.009321.

It is a pickled module rather than a bare tensor dictionary, so loading it needs the class
to exist under its original name. `src/models/published.py` does this:

```python
from src.models.published import load_published_chromecrispr, predict
model = load_published_chromecrispr("artifacts/models/CNN_GRU_GC.pth")
scores = predict(model, X_test, gc_test)
```

The network is one convolution of 256 filters with ELU, one GRU layer of 256 units, and the
flattened sequence output concatenated with GC content into a 5,377-feature head
(256/128/64/1) under a sigmoid. 1,816,065 parameters.

## Verifying

```bash
python3 scripts/verify_published_results.py
```

Checks every prediction vector against the published table and the checkpoint against its
own prediction vector. Non-zero exit on any disagreement.
