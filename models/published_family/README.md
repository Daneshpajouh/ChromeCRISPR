# Models trained in the published architecture

One checkpoint per neural model, trained in the architecture the published ChromeCRISPR
checkpoint uses: convolutional and recurrent stacks whose **whole sequence output** is
flattened, concatenated with GC content, and passed through a dense head under a sigmoid.
`src/models/published.py` defines that network for the published checkpoint;
`results.json` records each model's hyperparameters, selected epoch, parameter count and
SHA-256.

Scored the same way as `docs/results.md`: ten shuffled folds of the prediction vector,
reporting the mean and median per fold. Predictions are in
`artifacts/retrained_predictions/`, so every number here is recomputable.

Protocol for all of them: hyperparameters and the epoch count chosen on a validation split
drawn from the training portion and ranked by validation Spearman, then a refit on the
training and validation rows for that epoch count, then the test set read once.

## How they compare

Nine of the nineteen meet or exceed the corresponding published value, and two more sit within
0.0002, which is the width the table is rounded to. The eight below it trail by 0.0012 to
0.0070.

The remaining difference is a property of what the published figures are rather than of these
models. The training log shipped with the published checkpoint records 200 epochs for
`CNN_GRU+GC`: the per-epoch Spearman has a median of 0.8696 and a maximum of 0.8763, and
exactly one of the 200 epochs reaches the reported value. The best model here scores 0.8711,
which is above that run's median epoch. The gap to the reported figure is the distance between
a typical epoch and the best of two hundred, so it narrows with repeated evaluation rather
than with a better model, and five successive search campaigns moved it by less than 0.001 in
total.
