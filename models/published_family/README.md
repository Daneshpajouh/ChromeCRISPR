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

Protocol for all of them: hyperparameters, the epoch count and the seed are chosen on a
validation split drawn from the training portion and ranked by validation Spearman, then the
model is refitted on the training and validation rows for that epoch count, then the test set
is read once.

Several search campaigns were run per model, differing in budget and search space. **Which
campaign's model is shipped is decided by its validation score, never by its test score**, and
`results.json` records the campaign chosen and the validation figure that chose it. Taking the
best test result across campaigns would be selecting on the test set; it is also not better
here, giving nine models at or above their published value where validation picking gives ten.

## How they compare

Ten of the twenty meet or exceed the corresponding published value. The rest trail by 0.0007
to 0.0087.

Selecting the seed does not help, and that is itself the useful result. Picking the best of 24
seeds on validation changed the test score by -0.0008 on average across nine models, improving
one and leaving or worsening the other eight. If validation could resolve differences of this
size, best-of-24 would have lifted the test score; it does not, so the remaining gaps sit below
what any honest selection can distinguish.

The remaining difference is a property of what the published figures are rather than of these
models. The training log shipped with the published checkpoint records 200 epochs for
`CNN_GRU+GC`: the per-epoch Spearman has a median of 0.8696 and a maximum of 0.8763, and
exactly one of the 200 epochs reaches the reported value. The best model here scores 0.8711,
which is above that run's median epoch. The gap to the reported figure is the distance between
a typical epoch and the best of two hundred, so it narrows with repeated evaluation rather
than with a better model, and five successive search campaigns moved it by less than 0.001 in
total.

## Random Forest

`RF.joblib` is the documented configuration: `RandomForestRegressor(n_estimators=100)`, which
is the only setting the publication states. It scores Spearman 0.7534 mean / 0.7519 median and
MSE 0.0201 mean / 0.0200 median, against 0.7550 / 0.7554 and 0.0197 / 0.0195 reported.

RF is the one model with no stored prediction vector, so unlike the others there is nothing to
recompute the reported figures from. Sweeping the tree count on the same split shows why the
gap does not close:

| trees | Spearman mean | Spearman median | MSE mean |
|---|---|---|---|
| 100 | 0.7534 | 0.7519 | 0.0201 |
| 200 | 0.7563 | 0.7554 | 0.0200 |
| 500 | 0.7583 | 0.7589 | 0.0199 |
| 1200 | 0.7585 | 0.7595 | 0.0199 |
| reported | 0.7550 | 0.7554 | 0.0197 |

More trees push Spearman past the reported value while the mean squared error settles at
0.0199 and does not move below it, so no tree count reproduces both columns. Richer encodings
move it further away, not closer: adding the reverse complement changes Spearman by 0.0002,
while adding bigrams and trigrams overshoots to 0.7819. The documented setting is therefore
what is shipped, and the difference is recorded rather than tuned away.

## Against the article's own architecture

`models/retrained/` holds the same twenty models built from `src/models/architectures.py`, which
implements the Methods, and trained under an identical protocol: same search space, budget, seed
count, selection rule and refit. Six of those reach their published value against ten here, and
every one scores below its counterpart here. Mean Spearman is 0.8502 there and 0.8630 here.

The two sets differ in nothing but architecture, so that 0.0128 is a measurement of what the
architecture difference is worth on this dataset.

