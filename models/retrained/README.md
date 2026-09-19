# Models trained in the architecture the article describes

One checkpoint per neural model, built from `src/models/architectures.py`, which implements the
article's Methods exactly: two or three specialized layers of 128, the stated kernel, stride and
padding, and the 257-feature fusion. `tests/test_article_specification.py` holds that
correspondence to 51 assertions.

Trained under the same protocol as `models/published_family/`: 40 trials over the wide search
space, 250 epochs, patience 30, eight seeds, everything selected on validation Spearman, a refit
on the training and validation rows for the selected epoch count, and the test set read once.
Scored fold by fold like `docs/results.md`. Predictions are in
`artifacts/article_arch_predictions/`.

## What this set is for

It answers a single question: how far does the architecture in the Methods go, given the best
training this repository can give it.

Six of the nineteen reach or exceed their published value, against ten for the set in
`models/published_family/`, which is built in the architecture of the checkpoint that produced
the published predictions. Every model here scores below its counterpart there, and the mean
Spearman is 0.8502 against 0.8630, a difference of 0.0128.

The two sets differ only in architecture; the search space, budget, seed count, selection rule
and refit are identical. `CNN_GRU+GC` is 833,729 parameters here and 1,816,065 there.
