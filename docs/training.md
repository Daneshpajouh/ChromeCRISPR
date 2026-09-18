# Training

## Data

Wild-type SpCas9 activity from the DeepHF dataset. The eSpCas9 and SpCas9-HF data in the same
study are not combined with it, because their activity distributions differ. Each sgRNA is 20
nucleotides plus the variable PAM nucleotide, so 21 in total.

`scripts/build_dataset.py` rebuilds the encoded arrays and the split from `data/deephf.csv`.
15% is held out for testing under a fixed seed, so the split is identical on every run.

## Encoding

One-hot, giving a 21 x 4 matrix, then embedded per position to 128 dimensions. GC content is
the proportion of G and C in the sequence, appended as a single input in the last layer of the
GC models. It is derived from the sequence by the trainer.

## The run that produced `models/retrained/`

`scripts/train_all_models.py` trains one model at a time and is what produced the checkpoints
in `models/retrained/`.

1. **Search.** A random search over learning rate, batch size, dropout, weight decay and
   schedule, scored on a validation split drawn from the training portion. Trials run at the
   same epoch budget, patience and schedule horizon as the final fit, so a setting is judged
   under the conditions it will train in.
2. **Selection.** Trials and epochs are both ranked by validation Spearman, the same quantity
   the results report.
3. **Refit.** With the hyperparameters and the epoch count fixed, the model is refitted on the
   training and validation rows together for that many epochs, with nothing monitored and
   nothing restored. Selection cannot see the extra rows and the refit has no validation
   signal, so the split still does its job while the fitted model keeps all 47,263 rows.
4. **Test.** The held-out set is read once, after the final model is fitted.

The run is seeded, so the same command reproduces the same weights:

    python3 scripts/train_all_models.py --data-dir data --out-dir models/retrained

`models/retrained/results.json` records each model's selected hyperparameters, its selected
epoch, the number of rows it was fitted on, and the SHA-256 of its checkpoint.

## The Optuna path

`src/training/trainer.py` provides a separate, more general interface: `ChromeCRISPRTrainer`
for a single model and `optimize_hyperparameters` for a Bayesian search with Optuna over
cross-validation folds. It is not what produced `models/retrained/`.

```python
from src.training.trainer import ChromeCRISPRTrainer
from src.models.hybrid_models import create_cnn_gru_model

trainer = ChromeCRISPRTrainer(create_cnn_gru_model, {})
results = trainer.train_model(sequences, targets,
                              {"batch_size": 64, "learning_rate": 1e-3}, epochs=100)
```

Settings that configure the loop rather than the network, such as the batch size and learning
rate, are kept out of the model constructor.

## Evaluation

Spearman correlation via SciPy and mean squared error via scikit-learn, on the held-out test
set. One-way ANOVA followed by Tukey's HSD is used to test for differences between model
groups; see `src/evaluation/metrics.py`.
