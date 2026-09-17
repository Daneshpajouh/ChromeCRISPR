# Training

## Data

Wild-type SpCas9 activity from the DeepHF dataset. The eSpCas9 and SpCas9-HF data in the same
study are not combined with it, because their activity distributions differ. Each sgRNA is 20
nucleotides plus the variable PAM nucleotide, so 21 in total.

15% of the data is held out for testing. The remaining 85% is used for hyperparameter tuning
and training, with nested 5-fold cross-validation and a Bayesian search. The selected
hyperparameters are validated by 5-fold cross-validation and the model is then retrained on the
full 85%.

## Encoding

One-hot, giving a 21 × 4 matrix, then embedded per position to 128 dimensions. GC content is
the proportion of G and C in the sequence, appended as a single input in the last layer of the
GC models. It is derived from the sequence by the trainer.

## Loop

`src/training/trainer.py` selects on a validation split drawn from the training portion, with
early stopping on the validation loss. `optimize_hyperparameters` runs a Bayesian search with
Optuna over the cross-validation folds.

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
