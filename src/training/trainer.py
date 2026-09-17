import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import optuna
from typing import Dict, List, Tuple, Any
import logging
import os
from datetime import datetime

TRAINING_SETTINGS = frozenset({"batch_size", "learning_rate", "epochs", "patience",
                               "weight_decay", "optimizer"})

GC_BASES = (1, 2)   # C and G in the one-hot alphabet


def gc_content(sequences: torch.Tensor) -> torch.Tensor:
    """GC content as  defines it: the proportion of G and C in the sequence."""
    if sequences.dim() == 3:                       # one-hot, (batch, length, 4)
        sequences = sequences.argmax(dim=2)
    mask = torch.zeros_like(sequences, dtype=torch.float32)
    for base in GC_BASES:
        mask += (sequences == base).float()
    return mask.mean(dim=1)


def _model_kwargs(model_class, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Split model construction arguments from training-loop settings."""
    import inspect

    try:
        accepted = inspect.signature(model_class).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        accepted = None
    return {k: v for k, v in hyperparams.items()
            if k not in TRAINING_SETTINGS and (accepted is None or k in accepted)}


def _forward(model: nn.Module, data: torch.Tensor, gc: torch.Tensor) -> torch.Tensor:
    """Models that take GC content as a further input receive it; the others do not."""
    if getattr(model, "use_gc_content", False):
        return model(data, gc)
    return model(data)


class ChromeCRISPRTrainer:
    """
    Training pipeline for ChromeCRISPR models.

    Implements:
    - 5-fold cross-validation
    - Hyperparameter optimization with Optuna
    - Model evaluation with MSE and Spearman correlation
    - Early stopping and model checkpointing
    """

    def __init__(self, model_class, model_params: Dict, device: str = None):
        self.model_class = model_class
        self.model_params = model_params
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model = None
        self.best_score = -np.inf

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_data_loaders(self, sequences: np.ndarray, targets: np.ndarray,
                           batch_size: int = 64, train_idx: List[int] = None,
                           val_idx: List[int] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders.

        Args:
            sequences: Encoded sequences
            targets: Target values
            batch_size: Batch size for training
            train_idx: Training indices
            val_idx: Validation indices

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if train_idx is not None and val_idx is not None:
            X_train = torch.LongTensor(sequences[train_idx])
            y_train = torch.FloatTensor(targets[train_idx])
            X_val = torch.LongTensor(sequences[val_idx])
            y_val = torch.FloatTensor(targets[val_idx])
        else:
            X_train = torch.LongTensor(sequences)
            y_train = torch.FloatTensor(targets)
            X_val = None
            y_val = None

        # GC content, : the proportion of G and C in the sequence. It is
        # deterministic given the sequence, so the loaders derive it rather than requiring it.
        train_dataset = TensorDataset(X_train, gc_content(X_train), y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            val_dataset = TensorDataset(X_val, gc_content(X_val), y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        return train_loader, val_loader

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """
        Train for one epoch.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, gc, target) in enumerate(train_loader):
            data, gc, target = data.to(self.device), gc.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = _forward(model, data, gc).squeeze(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, model: nn.Module, val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate model.

        Args:
            model: PyTorch model
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (validation_loss, spearman_correlation)
        """
        model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for data, gc, target in val_loader:
                data, gc, target = (data.to(self.device), gc.to(self.device),
                                    target.to(self.device))
                output = _forward(model, data, gc).squeeze(-1)
                loss = criterion(output, target)

                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())

        # Calculate metrics
        val_loss = total_loss / len(val_loader)
        spearman_corr = spearmanr(predictions, targets)[0]

        return val_loss, spearman_corr

    def train_model(self, sequences: np.ndarray, targets: np.ndarray,
                   hyperparams: Dict[str, Any], epochs: int = 100,
                   patience: int = 10, train_idx: List[int] = None,
                   val_idx: List[int] = None) -> Dict[str, Any]:
        """
        Train a single model with given hyperparameters.

        Args:
            sequences: Encoded sequences
            targets: Target values
            hyperparams: Hyperparameters dictionary
            epochs: Maximum number of epochs
            patience: Early stopping patience

        Returns:
            Dictionary with training results
        """
        # Create model. Only the arguments the model actually accepts are passed to it;
        # the rest, such as the batch size and learning rate, configure the training loop.
        model = self.model_class(**_model_kwargs(self.model_class, hyperparams)).to(self.device)

        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=hyperparams.get('learning_rate', 0.001))
        criterion = nn.MSELoss()

        # Create data loaders.  selects on a validation split, so one is held out
        # from the training portion when the caller does not supply the indices.
        if train_idx is None or val_idx is None:
            n = len(sequences)
            rng = np.random.default_rng(0)
            order = rng.permutation(n)
            cut = max(1, int(0.2 * n))
            val_idx, train_idx = order[:cut], order[cut:]
        train_loader, val_loader = self.create_data_loaders(
            sequences, targets, batch_size=hyperparams.get('batch_size', 64),
            train_idx=train_idx, val_idx=val_idx,
        )

        # Training loop
        best_val_loss = np.inf
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_spearmans = []

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            train_losses.append(train_loss)

            # Validate
            val_loss, val_spearman = self.validate(model, val_loader, criterion)
            val_losses.append(val_loss)
            val_spearmans.append(val_spearman)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model_temp.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                               f"Val Loss: {val_loss:.4f}, Spearman: {val_spearman:.4f}")

        # Load best model
        model.load_state_dict(torch.load('best_model_temp.pth'))
        os.remove('best_model_temp.pth')

        return {
            'model': model,
            'best_val_loss': best_val_loss,
            'best_spearman': max(val_spearmans),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_spearmans': val_spearmans
        }

    def cross_validate(self, sequences: np.ndarray, targets: np.ndarray,
                      hyperparams: Dict[str, Any], n_folds: int = 5) -> Dict[str, float]:
        """
        Perform k-fold cross-validation.

        Args:
            sequences: Encoded sequences
            targets: Target values
            hyperparams: Hyperparameters dictionary
            n_folds: Number of folds for cross-validation

        Returns:
            Dictionary with cross-validation results
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_scores = []
        fold_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
            self.logger.info(f"Training fold {fold + 1}/{n_folds}")

            # Create data loaders for this fold
            train_loader, val_loader = self.create_data_loaders(
                sequences, targets,
                batch_size=hyperparams.get('batch_size', 64),
                train_idx=train_idx, val_idx=val_idx
            )

            # Train model
            results = self.train_model(
                sequences[train_idx], targets[train_idx], hyperparams
            )

            # Evaluate on validation set
            val_loss, val_spearman = self.validate(results['model'], val_loader, nn.MSELoss())

            fold_scores.append(val_spearman)
            fold_losses.append(val_loss)

            self.logger.info(f"Fold {fold + 1}: Val Loss: {val_loss:.4f}, "
                           f"Spearman: {val_spearman:.4f}")

        return {
            'mean_spearman': np.mean(fold_scores),
            'std_spearman': np.std(fold_scores),
            'mean_loss': np.mean(fold_losses),
            'std_loss': np.std(fold_losses),
            'fold_scores': fold_scores,
            'fold_losses': fold_losses
        }

    def optimize_hyperparameters(self, sequences: np.ndarray, targets: np.ndarray,
                                n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            sequences: Encoded sequences
            targets: Target values
            n_trials: Number of optimization trials

        Returns:
            Dictionary with best hyperparameters and results
        """
        def objective(trial):
            # Define hyperparameter search space
            hyperparams = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
                'num_layers': trial.suggest_int('num_layers', 1, 3),
                'embedding_dim': trial.suggest_categorical('embedding_dim', [64, 128, 256])
            }

            # Update model parameters
            trial_params = self.model_params.copy()
            trial_params.update(hyperparams)

            # Perform cross-validation
            cv_results = self.cross_validate(sequences, targets, trial_params)

            return cv_results['mean_spearman']

        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Get best parameters
        best_params = self.model_params.copy()
        best_params.update(study.best_params)

        self.logger.info(f"Best hyperparameters: {study.best_params}")
        self.logger.info(f"Best CV score: {study.best_value:.4f}")

        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'study': study
        }

    def train_final_model(self, sequences: np.ndarray, targets: np.ndarray,
                         hyperparams: Dict[str, Any], save_path: str = None) -> nn.Module:
        """
        Train final model on full dataset with best hyperparameters.

        Args:
            sequences: Encoded sequences
            targets: Target values
            hyperparams: Best hyperparameters
            save_path: Path to save the model

        Returns:
            Trained model
        """
        self.logger.info("Training final model on full dataset")

        # Train on full dataset
        results = self.train_model(sequences, targets, hyperparams, epochs=200)

        # Save model
        if save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{save_path}_{timestamp}.pth"
            torch.save(results['model'].state_dict(), save_path)
            self.logger.info(f"Model saved to {save_path}")

        return results['model']

if __name__ == "__main__":
    # A short demonstration on random data. `create_cnn_gru_model` builds the CNN_GRU+GC
    # hybrid, which takes GC content as a further input; the trainer derives it.
    from src.models.hybrid_models import create_cnn_gru_model

    trainer = ChromeCRISPRTrainer(create_cnn_gru_model, {})

    rng = np.random.default_rng(0)
    sequences = rng.integers(0, 4, (512, 21))
    targets = rng.random(512).astype(np.float32)

    results = trainer.train_model(sequences, targets, {"batch_size": 64, "learning_rate": 1e-3},
                                  epochs=2)
    print(f"best validation loss: {results['best_val_loss']:.4f}")
