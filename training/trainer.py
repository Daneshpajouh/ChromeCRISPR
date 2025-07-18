"""
Trainer for ChromeCRISPR models.

This module contains the main training loop and utilities for training
ChromeCRISPR models with proper validation and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..evaluation.metrics import EvaluationMetrics
from ..utils.config import Config


class Trainer:
    """
    Trainer class for ChromeCRISPR models.

    This trainer handles:
    - Training loop with validation
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Training metrics tracking
    - Visualization
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Config] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on ('cpu', 'cuda', or None for auto)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or Config()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training on device: {self.device}")

        # Initialize training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'learning_rate': []
        }

        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()

        # Setup evaluation metrics
        self.evaluator = EvaluationMetrics()

    def _setup_optimizer(self):
        """Setup the optimizer."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam')
        lr = optimizer_config.get('learning_rate', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0)

        if optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.logger.info(f"Using optimizer: {optimizer_name} with lr={lr}")

    def _setup_scheduler(self):
        """Setup the learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'none')

        if scheduler_name.lower() == 'none':
            self.scheduler = None
        elif scheduler_name.lower() == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_name.lower() == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            eta_min = scheduler_config.get('eta_min', 0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
        elif scheduler_name.lower() == 'reduce_on_plateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.5)
            min_lr = scheduler_config.get('min_lr', 1e-6)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=patience,
                factor=factor,
                min_lr=min_lr
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        if self.scheduler:
            self.logger.info(f"Using scheduler: {scheduler_name}")

    def _setup_loss_function(self):
        """Setup the loss function."""
        loss_config = self.config.get('loss', {})
        loss_name = loss_config.get('name', 'mse')

        if loss_name.lower() == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_name.lower() == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_name.lower() == 'huber':
            delta = loss_config.get('delta', 1.0)
            self.criterion = nn.HuberLoss(delta=delta)
        elif loss_name.lower() == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

        self.logger.info(f"Using loss function: {loss_name}")

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (train_loss, train_metric)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(data)
            loss = self.criterion(predictions.squeeze(), targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clipping', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clipping']
                )

            self.optimizer.step()

            # Collect metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.squeeze().cpu().detach().numpy())
            all_targets.extend(targets.cpu().detach().numpy())

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        train_metric = self.evaluator.calculate_spearman_correlation(
            all_targets, all_predictions
        )

        return avg_loss, train_metric

    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Returns:
            Tuple of (val_loss, val_metric)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                predictions = self.model(data)
                loss = self.criterion(predictions.squeeze(), targets)

                total_loss += loss.item()
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        val_metric = self.evaluator.calculate_spearman_correlation(
            all_targets, all_predictions
        )

        return avg_loss, val_metric

    def train(
        self,
        epochs: int,
        save_dir: Optional[str] = None,
        early_stopping_patience: Optional[int] = None,
        save_best_only: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model

        Returns:
            Training history
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        early_stopping_patience = early_stopping_patience or self.config.get('early_stopping_patience', 10)

        self.logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_metric = self.train_epoch()

            # Validate
            val_loss, val_metric = self.validate_epoch()

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metric'].append(train_metric)
            self.training_history['val_metric'].append(val_metric)
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # Save best model
            if save_dir and val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if save_best_only:
                    # Remove previous best model
                    for file in save_path.glob('best_model_*.pth'):
                        file.unlink()

                # Save current best model
                checkpoint_path = save_path / f'best_model_epoch_{epoch + 1}_metric_{val_metric:.4f}.pth'
                self.save_checkpoint(checkpoint_path, is_best=True)
                self.logger.info(f"Saved best model: {checkpoint_path}")
            else:
                self.patience_counter += 1

            # Save regular checkpoint
            if save_dir and not save_best_only:
                checkpoint_path = save_path / f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(checkpoint_path, is_best=False)

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")

        # Save final model
        if save_dir:
            final_path = save_path / 'final_model.pth'
            self.save_checkpoint(final_path, is_best=False)

        # Save training history
        if save_dir:
            history_path = save_path / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)

        return self.training_history

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """
        Save a training checkpoint.

        Args:
            filepath: Path to save the checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'training_history': self.training_history,
            'config': self.config.to_dict(),
            'is_best': is_best
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """
        Load a training checkpoint.

        Args:
            filepath: Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metric = checkpoint['best_val_metric']
        self.training_history = checkpoint['training_history']

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.

        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Metric plot
        axes[0, 1].plot(self.training_history['train_metric'], label='Train Metric')
        axes[0, 1].plot(self.training_history['val_metric'], label='Val Metric')
        axes[0, 1].set_title('Training and Validation Metric (Spearman)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Spearman Correlation')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate plot
        axes[1, 0].plot(self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)

        # Loss vs Metric
        axes[1, 1].scatter(self.training_history['val_loss'], self.training_history['val_metric'])
        axes[1, 1].set_title('Validation Loss vs Metric')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Metric')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
