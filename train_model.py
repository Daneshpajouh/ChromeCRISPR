#!/usr/bin/env python3
"""
ChromeCRISPR Training Script

This script trains ChromeCRISPR models with proper train/validation/test splits
and comprehensive evaluation metrics.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.dynamic_model import DynamicModel
from src.data.dataset import CRISPRDataset
from src.training.trainer import Trainer
from src.evaluation.metrics import EvaluationMetrics
from src.utils.config import Config


def setup_logging(config: Config):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create logs directory
    log_file = log_config.get('file', 'logs/chromecrispr.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if log_config.get('console', True) else logging.NullHandler()
        ]
    )


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data_loaders(config: Config):
    """Create train, validation, and test data loaders."""
    data_config = config.get_data_config()

    # Create dataset splits
    train_dataset, val_dataset, test_dataset = CRISPRDataset.create_splits(
        data_path=data_config['data_path'],
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        random_state=config.get('hardware.seed', 42),
        use_gc=data_config['use_gc'],
        use_bio_features=data_config['use_bio_features'],
        normalize=data_config['normalize']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset


def create_model(config: Config, feature_dim: int):
    """Create the model."""
    model_config = config.get_model_config()

    # Update input size based on feature dimension
    model_config['input_size'] = feature_dim

    model = DynamicModel(**model_config)

    return model


def train_model(model: DynamicModel, train_loader: DataLoader, val_loader: DataLoader, config: Config):
    """Train the model."""
    training_config = config.get_training_config()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Train the model
    history = trainer.train(
        epochs=training_config['epochs'],
        save_dir=config.get('output.checkpoints_dir', 'results/checkpoints'),
        early_stopping_patience=training_config['early_stopping_patience'],
        save_best_only=training_config['save_best_only']
    )

    return trainer, history


def evaluate_model(model: DynamicModel, test_loader: DataLoader, config: Config):
    """Evaluate the model on test set."""
    evaluator = EvaluationMetrics()

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            device = next(model.parameters()).device
            data, targets = data.to(device), targets.to(device)

            predictions = model(data)
            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    metrics = evaluator.calculate_all_metrics(all_targets, all_predictions, prefix="test_")

    return metrics, all_predictions, all_targets


def save_results(metrics: dict, predictions: list, targets: list, config: Config):
    """Save evaluation results."""
    output_dir = Path(config.get('output.save_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    import json
    metrics_file = output_dir / 'test_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    import pandas as pd
    predictions_df = pd.DataFrame({
        'true_activity': targets,
        'predicted_activity': predictions
    })
    predictions_file = output_dir / 'test_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)

    # Create evaluation report
    evaluator = EvaluationMetrics()
    report = evaluator.create_metrics_report(
        targets, predictions,
        model_name=config.get('model.model_type', 'ChromeCRISPR Model')
    )

    report_file = output_dir / 'evaluation_report.md'
    with open(report_file, 'w') as f:
        f.write(report)

    # Save plots
    plots_dir = Path(config.get('output.plots_dir', 'results/plots'))
    plots_dir.mkdir(parents=True, exist_ok=True)

    evaluator.plot_prediction_scatter(
        targets, predictions,
        title=f"ChromeCRISPR {config.get('model.model_type', 'Model')} Predictions",
        save_path=plots_dir / 'prediction_scatter.png'
    )

    evaluator.plot_residuals(
        targets, predictions,
        title=f"ChromeCRISPR {config.get('model.model_type', 'Model')} Residuals",
        save_path=plots_dir / 'residuals.png'
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ChromeCRISPR model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-path', type=str, help='Override data path')
    parser.add_argument('--model-type', type=str, help='Override model type')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--output-dir', type=str, help='Override output directory')

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Override config with command line arguments
    if args.data_path:
        config.set('data.data_path', args.data_path)
    if args.model_type:
        config.set('model.model_type', args.model_type)
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.batch_size:
        config.set('data.batch_size', args.batch_size)
    if args.learning_rate:
        config.set('optimizer.learning_rate', args.learning_rate)
    if args.output_dir:
        config.set('output.save_dir', args.output_dir)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    # Set random seed
    seed = config.get('hardware.seed', 42)
    set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Validate configuration
    if not config.validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)

    logger.info("Starting ChromeCRISPR training")
    logger.info(f"Configuration: {config.to_dict()}")

    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(config)

        # Create model
        logger.info("Creating model...")
        feature_dim = train_dataset.get_feature_dim()
        model = create_model(config, feature_dim)

        # Log model information
        model_info = model.get_model_info()
        logger.info(f"Model created: {model_info}")

        # Train model
        logger.info("Starting training...")
        trainer, history = train_model(model, train_loader, val_loader, config)

        # Evaluate model
        logger.info("Evaluating model...")
        metrics, predictions, targets = evaluate_model(model, test_loader, config)

        # Save results
        logger.info("Saving results...")
        save_results(metrics, predictions, targets, config)

        # Log final metrics
        logger.info("Training completed successfully!")
        logger.info(f"Test Spearman Correlation: {metrics['test_spearman_correlation']:.4f}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.6f}")
        logger.info(f"Test R² Score: {metrics['test_r2_score']:.4f}")

        # Save training history plot
        plots_dir = Path(config.get('output.plots_dir', 'results/plots'))
        trainer.plot_training_history(save_path=plots_dir / 'training_history.png')

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
