"""
Configuration management for ChromeCRISPR.

This module provides configuration management utilities for the
ChromeCRISPR project, including YAML configuration loading and
parameter management.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


class Config:
    """
    Configuration management class for ChromeCRISPR.

    This class handles loading, validation, and access to configuration
    parameters for the ChromeCRISPR project.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

    Args:
        config_path: Path to configuration file
        """
        self.config_data = {}
        self.logger = logging.getLogger(__name__)

        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_config()

    def _load_default_config(self):
        """Load default configuration."""
        self.config_data = {
            'data': {
                'data_path': 'data/processed/crispr_dataset.csv',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'batch_size': 32,
                'num_workers': 4,
                'use_gc': True,
                'use_bio_features': False,
                'normalize': True
            },
            'model': {
                'model_type': 'cnn_gru',
                'input_size': 21,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'use_gc': True,
                'use_bio_features': False,
                'num_classes': 1
            },
            'training': {
                'epochs': 100,
                'early_stopping_patience': 10,
                'save_best_only': True,
                'gradient_clipping': 1.0
            },
            'optimizer': {
                'name': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 0.0,
                'momentum': 0.9
            },
            'scheduler': {
                'name': 'reduce_on_plateau',
                'patience': 10,
                'factor': 0.5,
                'min_lr': 1e-6
            },
            'loss': {
                'name': 'mse',
                'delta': 1.0
            },
            'hyperparameter_tuning': {
                'n_trials': 100,
                'timeout': 3600,
                'n_jobs': 1
            },
            'evaluation': {
                'metrics': ['spearman', 'pearson', 'mse', 'mae', 'r2'],
                'high_activity_threshold': 0.8,
                'low_activity_threshold': 0.2
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/chromecrispr.log'
            },
            'output': {
                'save_dir': 'results',
                'model_dir': 'models',
                'plots_dir': 'results/plots',
                'logs_dir': 'results/logs'
            }
        }

    def load_config(self, config_path: str):
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file
        """
        config_file = Path(config_path)

        if not config_file.exists():
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return

        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)

            # Merge with defaults
            self._merge_config(file_config)
            self.logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
            self.logger.info("Using default configuration")

    def _merge_config(self, new_config: Dict[str, Any]):
        """
        Merge new configuration with existing config.

        Args:
            new_config: New configuration to merge
        """
        def merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge dictionaries."""
            result = base.copy()
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result

        self.config_data = merge_dicts(self.config_data, new_config)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config_data

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config_data

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def save_config(self, config_path: str):
        """
        Save configuration to file.

        Args:
            config_path: Path to save configuration
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_file, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config file: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config_data.copy()

    def validate_config(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid
        """
        required_keys = [
            'data.data_path',
            'model.model_type',
            'training.epochs',
            'optimizer.learning_rate'
        ]

        for key in required_keys:
            if self.get(key) is None:
                self.logger.error(f"Missing required configuration key: {key}")
                return False

        # Validate model type
        valid_model_types = [
            'cnn', 'lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'transformer',
            'cnn_lstm', 'cnn_gru', 'cnn_bilstm', 'cnn_bigru'
        ]

        model_type = self.get('model.model_type')
        if model_type not in valid_model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return False

        # Validate ratios sum to 1
        train_ratio = self.get('data.train_ratio', 0.7)
        val_ratio = self.get('data.val_ratio', 0.15)
        test_ratio = self.get('data.test_ratio', 0.15)

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            self.logger.error("Data split ratios must sum to 1.0")
            return False

        self.logger.info("Configuration validation passed")
        return True

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration.

        Returns:
            Model configuration dictionary
        """
        return {
            'model_type': self.get('model.model_type'),
            'input_size': self.get('model.input_size', 21),
            'hidden_size': self.get('model.hidden_size', 128),
            'num_layers': self.get('model.num_layers', 2),
            'dropout': self.get('model.dropout', 0.3),
            'use_gc': self.get('model.use_gc', True),
            'use_bio_features': self.get('model.use_bio_features', False),
            'num_classes': self.get('model.num_classes', 1)
        }

    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training-specific configuration.

        Returns:
            Training configuration dictionary
        """
        return {
            'epochs': self.get('training.epochs', 100),
            'early_stopping_patience': self.get('training.early_stopping_patience', 10),
            'save_best_only': self.get('training.save_best_only', True),
            'gradient_clipping': self.get('training.gradient_clipping', 1.0)
        }

    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data-specific configuration.

        Returns:
            Data configuration dictionary
        """
        return {
            'data_path': self.get('data.data_path'),
            'train_ratio': self.get('data.train_ratio', 0.7),
            'val_ratio': self.get('data.val_ratio', 0.15),
            'test_ratio': self.get('data.test_ratio', 0.15),
            'batch_size': self.get('data.batch_size', 32),
            'num_workers': self.get('data.num_workers', 4),
            'use_gc': self.get('data.use_gc', True),
            'use_bio_features': self.get('data.use_bio_features', False),
            'normalize': self.get('data.normalize', True)
        }
