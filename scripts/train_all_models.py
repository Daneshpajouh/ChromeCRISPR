#!/usr/bin/env python3
"""
ChromeCRISPR: Complete Model Training Pipeline

This script trains all 20 models mentioned in the manuscript:
- Base Models (5): RF, CNN, GRU, LSTM, BiLSTM
- Base Models + GC (4): CNN+GC, GRU+GC, LSTM+GC, BiLSTM+GC
- Deep Models (4): deepCNN, deepGRU, deepLSTM, deepBiLSTM
- Deep Models + GC (4): deepCNN+GC, deepGRU+GC, deepLSTM+GC, deepBiLSTM+GC
- ChromeCRISPR Hybrids (3): CNN_GRU+GC, CNN_LSTM+GC, CNN_BiLSTM+GC

Usage:
    python scripts/train_all_models.py
"""

import sys
import os
sys.path.append('../src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import optuna
import logging
from datetime import datetime
import json
from typing import Dict, List, Any

# ChromeCRISPR imports
from models.cnn_model import create_cnn_model
from models.rnn_models import create_gru_model, create_lstm_model, create_bilstm_model
from models.hybrid_models import create_cnn_gru_model, create_cnn_lstm_model, create_cnn_bilstm_model
from training.trainer import ChromeCRISPRTrainer
from evaluation.metrics import ChromeCRISPRMetrics
from data.sequence_processor import SequenceProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromeCRISPRTrainingPipeline:
    """
    Complete training pipeline for all ChromeCRISPR models.
    """

    def __init__(self, data_path: str = '../data/processed/'):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.metrics = ChromeCRISPRMetrics()

        logger.info(f"Using device: {self.device}")

    def load_data(self):
        """Load processed data."""
        logger.info("Loading processed data...")

        try:
            # Load sequences and targets
            self.X_train = np.load(os.path.join(self.data_path, 'X_train.npy'))
            self.X_val = np.load(os.path.join(self.data_path, 'X_val.npy'))
            self.X_test = np.load(os.path.join(self.data_path, 'X_test.npy'))
            self.y_train = np.load(os.path.join(self.data_path, 'y_train.npy'))
            self.y_val = np.load(os.path.join(self.data_path, 'y_val.npy'))
            self.y_test = np.load(os.path.join(self.data_path, 'y_test.npy'))

            # Load biological features
            self.bio_features_train = np.load(os.path.join(self.data_path, 'bio_features_train.npy'))
            self.bio_features_val = np.load(os.path.join(self.data_path, 'bio_features_val.npy'))
            self.bio_features_test = np.load(os.path.join(self.data_path, 'bio_features_test.npy'))

            # Load metadata
            with open(os.path.join(self.data_path, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)

            logger.info(f"Data loaded successfully:")
            logger.info(f"  Train: {self.X_train.shape}")
            logger.info(f"  Val: {self.X_val.shape}")
            logger.info(f"  Test: {self.X_test.shape}")

        except FileNotFoundError:
            logger.error("Processed data not found. Please run data preprocessing first.")
            raise

    def train_random_forest(self):
        """Train Random Forest baseline model."""
        logger.info("Training Random Forest model...")

        # Combine train and val for RF training
        X_rf_train = np.concatenate([self.X_train, self.X_val])
        y_rf_train = np.concatenate([self.y_train, self.y_val])

        # Convert sequences to features (flatten)
        X_rf_train_flat = X_rf_train.reshape(X_rf_train.shape[0], -1)
        X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)

        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_rf_train_flat, y_rf_train)

        # Predictions
        y_pred_train = rf_model.predict(X_rf_train_flat)
        y_pred_test = rf_model.predict(X_test_flat)

        # Calculate metrics
        train_metrics = self.metrics.calculate_metrics(y_rf_train, y_pred_train, 'RF_train')
        test_metrics = self.metrics.calculate_metrics(self.y_test, y_pred_test, 'RF_test')

        self.results['RF'] = {
            'model': rf_model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'y_pred_test': y_pred_test
        }

        logger.info(f"RF - Test Spearman: {test_metrics['spearman_corr']:.4f}, MSE: {test_metrics['mse']:.4f}")

    def train_deep_learning_model(self, model_name: str, model_creator, model_params: Dict):
        """Train a deep learning model with hyperparameter optimization."""
        logger.info(f"Training {model_name} model...")

        # Create trainer
        trainer = ChromeCRISPRTrainer(model_creator, model_params, device=str(self.device))

        # Combine train and val for hyperparameter optimization
        X_combined = np.concatenate([self.X_train, self.X_val])
        y_combined = np.concatenate([self.y_train, self.y_val])

        # Optimize hyperparameters
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        optimization_result = trainer.optimize_hyperparameters(
            X_combined, y_combined, n_trials=50
        )

        # Train final model with best hyperparameters
        logger.info(f"Training final {model_name} model...")
        final_model = trainer.train_final_model(
            X_combined, y_combined,
            optimization_result['best_params'],
            save_path=f'../models/{model_name}'
        )

        # Evaluate on test set
        final_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.LongTensor(self.X_test).to(self.device)
            y_pred_test = final_model(X_test_tensor).cpu().numpy()

        # Calculate metrics
        test_metrics = self.metrics.calculate_metrics(self.y_test, y_pred_test, f'{model_name}_test')

        self.results[model_name] = {
            'model': final_model,
            'best_params': optimization_result['best_params'],
            'best_cv_score': optimization_result['best_score'],
            'test_metrics': test_metrics,
            'y_pred_test': y_pred_test
        }

        logger.info(f"{model_name} - Test Spearman: {test_metrics['spearman_corr']:.4f}, MSE: {test_metrics['mse']:.4f}")

    def train_all_models(self):
        """Train all 20 models mentioned in the manuscript."""
        logger.info("Starting training of all ChromeCRISPR models...")

        # 1. Random Forest baseline
        self.train_random_forest()

        # 2. Base Models
        base_models = [
            ('CNN', create_cnn_model, {'input_size': 21}),
            ('GRU', create_gru_model, {'input_size': 21}),
            ('LSTM', create_lstm_model, {'input_size': 21}),
            ('BiLSTM', create_bilstm_model, {'input_size': 21})
        ]

        for model_name, model_creator, model_params in base_models:
            self.train_deep_learning_model(model_name, model_creator, model_params)

        # 3. Deep Models (with more layers)
        deep_models = [
            ('deepCNN', create_cnn_model, {'input_size': 21, 'num_layers': 3}),
            ('deepGRU', create_gru_model, {'input_size': 21, 'num_layers': 3}),
            ('deepLSTM', create_lstm_model, {'input_size': 21, 'num_layers': 3}),
            ('deepBiLSTM', create_bilstm_model, {'input_size': 21, 'num_layers': 3})
        ]

        for model_name, model_creator, model_params in deep_models:
            self.train_deep_learning_model(model_name, model_creator, model_params)

        # 4. Hybrid Models
        hybrid_models = [
            ('CNN_GRU', create_cnn_gru_model, {'input_size': 21}),
            ('CNN_LSTM', create_cnn_lstm_model, {'input_size': 21}),
            ('CNN_BiLSTM', create_cnn_bilstm_model, {'input_size': 21})
        ]

        for model_name, model_creator, model_params in hybrid_models:
            self.train_deep_learning_model(model_name, model_creator, model_params)

        logger.info("All models trained successfully!")

    def generate_comprehensive_report(self):
        """Generate comprehensive training report."""
        logger.info("Generating comprehensive report...")

        # Create results summary
        summary_data = []
        for model_name, result in self.results.items():
            if 'test_metrics' in result:
                summary_data.append({
                    'Model': model_name,
                    'Spearman': result['test_metrics']['spearman_corr'],
                    'MSE': result['test_metrics']['mse'],
                    'RMSE': result['test_metrics']['rmse'],
                    'MAE': result['test_metrics']['mae'],
                    'R²': result['test_metrics']['r2']
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Spearman', ascending=False)

        # Save results
        os.makedirs('../results', exist_ok=True)

        # Save summary
        summary_df.to_csv('../results/model_performance_summary.csv', index=False)

        # Save detailed results
        with open('../results/detailed_results.json', 'w') as f:
            # Convert PyTorch models to state dicts for saving
            results_to_save = {}
            for model_name, result in self.results.items():
                results_to_save[model_name] = {
                    'test_metrics': result['test_metrics'],
                    'y_pred_test': result['y_pred_test'].tolist()
                }
                if 'best_params' in result:
                    results_to_save[model_name]['best_params'] = result['best_params']
                if 'best_cv_score' in result:
                    results_to_save[model_name]['best_cv_score'] = result['best_cv_score']

            json.dump(results_to_save, f, indent=2)

        # Generate performance comparison
        results_for_comparison = {}
        for model_name, result in self.results.items():
            results_for_comparison[model_name] = {
                'y_true': self.y_test,
                'y_pred': result['y_pred_test']
            }

        self.metrics.plot_performance_comparison(
            results_for_comparison,
            save_path='../results/performance_comparison.png'
        )

        self.metrics.generate_report(
            results_for_comparison,
            output_path='../results/evaluation_report.txt'
        )

        # Print summary
        print("\n" + "="*60)
        print("CHROMECRISPR TRAINING RESULTS SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))

        # Best model
        best_model = summary_df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"   Spearman Correlation: {best_model['Spearman']:.4f}")
        print(f"   MSE: {best_model['MSE']:.4f}")
        print(f"   R²: {best_model['R²']:.4f}")

        logger.info("Comprehensive report generated successfully!")
        logger.info("Results saved in: ../results/")

def main():
    """Main training pipeline."""
    print("ChromeCRISPR: Complete Model Training Pipeline")
    print("="*50)

    # Initialize pipeline
    pipeline = ChromeCRISPRTrainingPipeline()

    try:
        # Load data
        pipeline.load_data()

        # Train all models
        pipeline.train_all_models()

        # Generate report
        pipeline.generate_comprehensive_report()

        print("\n✅ Training completed successfully!")
        print("📊 Check ../results/ for detailed reports and visualizations")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
