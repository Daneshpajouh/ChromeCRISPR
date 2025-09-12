#  ChromeCRISPR Source Code

Complete implementation of all ChromeCRISPR models, training utilities, and evaluation tools.

##  Source Code Structure

```
src/
├── models/           # Model implementations
├── evaluation/       # Performance evaluation tools
└── training/         # Training utilities and procedures
```

## 🏗️ Model Implementations (`models/`)

### Core Model Classes
- **`ChromeCRISPRModel`**: Main model class supporting all 20 architectures
- **`CNN_GRU_GC`**: Best performing hybrid model implementation
- **`BaseModel`**: Abstract base class for all model types

### Supported Architectures
```python
# All 20 model types supported
architectures = [
    # Hybrid Models (Best Performance)
    'cnn_gru_gc',        # CNN_GRU+GC (0.876 Spearman)
    'cnn_bilstm_gc',     # CNN_BiLSTM+GC (0.870 Spearman)
    'cnn_lstm_gc',       # CNN_LSTM+GC (0.867 Spearman)

    # Deep Models + GC
    'deep_cnn_gc',       # deepCNN+GC (0.873 Spearman)
    'deep_gru_gc',       # deepGRU+GC (0.867 Spearman)
    'deep_lstm_gc',      # deepLSTM+GC (0.860 Spearman)
    'deep_bilstm_gc',    # deepBiLSTM+GC (0.867 Spearman)

    # Deep Models
    'deep_cnn',          # deepCNN (0.869 Spearman)
    'deep_gru',          # deepGRU (0.868 Spearman)
    'deep_lstm',         # deepLSTM (0.862 Spearman)
    'deep_bilstm',       # deepBiLSTM (0.862 Spearman)

    # Base Models + GC
    'cnn_gc',            # CNN+GC (0.781 Spearman)
    'gru_gc',            # GRU+GC (0.840 Spearman)
    'lstm_gc',           # LSTM+GC (0.856 Spearman)
    'bilstm_gc',         # BiLSTM+GC (0.855 Spearman)

    # Base Models
    'cnn',               # CNN (0.793 Spearman)
    'gru',               # GRU (0.837 Spearman)
    'lstm',              # LSTM (0.837 Spearman)
    'bilstm'             # BiLSTM (0.843 Spearman)
]
```

##  Evaluation Tools (`evaluation/`)

### Performance Metrics
- **`spearman_correlation`**: Primary evaluation metric
- **`mean_squared_error`**: Secondary performance measure
- **`cross_validation_scores`**: 5-fold CV evaluation

### Model Comparison
```python
from src.evaluation import ModelEvaluator

# Compare multiple models
evaluator = ModelEvaluator()
results = evaluator.compare_models(X_test, y_test, model_list)

# Generate performance report
report = evaluator.generate_report(results)
print(report)
```

### Benchmarking Tools
- **Statistical Tests**: Significance testing between models
- **Performance Plots**: Visualization of model comparisons
- **Cross-validation**: Robust evaluation procedures

##  Training Utilities (`training/`)

### Data Processing
- **`SequenceEncoder`**: One-hot encoding for DNA sequences
- **`GCContentCalculator`**: Biological feature extraction
- **`DataLoader`**: Efficient batch loading and preprocessing

### Training Procedures
- **`Trainer`**: Unified training interface for all models
- **`HyperparameterOptimizer`**: Bayesian optimization with Optuna
- **`EarlyStopping`**: Prevent overfitting with validation monitoring

### Example Training Workflow
```python
from src.training import Trainer, HyperparameterOptimizer
from src.models import ChromeCRISPRModel

# Load and preprocess data
data_loader = DataLoader()
X_train, y_train, X_val, y_val = data_loader.load_deephf_data()

# Optimize hyperparameters
optimizer = HyperparameterOptimizer(model_type='cnn_gru_gc')
best_params = optimizer.optimize(X_train, y_train, X_val, y_val)

# Train final model
trainer = Trainer()
model = trainer.train(
    model_type='cnn_gru_gc',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    **best_params
)

# Save trained model
model.save('cnn_gru_gc_trained.pth')
```

##  Quick Start Examples

### Load and Use Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load best performing model
model = ChromeCRISPRModel.load_from_file('../models/chromecrispr_hybrid_models/CNN_GRU+GC.pth')

# Prepare input data
sequences = ['ATCGATCGATCGATCGATCGATCGATCGATCGATCG', ...]  # 21-mer sequences
X = model.preprocess_sequences(sequences)

# Make predictions
with torch.no_grad():
    predictions = model(X)
    spearman_corr = predictions.mean()  # Example aggregation
```

### Train New Model from Scratch
```python
from src.training import Trainer

# Initialize trainer
trainer = Trainer()

# Train CNN_GRU+GC model
model = trainer.train(
    model_type='cnn_gru_gc',
    X_train=X_train,
    y_train=y_train,
    learning_rate=0.000178,  # From hyperparameter optimization
    batch_size=64,
    epochs=100,
    early_stopping_patience=10
)

# Evaluate on test set
test_predictions = model.predict(X_test)
test_spearman = calculate_spearman(test_predictions, y_test)
print(f"Test Spearman: {test_spearman:.4f}")
```

### Hyperparameter Optimization
```python
from src.training import HyperparameterOptimizer

# Set up optimization
optimizer = HyperparameterOptimizer(
    model_type='cnn_gru_gc',
    search_space={
        'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-2},
        'batch_size': {'type': 'categorical', 'choices': [32, 64, 128]},
        'dropout_rate': {'type': 'uniform', 'low': 0.1, 'high': 0.5}
    }
)

# Run optimization
best_params = optimizer.optimize(
    X_train, y_train, X_val, y_val,
    n_trials=100,
    timeout=3600  # 1 hour timeout
)

print("Best hyperparameters:", best_params)
```

##  Requirements & Dependencies

### Core Dependencies
```txt
torch>=1.9.0          # PyTorch for neural networks
numpy>=1.21.0         # Numerical computing
pandas>=1.3.0         # Data manipulation
scikit-learn>=1.0.0   # Machine learning utilities
scipy>=1.7.0          # Scientific computing
optuna>=2.10.0        # Hyperparameter optimization
```

### Installation
```bash
# From project root
pip install -r requirements.txt

# Or install specific versions
pip install torch==1.12.0 numpy==1.21.0 pandas==1.3.0
```

##  Configuration & Customization

### Model Configuration
```python
# Custom model configuration
config = {
    'architecture': 'cnn_gru_gc',
    'input_dim': 85,           # 84 sequence + 1 GC
    'embedding_dim': 128,
    'cnn_filters': 64,
    'rnn_hidden': 384,
    'dense_units': [128, 64, 32],
    'dropout_rate': 0.142,
    'learning_rate': 0.000178,
    'batch_size': 64
}

model = ChromeCRISPRModel(config)
```

### Training Configuration
```python
training_config = {
    'optimizer': 'adam',
    'weight_decay': 2e-5,
    'lr_scheduler': 'cosine',
    'early_stopping': {
        'patience': 10,
        'min_delta': 0.001,
        'restore_best_weights': True
    },
    'checkpoint': {
        'save_best_only': True,
        'monitor': 'val_spearman',
        'mode': 'max'
    }
}

trainer = Trainer(training_config)
```

## 🐛 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **NaN Loss**: Check input data normalization and learning rate
3. **Poor Convergence**: Adjust learning rate or increase model capacity
4. **Memory Issues**: Use smaller models or implement gradient checkpointing

### Performance Optimization
```python
# Enable mixed precision training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    output = model(batch)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

##  Documentation Links

- **[Model Files](../models/)**: All 20 trained models
- **[Hyperparameters](../docs/hyperparameters/)**: Complete specifications for all models
- **[Training Procedures](../docs/training_procedures/)**: Detailed training protocols
- **[Model Architectures](../docs/MODEL_ARCHITECTURES.md)**: Technical implementation details

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/Daneshpajouh/ChromeCRISPR.git
cd ChromeCRISPR

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Include docstrings for all public functions
- Write unit tests for new functionality

## 📞 Support

For technical questions or issues:
- **Documentation**: Check hyperparameter files for model-specific details
- **Examples**: See usage examples above
- **Issues**: [GitHub Issues](../../issues) for bugs and feature requests
- **Contact**: amir_dp@sfu.ca for ChromeCRISPR-specific questions

---
** Note**: This source code provides complete implementations for reproducing all ChromeCRISPR results and extending the research to new architectures or datasets.
