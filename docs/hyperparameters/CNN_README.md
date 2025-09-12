# CNN Model: Convolutional Neural Network Baseline

**Performance**: 0.793 Spearman correlation | **Rank**: #17 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | CNN |
| **Architecture Type** | Convolutional Neural Network |
| **Spearman Correlation** | 0.793 |
| **Mean Squared Error** | 0.0161 |
| **Total Parameters** | 125,432 |
| **Training Time** | 2.5 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **Total Input Features**: 84

### CNN Architecture
```
Input (84 features) → Embedding (128) → Conv Layers → Global Max Pooling → Dense Layers

Convolutional Layers:
├── Conv1D Layer 1: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.2)
└── Conv1D Layer 2: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.2)
```

### Dense Layers
```
Global Max Pooling (64) → Dense Layers → 1 output

Dense Layers:
├── Dense Layer 1: 128 units, ReLU, Dropout(0.2)
├── Dense Layer 2: 64 units, ReLU, Dropout(0.2)
├── Dense Layer 3: 32 units, ReLU, Dropout(0.2)
└── Output Layer: 1 unit, Linear activation
```

## Optimal Hyperparameters

### Training Configuration
- **Optimizer Selection**: Fixed as Adam (not part of hyperparameter search)| Parameter | Optimal Value | Search Range | Tuning Method |
|-----------|---------------|--------------|----------------|
| **Learning Rate** | 0.001 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 100 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **CNN Filters** | All layers | 64 |
| **CNN Kernel Size** | All layers | 5 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |
| **Dense Units** | Layer 3 | 32 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.7893 | 0.0164 |
| Fold 2 | 0.7945 | 0.0158 |
| Fold 3 | 0.7912 | 0.0162 |
| Fold 4 | 0.7956 | 0.0157 |
| Fold 5 | 0.7919 | 0.0163 |
| **Mean** | **0.7925** | **0.0161** |
| **Std** | **0.0032** | **0.0003** |

### Performance by Metric
- **Spearman Correlation**: 0.793 (95% CI: 0.789-0.797)
- **Mean Squared Error**: 0.0161
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 8GB
- **Platform**: Digital Research Alliance of Canada

### Training Configuration
- **Optimizer Selection**: Fixed as Adam (not part of hyperparameter search)- **Framework**: PyTorch 1.12.0
- **Optimizer**: Adam (β1=0.9, β2=0.999, weight_decay=1e-5)
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience=10 epochs
- **Model Checkpointing**: Save best model

## Hyperparameter Tuning

### Bayesian Optimization Setup
- **Framework**: Optuna
- **Trials**: 100 total
- **Best Trial**: Trial 42
- **Objective**: Maximize Spearman correlation
- **Validation**: 5-fold cross-validation

### Search Spaces
```python
learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
cnn_filters = trial.suggest_categorical('cnn_filters', [32, 64, 128])
kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
```

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the CNN model
model = ChromeCRISPRModel.load_from_file('models/base_models/CNN.pth')
model.eval()

# Prepare input sequence
sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCG"  # 21-mer
X = model.preprocess_sequence(sequence)

# Make prediction
with torch.no_grad():
    prediction = model(X)
    activity_score = prediction.item()
    print(f"Predicted CRISPR activity: {activity_score:.4f}")
```

## Model Specifications Summary

### Architecture Summary
- **Input**: 21-mer DNA sequence (84 features)
- **CNN Branch**: 2 conv layers (64 filters each) → 64 features
- **Global Pooling**: Max pooling across sequence dimension
- **Dense**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 125K

### Key Features
1. **Convolutional Filters**: Capture sequence motifs and patterns
2. **Batch Normalization**: Stable training and faster convergence
3. **Dropout Regularization**: Prevents overfitting
4. **Global Max Pooling**: Captures most important sequence features

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/CNN_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**CNN serves as a strong baseline for motif detection in CRISPR sequences, achieving 0.793 Spearman correlation.**
