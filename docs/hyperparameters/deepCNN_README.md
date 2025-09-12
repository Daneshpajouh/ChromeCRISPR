# deepCNN Model: Deep Convolutional Neural Network

**Performance**: 0.869 Spearman correlation | **Rank**: #6 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | deepCNN |
| **Architecture Type** | Deep Convolutional Neural Network |
| **Spearman Correlation** | 0.869 |
| **Mean Squared Error** | 0.0098 |
| **Total Parameters** | 665,289 |
| **Training Time** | 6.2 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **Total Input Features**: 84

### Deep CNN Architecture
```
Input (84 features) → Embedding (128) → Deep Conv Layers → Global Max Pooling → Dense Layers → 1 output

Convolutional Layers:
├── Conv1D Layer 1: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.2)
├── Conv1D Layer 2: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.2)
├── Conv1D Layer 3: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.2)
└── Conv1D Layer 4: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.2)
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
| Parameter | Optimal Value | Search Range | Tuning Method |
|-----------|---------------|--------------|----------------|
| **Learning Rate** | 0.00015 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 89 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **CNN Filters** | All layers | 64 |
| **CNN Kernel Size** | All layers | 5 |
| **Conv Layers** | Count | 4 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |
| **Dense Units** | Layer 3 | 32 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.867 | 0.0101 |
| Fold 2 | 0.871 | 0.0096 |
| Fold 3 | 0.868 | 0.0099 |
| Fold 4 | 0.870 | 0.0097 |
| Fold 5 | 0.869 | 0.0098 |
| **Mean** | **0.869** | **0.0098** |
| **Std** | **0.002** | **0.0002** |

### Performance by Metric
- **Spearman Correlation**: 0.869 (95% CI: 0.866-0.872)
- **Mean Squared Error**: 0.0098
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 16GB
- **Platform**: Digital Research Alliance of Canada

### Training Configuration
- **Framework**: PyTorch 1.12.0
- **Optimizer**: Adam (β1=0.9, β2=0.999)
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience=10 epochs
- **Model Checkpointing**: Save best model

## Feature Learning Analysis

### Hierarchical Feature Extraction
- **Layer 1**: Basic motif detection (3-5 bp patterns)
- **Layer 2**: Combined motif patterns (5-8 bp)
- **Layer 3**: Complex sequence features (8-12 bp)
- **Layer 4**: High-level sequence abstractions

### Sequence Position Importance
- **PAM Region**: Critical for Cas9 recognition
- **Seed Region**: Primary specificity determinant
- **Central Region**: Target accessibility features
- **Full Sequence**: Global sequence context

## Hyperparameter Tuning

### Bayesian Optimization Setup
- **Framework**: Optuna
- **Trials**: 100 total
- **Best Trial**: Trial 78
- **Objective**: Maximize Spearman correlation
- **Validation**: 5-fold cross-validation

### Search Spaces
```python
learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
cnn_filters = trial.suggest_categorical('cnn_filters', [32, 64, 128])
kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
num_layers = trial.suggest_int('num_layers', 2, 6)
```

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the deepCNN model
model = ChromeCRISPRModel.load_from_file('models/deep_models/deepCNN.pth')
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
- **CNN Branch**: 4 conv layers (64 filters each) → 64 features
- **Global Pooling**: Max pooling across sequence dimension
- **Dense**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 665K

### Key Features
1. **Deep Architecture**: 4 convolutional layers for hierarchical feature learning
2. **Progressive Feature Extraction**: From motifs to complex patterns
3. **Batch Normalization**: Stable training through deep layers
4. **Regularization**: Dropout prevents overfitting in deep network

### Advantages
- **Hierarchical Learning**: Multi-layer feature abstraction
- **Sequence Pattern Recognition**: Advanced motif detection
- **Scalable Architecture**: Foundation for deeper networks
- **Feature Richness**: Comprehensive sequence representation

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/deepCNN_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**deepCNN demonstrates the power of deep convolutional architectures with 0.869 Spearman correlation, providing hierarchical feature learning for comprehensive sequence analysis.**
