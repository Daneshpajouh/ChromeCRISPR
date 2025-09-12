# GRU Model: Gated Recurrent Unit Baseline

**Performance**: 0.837 Spearman correlation | **Rank**: #13 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | GRU |
| **Architecture Type** | Gated Recurrent Unit |
| **Spearman Correlation** | 0.837 |
| **Mean Squared Error** | 0.0121 |
| **Total Parameters** | 907,525 |
| **Training Time** | 3.2 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **Total Input Features**: 84

### GRU Architecture
```
Input (84 features) → Embedding (128) → GRU Layers → Dense Layers → 1 output

GRU Layers:
├── GRU Layer 1: 128 hidden units, Dropout(0.2)
└── GRU Layer 2: 128 hidden units, Dropout(0.2)
```

### Dense Layers
```
GRU Output (128) → Dense Layers → 1 output

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
| **Learning Rate** | 0.0005 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 95 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **GRU Hidden Size** | All layers | 128 |
| **GRU Layers** | Count | 2 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |
| **Dense Units** | Layer 3 | 32 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.8352 | 0.0123 |
| Fold 2 | 0.8389 | 0.0119 |
| Fold 3 | 0.8367 | 0.0121 |
| Fold 4 | 0.8378 | 0.0120 |
| Fold 5 | 0.8365 | 0.0122 |
| **Mean** | **0.8370** | **0.0121** |
| **Std** | **0.0014** | **0.0002** |

### Performance by Metric
- **Spearman Correlation**: 0.837 (95% CI: 0.834-0.840)
- **Mean Squared Error**: 0.0121
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 12GB
- **Platform**: Digital Research Alliance of Canada

### Training Configuration
- **Framework**: PyTorch 1.12.0
- **Optimizer**: Adam (β1=0.9, β2=0.999)
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience=10 epochs
- **Model Checkpointing**: Save best model

## Hyperparameter Tuning

### Bayesian Optimization Setup
- **Framework**: Optuna
- **Trials**: 100 total
- **Best Trial**: Trial 67
- **Objective**: Maximize Spearman correlation
- **Validation**: 5-fold cross-validation

### Search Spaces
```python
learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
gru_hidden = trial.suggest_categorical('gru_hidden', [64, 128, 256])
num_layers = trial.suggest_int('num_layers', 1, 3)
```

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the GRU model
model = ChromeCRISPRModel.load_from_file('models/base_models/GRU.pth')
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
- **GRU Branch**: 2 GRU layers (128 hidden each) → 128 features
- **Dense**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 907K

### Key Features
1. **Gated Memory**: Update and reset gates control information flow
2. **Efficient Training**: Fewer parameters than LSTM
3. **Sequence Context**: Captures long-range dependencies
4. **Regularization**: Dropout prevents overfitting

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/GRU_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**GRU provides efficient sequence modeling with 0.837 Spearman correlation, offering a good balance between performance and computational cost.**
