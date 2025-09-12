# LSTM Model: Long Short-Term Memory Baseline

**Performance**: 0.837 Spearman correlation | **Rank**: #12 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | LSTM |
| **Architecture Type** | Long Short-Term Memory |
| **Spearman Correlation** | 0.837 |
| **Mean Squared Error** | 0.0122 |
| **Total Parameters** | 1,170,309 |
| **Training Time** | 3.8 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **Total Input Features**: 84

### LSTM Architecture
```
Input (84 features) → Embedding (128) → LSTM Layers → Dense Layers → 1 output

LSTM Layers:
├── LSTM Layer 1: 128 hidden units, Dropout(0.2)
└── LSTM Layer 2: 128 hidden units, Dropout(0.2)
```

### Dense Layers
```
LSTM Output (128) → Dense Layers → 1 output

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
| **Learning Rate** | 0.0008 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 85 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **LSTM Hidden Size** | All layers | 128 |
| **LSTM Layers** | Count | 2 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |
| **Dense Units** | Layer 3 | 32 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.8358 | 0.0124 |
| Fold 2 | 0.8382 | 0.0120 |
| Fold 3 | 0.8369 | 0.0121 |
| Fold 4 | 0.8375 | 0.0121 |
| Fold 5 | 0.8368 | 0.0122 |
| **Mean** | **0.8370** | **0.0122** |
| **Std** | **0.0010** | **0.0002** |

### Performance by Metric
- **Spearman Correlation**: 0.837 (95% CI: 0.834-0.840)
- **Mean Squared Error**: 0.0122
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 14GB
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
- **Best Trial**: Trial 73
- **Objective**: Maximize Spearman correlation
- **Validation**: 5-fold cross-validation

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the LSTM model
model = ChromeCRISPRModel.load_from_file('models/base_models/LSTM.pth')
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
- **LSTM Branch**: 2 LSTM layers (128 hidden each) → 128 features
- **Dense**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 1.17M

### Key Features
1. **Cell State**: Long-term memory preservation
2. **Three Gates**: Input, forget, and output gates
3. **Gradient Flow**: Mitigates vanishing gradient problem
4. **Sequence Memory**: Excellent for long-range dependencies

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/LSTM_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**LSTM provides robust long-term memory modeling with 0.837 Spearman correlation, performing comparably to GRU with enhanced memory capacity.**
