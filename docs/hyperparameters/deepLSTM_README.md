# deepLSTM Model: Deep Long Short-Term Memory

**Performance**: 0.862 Spearman correlation | **Rank**: #9 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | deepLSTM |
| **Architecture Type** | Deep Long Short-Term Memory |
| **Spearman Correlation** | 0.862 |
| **Mean Squared Error** | 0.0103 |
| **Total Parameters** | 2,343,557 |
| **Training Time** | 9.1 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **Total Input Features**: 84

### Deep LSTM Architecture
```
Input (84 features) → Embedding (128) → Deep LSTM Layers → Dense Layers → 1 output

LSTM Layers:
├── LSTM Layer 1: 256 hidden units, Dropout(0.2)
├── LSTM Layer 2: 256 hidden units, Dropout(0.2)
├── LSTM Layer 3: 256 hidden units, Dropout(0.2)
└── LSTM Layer 4: 256 hidden units, Dropout(0.2)
```

### Dense Layers
```
LSTM Output (256) → Dense Layers → 1 output

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
| **Learning Rate** | 0.00009 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 71 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **LSTM Hidden Size** | All layers | 256 |
| **LSTM Layers** | Count | 4 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |
| **Dense Units** | Layer 3 | 32 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.860 | 0.0105 |
| Fold 2 | 0.864 | 0.0101 |
| Fold 3 | 0.861 | 0.0104 |
| Fold 4 | 0.863 | 0.0102 |
| Fold 5 | 0.862 | 0.0103 |
| **Mean** | **0.862** | **0.0103** |
| **Std** | **0.002** | **0.0002** |

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 25GB
- **Platform**: Digital Research Alliance of Canada

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the deepLSTM model
model = ChromeCRISPRModel.load_from_file('models/deep_models/deepLSTM.pth')
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

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/deepLSTM_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**deepLSTM provides deep memory modeling with 0.862 Spearman correlation, offering comprehensive temporal processing for sequence analysis.**
