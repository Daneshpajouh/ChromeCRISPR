# deepBiLSTM Model: Deep Bidirectional LSTM

**Performance**: 0.862 Spearman correlation | **Rank**: #10 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | deepBiLSTM |
| **Architecture Type** | Deep Bidirectional LSTM |
| **Spearman Correlation** | 0.862 |
| **Mean Squared Error** | 0.0104 |
| **Total Parameters** | 5,992,229 |
| **Training Time** | 11.2 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **Total Input Features**: 84

### Deep BiLSTM Architecture
```
Input (84 features) → Embedding (128) → Deep BiLSTM Layers → Dense Layers → 1 output

BiLSTM Layers:
├── BiLSTM Layer 1: 256 hidden × 2, Dropout(0.2)
├── BiLSTM Layer 2: 256 hidden × 2, Dropout(0.2)
├── BiLSTM Layer 3: 256 hidden × 2, Dropout(0.2)
└── BiLSTM Layer 4: 256 hidden × 2, Dropout(0.2)
```

### Dense Layers
```
BiLSTM Output (512) → Dense Layers → 1 output

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
| **Learning Rate** | 0.00008 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 68 | Up to 200 | Early Stopping |

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the deepBiLSTM model
model = ChromeCRISPRModel.load_from_file('models/deep_models/deepBiLSTM.pth')
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

- **[Raw JSON Specs](../hyperparameters/deepBiLSTM_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**deepBiLSTM provides comprehensive bidirectional context with 0.862 Spearman correlation, offering the most complete sequence representation.**
