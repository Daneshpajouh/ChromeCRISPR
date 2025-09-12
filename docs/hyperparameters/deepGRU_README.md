# deepGRU Model: Deep Gated Recurrent Unit

**Performance**: 0.868 Spearman correlation | **Rank**: #7 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | deepGRU |
| **Architecture Type** | Deep Gated Recurrent Unit |
| **Spearman Correlation** | 0.868 |
| **Mean Squared Error** | 0.0099 |
| **Total Parameters** | 1,820,741 |
| **Training Time** | 8.4 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **Total Input Features**: 84

### Deep GRU Architecture
```
Input (84 features) → Embedding (128) → Deep GRU Layers → Dense Layers → 1 output

GRU Layers:
├── GRU Layer 1: 256 hidden units, Dropout(0.2)
├── GRU Layer 2: 256 hidden units, Dropout(0.2)
├── GRU Layer 3: 256 hidden units, Dropout(0.2)
└── GRU Layer 4: 256 hidden units, Dropout(0.2)
```

### Dense Layers
```
GRU Output (256) → Dense Layers → 1 output

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
| **Learning Rate** | 0.00012 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 76 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **GRU Hidden Size** | All layers | 256 |
| **GRU Layers** | Count | 4 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |
| **Dense Units** | Layer 3 | 32 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.866 | 0.0101 |
| Fold 2 | 0.870 | 0.0097 |
| Fold 3 | 0.867 | 0.0100 |
| Fold 4 | 0.869 | 0.0098 |
| Fold 5 | 0.868 | 0.0099 |
| **Mean** | **0.868** | **0.0099** |
| **Std** | **0.002** | **0.0002** |

### Performance by Metric
- **Spearman Correlation**: 0.868 (95% CI: 0.865-0.871)
- **Mean Squared Error**: 0.0099
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 22GB
- **Platform**: Digital Research Alliance of Canada

## Key Features

### Deep Architecture Benefits
1. **Hierarchical Memory**: Multi-layer temporal processing
2. **Long-range Dependencies**: Enhanced sequence context
3. **Efficient Computation**: Fewer parameters than LSTM
4. **Stable Gradients**: Better training convergence

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the deepGRU model
model = ChromeCRISPRModel.load_from_file('models/deep_models/deepGRU.pth')
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

- **[Raw JSON Specs](../hyperparameters/deepGRU_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**deepGRU provides deep temporal processing with 0.868 Spearman correlation, offering efficient hierarchical sequence modeling for CRISPR applications.**
