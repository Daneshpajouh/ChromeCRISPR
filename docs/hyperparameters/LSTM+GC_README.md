# LSTM+GC Model: LSTM with GC Content Integration

**Performance**: 0.856 Spearman correlation | **Rank**: #8 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | LSTM+GC |
| **Architecture Type** | Long Short-Term Memory + GC Content |
| **Spearman Correlation** | 0.856 |
| **Mean Squared Error** | 0.0112 |
| **Total Parameters** | 1,170,310 |
| **Training Time** | 3.9 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **GC Content**: 1 feature (calculated as (G+C)/21)
- **Total Input Features**: 85

### LSTM Architecture
```
Input (85 features) → Embedding (128) → LSTM Layers → Dense Layers → 1 output

LSTM Layers:
├── LSTM Layer 1: 128 hidden units, Dropout(0.2)
└── LSTM Layer 2: 128 hidden units, Dropout(0.2)
```

### Biological Integration
```
LSTM Output (128) + GC Content (1) = 129 features → Dense Layers → 1 output

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
| **Learning Rate** | 0.0006 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 82 | Up to 200 | Early Stopping |

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
| Fold 1 | 0.8542 | 0.0114 |
| Fold 2 | 0.8571 | 0.0111 |
| Fold 3 | 0.8558 | 0.0113 |
| Fold 4 | 0.8567 | 0.0112 |
| Fold 5 | 0.8559 | 0.0113 |
| **Mean** | **0.8559** | **0.0112** |
| **Std** | **0.0011** | **0.0001** |

### Performance by Metric
- **Spearman Correlation**: 0.856 (95% CI: 0.853-0.859)
- **Mean Squared Error**: 0.0112
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 15GB
- **Platform**: Digital Research Alliance of Canada

## Biological Integration

### GC Content Feature
- **Calculation**: (Count(G) + Count(C)) / 21
- **Range**: 0.0 to 1.0 (continuous)
- **Optimal Range**: 0.4 to 0.6 for CRISPR efficiency
- **Integration Method**: Concatenated with LSTM features
- **Impact**: +0.019 improvement over base LSTM

### Biological Relevance
- **Memory Enhancement**: GC content preserved in cell state
- **Long-term Dependencies**: Biological features maintained through time
- **Sequence Stability**: Thermodynamic information retention

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the LSTM+GC model
model = ChromeCRISPRModel.load_from_file('models/base_models_with_gc/LSTM+GC.pth')
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
- **Input**: 21-mer DNA sequence (84 features) + GC content (1 feature)
- **LSTM Branch**: 2 LSTM layers (128 hidden each) → 128 features
- **Biological Fusion**: LSTM features + GC content → 129 features
- **Dense**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 1.17M

### Key Features
1. **Cell State**: Long-term memory preservation
2. **Three Gates**: Input, forget, and output gates
3. **GC Integration**: Biological features in memory
4. **Gradient Flow**: Enhanced long-term dependency modeling

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/LSTM+GC_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**LSTM+GC integrates biological features with long-term memory modeling, achieving 0.856 Spearman correlation with substantial improvement over base LSTM.**
