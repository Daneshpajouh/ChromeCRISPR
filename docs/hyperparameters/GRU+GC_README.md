# GRU+GC Model: GRU with GC Content Integration

**Performance**: 0.840 Spearman correlation | **Rank**: #10 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | GRU+GC |
| **Architecture Type** | Gated Recurrent Unit + GC Content |
| **Spearman Correlation** | 0.840 |
| **Mean Squared Error** | 0.0122 |
| **Total Parameters** | 907,526 |
| **Training Time** | 3.3 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **GC Content**: 1 feature (calculated as (G+C)/21)
- **Total Input Features**: 85

### GRU Architecture
```
Input (85 features) → Embedding (128) → GRU Layers → Dense Layers → 1 output

GRU Layers:
├── GRU Layer 1: 128 hidden units, Dropout(0.2)
└── GRU Layer 2: 128 hidden units, Dropout(0.2)
```

### Biological Integration
```
GRU Output (128) + GC Content (1) = 129 features → Dense Layers → 1 output

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
| **Learning Rate** | 0.0004 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 89 | Up to 200 | Early Stopping |

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
| Fold 1 | 0.8387 | 0.0124 |
| Fold 2 | 0.8412 | 0.0121 |
| Fold 3 | 0.8398 | 0.0122 |
| Fold 4 | 0.8405 | 0.0121 |
| Fold 5 | 0.8399 | 0.0123 |
| **Mean** | **0.8400** | **0.0122** |
| **Std** | **0.0009** | **0.0001** |

### Performance by Metric
- **Spearman Correlation**: 0.840 (95% CI: 0.837-0.843)
- **Mean Squared Error**: 0.0122
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 13GB
- **Platform**: Digital Research Alliance of Canada

## Biological Integration

### GC Content Feature
- **Calculation**: (Count(G) + Count(C)) / 21
- **Range**: 0.0 to 1.0 (continuous)
- **Optimal Range**: 0.4 to 0.6 for CRISPR efficiency
- **Integration Method**: Concatenated with GRU features
- **Impact**: +0.003 improvement over base GRU

### Biological Relevance
- **Sequence Thermodynamics**: GC content affects binding stability
- **CRISPR Efficiency**: Influences guide RNA performance
- **Molecular Interactions**: Affects Cas9-gRNA complex formation

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the GRU+GC model
model = ChromeCRISPRModel.load_from_file('models/base_models_with_gc/GRU+GC.pth')
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
- **GRU Branch**: 2 GRU layers (128 hidden each) → 128 features
- **Biological Fusion**: GRU features + GC content → 129 features
- **Dense**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 907K

### Key Features
1. **Gated Memory**: Update and reset gates control information flow
2. **GC Integration**: Biological feature enhancement
3. **Efficient Training**: Fewer parameters than LSTM
4. **Sequence Context**: Captures long-range dependencies

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/GRU+GC_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**GRU+GC combines efficient sequence modeling with biological features, achieving 0.840 Spearman correlation with measurable improvement over base GRU.**
