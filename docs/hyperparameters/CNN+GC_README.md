# CNN+GC Model: CNN with GC Content Integration

**Performance**: 0.781 Spearman correlation | **Rank**: #20 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | CNN+GC |
| **Architecture Type** | Convolutional Neural Network + GC Content |
| **Spearman Correlation** | 0.781 |
| **Mean Squared Error** | 0.0170 |
| **Total Parameters** | 125,433 |
| **Training Time** | 2.6 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **GC Content**: 1 feature (calculated as (G+C)/21)
- **Total Input Features**: 85

### CNN Architecture
```
Input (85 features) → Embedding (128) → Conv Layers → Global Max Pooling → 64 features
```

### Biological Integration
```
CNN Output (64) + GC Content (1) = 65 features → Dense Layers → 1 output

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
| **Learning Rate** | 0.0007 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 92 | Up to 200 | Early Stopping |

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
| Fold 1 | 0.7785 | 0.0173 |
| Fold 2 | 0.7824 | 0.0168 |
| Fold 3 | 0.7801 | 0.0171 |
| Fold 4 | 0.7818 | 0.0169 |
| Fold 5 | 0.7809 | 0.0170 |
| **Mean** | **0.7807** | **0.0170** |
| **Std** | **0.0015** | **0.0002** |

### Performance by Metric
- **Spearman Correlation**: 0.781 (95% CI: 0.777-0.785)
- **Mean Squared Error**: 0.0170
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 9GB
- **Platform**: Digital Research Alliance of Canada

## Biological Integration

### GC Content Feature
- **Calculation**: (Count(G) + Count(C)) / 21
- **Range**: 0.0 to 1.0 (continuous)
- **Optimal Range**: 0.4 to 0.6 for CRISPR efficiency
- **Integration Method**: Concatenated with CNN features
- **Impact**: Limited improvement over base CNN

### Biological Relevance
- **Thermodynamic Stability**: GC content affects DNA duplex stability
- **CRISPR Binding**: Influences Cas9-gRNA complex formation
- **Sequence Context**: Provides biophysical sequence information

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the CNN+GC model
model = ChromeCRISPRModel.load_from_file('models/base_models_with_gc/CNN+GC.pth')
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
- **CNN Branch**: 2 conv layers (64 filters each) → 64 features
- **Biological Fusion**: CNN features + GC content → 65 features
- **Dense**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 125K

### Key Features
1. **Convolutional Filters**: Capture sequence motifs and patterns
2. **GC Content Integration**: Biological feature enhancement
3. **Feature Fusion**: Combined learned and biological features
4. **Regularization**: Dropout prevents overfitting

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/CNN+GC_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**CNN+GC integrates biological features with convolutional motif detection, achieving 0.781 Spearman correlation with limited improvement over base CNN.**
