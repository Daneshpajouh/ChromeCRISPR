# deepLSTM+GC Model: Deep LSTM with GC Content Integration

**Performance**: 0.860 Spearman correlation | **Rank**: #9 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | deepLSTM+GC |
| **Architecture Type** | Deep Long Short-Term Memory + GC Content |
| **Spearman Correlation** | 0.860 |
| **Mean Squared Error** | 0.0104 |
| **Total Parameters** | 2,343,561 |
| **Training Time** | 9.6 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **GC Content**: 1 feature (calculated as (G+C)/21)
- **Total Input Features**: 85

### Deep LSTM Architecture
```
Input (85 features) → Embedding (128) → Deep LSTM Layers → Dense Layers → 1 output

LSTM Layers:
├── LSTM Layer 1: 256 hidden units, Dropout(0.2)
├── LSTM Layer 2: 256 hidden units, Dropout(0.2)
├── LSTM Layer 3: 256 hidden units, Dropout(0.2)
└── LSTM Layer 4: 256 hidden units, Dropout(0.2)
```

### Biological Integration
```
LSTM Output (256) + GC Content (1) = 257 features → Dense Layers → 1 output

Dense Layers:
├── Dense Layer 1: 128 units, ReLU, Dropout(0.2)
├── Dense Layer 2: 64 units, ReLU, Dropout(0.2)
└── Output Layer: 1 unit, Linear activation
```

## Optimal Hyperparameters

### Training Configuration
| Parameter | Optimal Value | Search Range | Tuning Method |
|-----------|---------------|--------------|----------------|
| **Learning Rate** | 0.00008 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 75 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **LSTM Hidden Size** | All layers | 256 |
| **LSTM Layers** | Count | 4 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.858 | 0.0106 |
| Fold 2 | 0.862 | 0.0102 |
| Fold 3 | 0.859 | 0.0105 |
| Fold 4 | 0.861 | 0.0103 |
| Fold 5 | 0.860 | 0.0104 |
| **Mean** | **0.860** | **0.0104** |
| **Std** | **0.002** | **0.0002** |

### Performance by Metric
- **Spearman Correlation**: 0.860 (95% CI: 0.857-0.863)
- **Mean Squared Error**: 0.0104
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 26GB
- **Platform**: Digital Research Alliance of Canada

## Biological Integration

### GC Content Feature
- **Calculation**: (Count(G) + Count(C)) / 21
- **Range**: 0.0 to 1.0 (continuous)
- **Optimal Range**: 0.4 to 0.6 for CRISPR efficiency
- **Integration Method**: Concatenated with LSTM features
- **Impact**: Enhanced memory preservation of biological features

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the deepLSTM+GC model
model = ChromeCRISPRModel.load_from_file('models/deep_models_with_gc/deepLSTM+GC.pth')
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
- **Deep LSTM Branch**: 4 LSTM layers (256 hidden each) → 256 features
- **Biological Fusion**: LSTM features + GC content → 257 features
- **Dense**: 2 layers (128→64) → 1 output
- **Total Parameters**: 2.34M

### Key Features
1. **Deep Architecture**: 4-layer hierarchical processing
2. **Cell State Memory**: Long-term biological feature preservation
3. **Three Gates**: Enhanced control of information flow
4. **GC Integration**: Biological context in memory cells

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/deepLSTM+GC_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**deepLSTM+GC provides deep memory processing with biological integration, achieving 0.860 Spearman correlation with enhanced long-term dependency modeling.**
