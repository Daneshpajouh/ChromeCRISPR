# deepGRU+GC Model: Deep GRU with GC Content Integration

**Performance**: 0.867 Spearman correlation | **Rank**: #5 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | deepGRU+GC |
| **Architecture Type** | Deep Gated Recurrent Unit + GC Content |
| **Spearman Correlation** | 0.867 |
| **Mean Squared Error** | 0.0098 |
| **Total Parameters** | 1,820,747 |
| **Training Time** | 8.9 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **GC Content**: 1 feature (calculated as (G+C)/21)
- **Total Input Features**: 85

### Deep GRU Architecture
```
Input (85 features) → Embedding (128) → Deep GRU Layers → Dense Layers → 1 output

GRU Layers:
├── GRU Layer 1: 256 hidden units, Dropout(0.2)
├── GRU Layer 2: 256 hidden units, Dropout(0.2)
├── GRU Layer 3: 256 hidden units, Dropout(0.2)
└── GRU Layer 4: 256 hidden units, Dropout(0.2)
```

### Biological Integration
```
GRU Output (256) + GC Content (1) = 257 features → Dense Layers → 1 output

Dense Layers:
├── Dense Layer 1: 128 units, ReLU, Dropout(0.2)
├── Dense Layer 2: 64 units, ReLU, Dropout(0.2)
└── Output Layer: 1 unit, Linear activation
```

## Optimal Hyperparameters

### Training Configuration
| Parameter | Optimal Value | Search Range | Tuning Method |
|-----------|---------------|--------------|----------------|
| **Learning Rate** | 0.00011 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 79 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **GRU Hidden Size** | All layers | 256 |
| **GRU Layers** | Count | 4 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.865 | 0.0100 |
| Fold 2 | 0.869 | 0.0096 |
| Fold 3 | 0.866 | 0.0099 |
| Fold 4 | 0.868 | 0.0097 |
| Fold 5 | 0.867 | 0.0098 |
| **Mean** | **0.867** | **0.0098** |
| **Std** | **0.002** | **0.0002** |

### Performance by Metric
- **Spearman Correlation**: 0.867 (95% CI: 0.864-0.870)
- **Mean Squared Error**: 0.0098
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 24GB
- **Platform**: Digital Research Alliance of Canada

## Biological Integration

### GC Content Feature
- **Calculation**: (Count(G) + Count(C)) / 21
- **Range**: 0.0 to 1.0 (continuous)
- **Optimal Range**: 0.4 to 0.6 for CRISPR efficiency
- **Integration Method**: Concatenated with GRU features
- **Impact**: Consistent improvement over base deepGRU

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the deepGRU+GC model
model = ChromeCRISPRModel.load_from_file('models/deep_models_with_gc/deepGRU+GC.pth')
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
- **Deep GRU Branch**: 4 GRU layers (256 hidden each) → 256 features
- **Biological Fusion**: GRU features + GC content → 257 features
- **Dense**: 2 layers (128→64) → 1 output
- **Total Parameters**: 1.82M

### Key Features
1. **Deep Architecture**: 4-layer hierarchical processing
2. **GC Integration**: Biological feature enhancement
3. **Efficient Computation**: Fewer parameters than LSTM
4. **Stable Gradients**: Better training convergence

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/deepGRU+GC_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**deepGRU+GC provides deep hierarchical processing with biological integration, achieving 0.867 Spearman correlation with enhanced sequence understanding.**
