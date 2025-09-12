# CNN_LSTM+GC Model: Balanced Hybrid Performance

**Performance**: 0.867 Spearman correlation | **Rank**: #4 of 20 models | **Balance**: Good performance with moderate complexity

##  Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | CNN_LSTM+GC |
| **Architecture Type** | Hybrid CNN-LSTM with GC Content Integration |
| **Spearman Correlation** | 0.867 |
| **Mean Squared Error** | 0.0115 |
| **Total Parameters** | 7,994,885 |
| **Training Time** | 4.8 hours |
| **Strength** |  Good balance of performance and complexity |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **GC Content Feature**: 1 (calculated as (G+C)/sequence_length)
- **Total Input Features**: 85

### CNN Branch (Motif Detection)
```
Input (85 features) → Embedding (128) → Conv Layers → Global Max Pooling → 64 features

Convolutional Layers:
├── Conv1D Layer 1: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.142)
├── Conv1D Layer 2: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.142)
└── Conv1D Layer 3: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.142)
```

### LSTM Branch (Sequence Memory)
```
Input (85 features) → Embedding (128) → LSTM Layers → 384 features

LSTM Layers:
├── LSTM Layer 1: 384 hidden, Dropout(0.142)
├── LSTM Layer 2: 384 hidden, Dropout(0.142)
└── LSTM Layer 3: 384 hidden, Dropout(0.142)
```

### Fusion and Output
```
Fusion: CNN (64) + LSTM (384) + GC (1) = 449 features → Dense Layers

Dense Layers:
├── Dense Layer 1: 128 units, ReLU, Dropout(0.142)
├── Dense Layer 2: 64 units, ReLU, Dropout(0.142)
├── Dense Layer 3: 32 units, ReLU, Dropout(0.142)
└── Output Layer: 1 unit, Linear activation
```

##  Optimal Hyperparameters

### Training Configuration
| Parameter | Optimal Value | Search Range | Notes |
|-----------|---------------|--------------|-------|
| **Learning Rate** | 0.000209 | 1e-5 to 1e-2 | Similar to CNN_GRU+GC |
| **Batch Size** | 64 | 32-128 | Standard optimization |
| **Dropout Rate** | 0.142 | 0.1-0.5 | Consistent with other hybrids |
| **Weight Decay** | 2.15e-05 | Auto-tuned | Moderate regularization |
| **Epochs** | 92 | Up to 200 | Good convergence |

### Architecture Parameters
| Component | Parameter | Value | Notes |
|-----------|-----------|-------|-------|
| **CNN Filters** | All layers | 64 | Consistent across hybrids |
| **CNN Kernel Size** | All layers | 5 | Optimal motif detection |
| **LSTM Hidden Size** | All layers | 384 | Same as other hybrids |
| **Dense Units** | Layer 1 | 128 | Standard configuration |
| **Dense Units** | Layer 2 | 64 | Feature refinement |
| **Dense Units** | Layer 3 | 32 | Final representation |

##  Performance Analysis

### Benchmark Comparison
```
CNN_LSTM+GC (0.867) ━━ Fourth Best Overall
CNN_GRU+GC (0.876) ━━  Best (+0.009 difference)
deepCNN+GC (0.873) ━━ Second (+0.006 difference)
CNN_BiLSTM+GC (0.870) ━━ Third (+0.003 difference)
DeepHF (0.867) ━━ Tied with previous SOTA
```

### Hybrid Architecture Comparison
```
LSTM-based Hybrids:
CNN_GRU+GC (0.876) ━━  GRU superior to LSTM
CNN_LSTM+GC (0.867) ━━ Good LSTM performance
CNN_BiLSTM+GC (0.870) ━━ BiLSTM better than LSTM

Key Insight: GRU gates are more effective than LSTM gates in hybrid architectures
```

### Cross-Validation Results
| Fold | Spearman | MSE | Notes |
|------|----------|-----|-------|
| Fold 1 | 0.8661 | 0.0116 | Consistent performance |
| Fold 2 | 0.8672 | 0.0115 | Best fold |
| Fold 3 | 0.8665 | 0.0115 | Stable results |
| Fold 4 | 0.8671 | 0.0115 | Reliable prediction |
| Fold 5 | 0.8668 | 0.0115 | Good generalization |
| **Mean** | **0.8667** | **0.0115** | **Solid performance** |
| **Std** | **0.0004** | **0.0001** | **Very stable** |

## 🖥️ Training Details

### Hardware Configuration
- **GPU**: NVIDIA V100 Volta (32GB HBM2)
- **Memory Usage**: 29GB
- **Training Time**: 4.8 hours
- **Platform**: Digital Research Alliance of Canada

### Training Dynamics
- **Convergence**: Steady improvement over 92 epochs
- **Early Stopping**: Effective convergence monitoring
- **Learning Stability**: Consistent loss reduction
- **Generalization**: Good validation performance
- **Optimizer**: Adam (β1=0.9, β2=0.999, weight_decay=2.15e-05)
- **Loss Function**: Mean Squared Error

##  Biological Integration

### GC Content Feature
- **Calculation**: (Count(G) + Count(C)) / 21
- **Integration**: Concatenated with hybrid features
- **Impact**: Maintains performance level of base hybrid
- **Biological Relevance**: Thermodynamic stability information

### LSTM Memory Integration
- **Cell State**: Long-term memory preservation
- **GC Influence**: Biological features affect memory retention
- **Sequence Context**: Enhanced with biological information
- **Temporal Integration**: Biological features in temporal processing

##  Comparative Analysis

### Performance vs Complexity
```
Performance per Million Parameters:
CNN_GRU+GC: 0.876 / 6.12M = 0.143  Best efficiency
CNN_LSTM+GC: 0.867 / 7.99M = 0.108  Good balance
CNN_BiLSTM+GC: 0.870 / 20.4M = 0.043  Lower efficiency

CNN_LSTM+GC offers good performance with reasonable computational requirements
```

### Architecture Insights
- **GRU Advantage**: Simpler gates perform better in hybrid setting
- **LSTM Complexity**: More parameters don't translate to better performance
- **Bidirectional Benefit**: BiLSTM provides better context than unidirectional LSTM
- **Hybrid Synergy**: CNN + RNN combination consistently outperforms individual models

##  Usage Examples

### Load and Use Model
```python
import torch
from src.models import CNN_LSTM_GC

# Load the balanced hybrid model
model = CNN_LSTM_GC.load_from_file('models/chromecrispr_hybrid_models/CNN_LSTM+GC.pth')
model.eval()

# Process sequence
sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCG"
input_tensor = model.preprocess_with_gc(sequence)

# Make prediction
with torch.no_grad():
    prediction = model(input_tensor)
    print(f"Hybrid LSTM prediction: {prediction.item():.4f}")
```

### Training Configuration
```python
# Optimal setup for CNN_LSTM+GC
config = {
    'architecture': 'cnn_lstm_gc',
    'learning_rate': 0.000209,
    'batch_size': 64,
    'dropout_rate': 0.142,
    'weight_decay': 2.15e-05,
    'epochs': 100,
    'cnn_filters': 64,
    'lstm_hidden': 384,
    'dense_units': [128, 64, 32]
}

trainer = Trainer(config)
model = trainer.train('cnn_lstm_gc', X_train, y_train)
```

##  Model Specifications

### Architecture Summary
- **Input**: 21-mer DNA + GC content (85 features)
- **CNN Branch**: 3 conv layers (64 filters) → 64 features
- **LSTM Branch**: 3 LSTM layers (384 hidden) → 384 features
- **Fusion**: Concatenation → 449 features
- **Dense Network**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 7.99M

### Key Features
1. **Hybrid Design**: CNN for motifs + LSTM for memory
2. **GC Integration**: Biological feature enhancement
3. **Balanced Complexity**: Good performance-to-resource ratio
4. **Stable Training**: Reliable convergence characteristics

### Performance Highlights
- **Strong Performer**: Fourth best overall (0.867 Spearman)
- **Balanced Approach**: Good performance with moderate complexity
- **Consistent Results**: Stable across cross-validation folds
- **Biological Aware**: Incorporates domain-specific features

##  Documentation Links

- **[Raw JSON Specs](../hyperparameters/CNN_LSTM+GC_hyperparameters.json)** - Complete technical details
- **[Training Procedures](../training_procedures/)** - Hybrid training protocols
- **[Architecture Details](../MODEL_ARCHITECTURES.md)** - Technical specifications
- **[Performance Comparison](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
** CNN_LSTM+GC provides a solid balance of performance and computational efficiency, making it a reliable choice for CRISPR prediction tasks.**
