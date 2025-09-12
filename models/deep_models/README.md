# Deep Models (4 Models)

Enhanced neural network architectures with additional layers for deeper feature extraction and improved CRISPR/Cas9 predictions.

##  Models Overview

| Model | File | Spearman | MSE | Parameters | Layers | Training Time |
|-------|------|----------|-----|------------|--------|---------------|
| **deepCNN** | `deepCNN.pth` | 0.869 | 0.0098 | 665K | 4 | 6.2h |
| **deepGRU** | `deepGRU.pth` | 0.868 | 0.0099 | 1.81M | 4 | 8.4h |
| **deepBiLSTM** | `deepBiLSTM.pth` | 0.862 | 0.0104 | 5.99M | 4 | 11.2h |
| **deepLSTM** | `deepLSTM.pth` | 0.862 | 0.0103 | 2.34M | 4 | 9.1h |

## 🏗️ Model Architectures

### deepCNN (4-Layer Deep Convolutional Network)
```python
Architecture:
├── Input: 21-mer sequence (84 features)
├── Embedding: 84 → 128 dimensions
├── Conv1D Layer 1: 128 filters, kernel=5, ReLU, BatchNorm, Dropout(0.15)
├── Conv1D Layer 2: 128 filters, kernel=5, ReLU, BatchNorm, Dropout(0.15)
├── Conv1D Layer 3: 128 filters, kernel=5, ReLU, BatchNorm, Dropout(0.15)
├── Conv1D Layer 4: 128 filters, kernel=5, ReLU, BatchNorm, Dropout(0.15)
├── Global Max Pooling
├── Dense Layer 1: 256 units, ReLU, Dropout(0.2)
├── Dense Layer 2: 128 units, ReLU, Dropout(0.2)
├── Dense Layer 3: 64 units, ReLU, Dropout(0.2)
├── Dense Layer 4: 32 units, ReLU, Dropout(0.2)
└── Output: 1 unit (linear)
```

**Key Features:**
- 4 convolutional layers for hierarchical feature extraction
- Progressive feature refinement through multiple conv layers
- 4 dense layers for complex pattern learning
- Batch normalization for training stability

### deepGRU (4-Layer Deep Gated Recurrent Unit)
```python
Architecture:
├── Input: 21-mer sequence (84 features)
├── Embedding: 84 → 128 dimensions
├── GRU Layer 1: 256 hidden, Dropout(0.18)
├── GRU Layer 2: 256 hidden, Dropout(0.18)
├── GRU Layer 3: 256 hidden, Dropout(0.18)
├── GRU Layer 4: 256 hidden, Dropout(0.18)
├── Dense Layer 1: 256 units, ReLU, Dropout(0.25)
├── Dense Layer 2: 128 units, ReLU, Dropout(0.25)
├── Dense Layer 3: 64 units, ReLU, Dropout(0.25)
├── Dense Layer 4: 32 units, ReLU, Dropout(0.25)
└── Output: 1 unit (linear)
```

**Key Features:**
- 4 GRU layers with 256 hidden units each
- Deep temporal hierarchy for sequence processing
- Efficient parameter usage compared to LSTM
- Gated mechanism for long-range dependencies

### deepLSTM (4-Layer Deep Long Short-Term Memory)
```python
Architecture:
├── Input: 21-mer sequence (84 features)
├── Embedding: 84 → 128 dimensions
├── LSTM Layer 1: 256 hidden, Dropout(0.18)
├── LSTM Layer 2: 256 hidden, Dropout(0.18)
├── LSTM Layer 3: 256 hidden, Dropout(0.18)
├── LSTM Layer 4: 256 hidden, Dropout(0.18)
├── Dense Layer 1: 256 units, ReLU, Dropout(0.25)
├── Dense Layer 2: 128 units, ReLU, Dropout(0.25)
├── Dense Layer 3: 64 units, ReLU, Dropout(0.25)
├── Dense Layer 4: 32 units, ReLU, Dropout(0.25)
└── Output: 1 unit (linear)
```

**Key Features:**
- 4 LSTM layers with 256 hidden units each
- Cell state preservation through deep layers
- Complex gating for information control
- Memory mechanism for long-term dependencies

### deepBiLSTM (4-Layer Deep Bidirectional LSTM)
```python
Architecture:
├── Input: 21-mer sequence (84 features)
├── Embedding: 84 → 128 dimensions
├── BiLSTM Layer 1: 256 hidden (128×2), Dropout(0.18)
├── BiLSTM Layer 2: 256 hidden (128×2), Dropout(0.18)
├── BiLSTM Layer 3: 256 hidden (128×2), Dropout(0.18)
├── BiLSTM Layer 4: 256 hidden (128×2), Dropout(0.18)
├── Dense Layer 1: 512 units, ReLU, Dropout(0.25)
├── Dense Layer 2: 256 units, ReLU, Dropout(0.25)
├── Dense Layer 3: 128 units, ReLU, Dropout(0.25)
├── Dense Layer 4: 64 units, ReLU, Dropout(0.25)
└── Output: 1 unit (linear)
```

**Key Features:**
- 4 bidirectional LSTM layers (512 total hidden per layer)
- Forward and backward processing for full context
- Enhanced sequence understanding
- Comprehensive temporal feature extraction

##  Performance Analysis

### Deep Architecture Benefits
```
deepCNN   (0.869) ━━ Best parameter efficiency
deepGRU   (0.868) ━━ Strong performance with efficiency
deepLSTM  (0.862) ━━ Good memory capability
deepBiLSTM (0.862) ━━ Comprehensive context modeling
```

### Performance Improvements Over Base Models
- **CNN**: Base (0.793) → Deep (0.869) = **+0.076 improvement**
- **GRU**: Base (0.837) → Deep (0.868) = **+0.031 improvement**
- **LSTM**: Base (0.837) → Deep (0.862) = **+0.025 improvement**
- **BiLSTM**: Base (0.843) → Deep (0.862) = **+0.019 improvement**

##  Technical Specifications

### Layer Configuration
- **Convolutional Layers**: 4 × 128 filters, 5×5 kernels
- **Recurrent Layers**: 4 × 256 hidden units (128×2 for bidirectional)
- **Dense Layers**: 4 × progressive reduction (256→128→64→32)
- **Dropout**: 0.15-0.18 in recurrent layers, 0.25 in dense layers

### Training Configuration
- **Batch Size**: 64 (optimized for GPU memory)
- **Learning Rate**: 0.00005 - 0.00009 (Bayesian optimized)
- **Epochs**: 95 - 145 (early stopping with patience=10)
- **Optimizer**: Adam with weight decay (2×10^-5)

### Hardware Requirements
- **GPU**: NVIDIA V100 Volta (32GB HBM2)
- **Memory**: 14GB - 28GB depending on model
- **Training Time**: 6.2h - 11.2h per model

##  Model Files & Access

### Direct File Access
```bash
# Deep model files
deepCNN.pth     # Most efficient (0.869, 665K params)
deepGRU.pth     # Best balance (0.868, 1.81M params)
deepLSTM.pth    # Memory focus (0.862, 2.34M params)
deepBiLSTM.pth  # Context focus (0.862, 5.99M params)
```

### Loading Deep Models
```python
import torch

# Load high-performance deep model
model = torch.load('deepCNN.pth')  # Best efficiency
model.eval()

# For batch processing
batch_size = 64
with torch.no_grad():
    for batch in data_loader:
        predictions = model(batch)
        # Process predictions
```

##  Related Documentation

- **[Complete Hyperparameters](../../docs/hyperparameters/)** - Full deep model specifications
- **[Training Procedures](../../docs/training_procedures/)** - Deep architecture training protocols
- **[Model Architectures](../../docs/MODEL_ARCHITECTURES.md)** - Technical deep layer details
- **[Performance Analysis](../../docs/COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative deep model analysis

##  Key Insights

### Architecture Performance Hierarchy
1. **deepCNN**: Best performance per parameter (0.869 with 665K params)
2. **deepGRU**: Excellent balance of performance and efficiency
3. **deepLSTM/deepBiLSTM**: Strong sequence modeling capabilities
4. **All Deep Models**: Significant improvements over base architectures

### Deep Learning Benefits
- **Hierarchical Features**: Multiple layers capture increasingly complex patterns
- **Parameter Efficiency**: Deep CNN achieves high performance with few parameters
- **Regularization**: Dropout and batch norm prevent overfitting
- **Gradient Flow**: Proper depth enables better optimization

### Practical Considerations
- **Training Time**: Deep models require more training time
- **Memory Usage**: Larger models need more GPU memory
- **Convergence**: Deeper models may need more careful optimization
- **Performance Gain**: Generally worth the increased complexity

##  Usage Recommendations

### For Different Use Cases
- **Best Efficiency**: Use deepCNN (high performance, low parameters)
- **Balanced Approach**: Use deepGRU (good performance, reasonable complexity)
- **Memory Focus**: Use deepLSTM (strong long-term dependencies)
- **Context Focus**: Use deepBiLSTM (comprehensive sequence understanding)

### Research Applications
- **Compare depth effects**: Test base vs deep versions of same architecture
- **Efficiency studies**: Use deepCNN as parameter-efficient baseline
- **Memory research**: Use deepLSTM for long-range dependency analysis
- **Context analysis**: Use deepBiLSTM for comprehensive sequence modeling

---
**Note**: Deep models generally provide significant performance improvements over base architectures, with deepCNN offering the best trade-off between performance and computational efficiency.
