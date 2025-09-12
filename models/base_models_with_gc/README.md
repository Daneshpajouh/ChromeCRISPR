# Base Models + GC Content (4 Models)

Base neural network architectures enhanced with biological GC content integration for improved CRISPR/Cas9 predictions.

##  Models Overview

| Model | File | Spearman | MSE | Parameters | GC Impact | Training Time |
|-------|------|----------|-----|------------|-----------|---------------|
| **LSTM+GC** | `LSTM+GC.pth` | 0.856 | 0.0112 | 1.17M | +0.019 | 3.9h |
| **BiLSTM+GC** | `BiLSTM+GC.pth` | 0.855 | 0.0110 | 2.82M | +0.012 | 4.7h |
| **GRU+GC** | `GRU+GC.pth` | 0.840 | 0.0122 | 907K | +0.003 | 3.3h |
| **CNN+GC** | `CNN+GC.pth` | 0.781 | 0.0170 | 332K | -0.012 | 2.6h |

## 🏗️ Model Architectures

### LSTM+GC (Long Short-Term Memory + GC Content)
```python
Architecture:
├── Input: 21-mer sequence (84 features) + GC (1 feature) = 85 total
├── Embedding: 84 → 128 dimensions
├── LSTM Layer 1: 128 hidden, Dropout(0.2)
├── LSTM Layer 2: 128 hidden, Dropout(0.2)
├── Dense: 64 units, ReLU, Dropout(0.2)
├── Dense: 32 units, ReLU, Dropout(0.2)
├── GC Integration: Concatenated with final LSTM output
└── Output: 1 unit (linear)
```

**GC Integration:** Concatenated with final hidden state before dense layers

### BiLSTM+GC (Bidirectional LSTM + GC Content)
```python
Architecture:
├── Input: 21-mer sequence (84 features) + GC (1 feature) = 85 total
├── Embedding: 84 → 128 dimensions
├── BiLSTM Layer 1: 128 hidden × 2, Dropout(0.2)
├── BiLSTM Layer 2: 128 hidden × 2, Dropout(0.2)
├── Dense: 128 units, ReLU, Dropout(0.2)
├── Dense: 64 units, ReLU, Dropout(0.2)
├── GC Integration: Concatenated with final BiLSTM output
└── Output: 1 unit (linear)
```

**GC Integration:** Concatenated with bidirectional hidden state (256 features + 1 GC = 257 total)

### GRU+GC (Gated Recurrent Unit + GC Content)
```python
Architecture:
├── Input: 21-mer sequence (84 features) + GC (1 feature) = 85 total
├── Embedding: 84 → 128 dimensions
├── GRU Layer 1: 128 hidden, Dropout(0.2)
├── GRU Layer 2: 128 hidden, Dropout(0.2)
├── Dense: 64 units, ReLU, Dropout(0.2)
├── Dense: 32 units, ReLU, Dropout(0.2)
├── GC Integration: Concatenated with final GRU output
└── Output: 1 unit (linear)
```

**GC Integration:** Concatenated with final hidden state before dense layers

### CNN+GC (Convolutional Neural Network + GC Content)
```python
Architecture:
├── Input: 21-mer sequence (84 features) + GC (1 feature) = 85 total
├── Embedding: 84 → 128 dimensions
├── Conv1D: 128 filters, kernel=5, ReLU
├── Conv1D: 128 filters, kernel=5, ReLU
├── Global Max Pooling
├── Dense: 64 units, ReLU, Dropout(0.2)
├── Dense: 32 units, ReLU, Dropout(0.2)
├── GC Integration: Concatenated after global pooling
└── Output: 1 unit (linear)
```

**GC Integration:** Concatenated with pooled convolutional features (128 + 1 = 129 total)

##  Performance Analysis

### GC Content Impact by Architecture
```
LSTM+GC    (+0.019) ━━ Largest improvement
BiLSTM+GC  (+0.012) ━━ Significant bidirectional benefit
GRU+GC     (+0.003) ━━ Minimal improvement
CNN+GC     (-0.012) ━━ Slight performance decrease
```

### Best Performers with GC
1. **LSTM+GC** (0.856) - Best overall performance
2. **BiLSTM+GC** (0.855) - Strong bidirectional context
3. **GRU+GC** (0.840) - Efficient with minor improvement
4. **CNN+GC** (0.781) - Performance decrease

##  Biological GC Content Integration

### GC Content Calculation
```python
GC_content = (count_G + count_C) / sequence_length
# Range: 0.0 to 1.0
# Optimal range: 0.4 to 0.6 for CRISPR efficiency
```

### Integration Methods
- **LSTM/GRU**: Concatenated with final hidden state (128 + 1 = 129 features)
- **BiLSTM**: Concatenated with bidirectional output (256 + 1 = 257 features)
- **CNN**: Concatenated with global pooled features (128 + 1 = 129 features)

### Biological Relevance
- **Thermodynamic stability**: GC content affects DNA duplex stability
- **CRISPR binding**: Influences Cas9-gRNA complex formation
- **Off-target potential**: GC-rich regions may have different binding characteristics

##  Model Files & Access

### Direct File Access
```bash
# Model files with GC integration
LSTM+GC.pth     # Best performing (0.856)
BiLSTM+GC.pth   # Strong bidirectional (0.855)
GRU+GC.pth      # Efficient baseline (0.840)
CNN+GC.pth      # Pattern detection (0.781)
```

### Loading Models with GC Features
```python
import torch
import numpy as np

# Load model
model = torch.load('LSTM+GC.pth')
model.eval()

# Prepare input with GC content
sequence_features = one_hot_encode(sequence)  # 84 features
gc_content = calculate_gc_content(sequence)   # 1 feature
input_features = np.concatenate([sequence_features, [gc_content]])

# Make prediction
with torch.no_grad():
    prediction = model(torch.tensor(input_features, dtype=torch.float32))
```

##  Related Documentation

- **[Complete Hyperparameters](../../docs/hyperparameters/)** - Full GC integration specifications
- **[Training Procedures](../../docs/training_procedures/)** - GC feature preprocessing
- **[Biological Features](../../docs/MODEL_ARCHITECTURES.md)** - GC content integration details
- **[Performance Comparison](../../docs/COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - GC impact analysis

##  Key Insights

### Architecture-Specific GC Effects
1. **RNN Superiority**: LSTM and BiLSTM show significant GC benefit (+1.9%, +1.2%)
2. **CNN Challenge**: Convolutional features may conflict with global GC information
3. **Bidirectional Advantage**: BiLSTM effectively utilizes GC context from both directions
4. **GRU Neutral**: Minimal impact, maintains efficiency advantage

### Performance vs Complexity Trade-off
- **LSTM+GC**: Best performance (0.856) with moderate complexity
- **BiLSTM+GC**: Strong performance (0.855) with higher complexity
- **GRU+GC**: Good efficiency (0.840) with low complexity
- **CNN+GC**: Specialized patterns (0.781) with lowest complexity

##  Usage Recommendations

### For Biological Accuracy
- **Use LSTM+GC or BiLSTM+GC** for best performance with biological features
- **Consider GRU+GC** for efficient biological modeling
- **Evaluate CNN+GC** carefully due to performance decrease

### For Research Applications
- **Compare with/without GC** to quantify biological feature importance
- **Use LSTM+GC** as baseline for biological feature integration
- **Consider BiLSTM+GC** for comprehensive sequence context

---
**Note**: GC content integration provides domain-specific biological information. Performance improvements vary by architecture, with RNN models showing the most benefit.
