# Base Models (4 Models)

Standard implementations of individual neural network architectures for CRISPR/Cas9 on-target prediction.

##  Models Overview

| Model | File | Spearman | MSE | Parameters | Training Time |
|-------|------|----------|-----|------------|---------------|
| **CNN** | `CNN.pth` | 0.793 | 0.0161 | 332K | 2.5h |
| **GRU** | `GRU.pth` | 0.837 | 0.0121 | 907K | 3.2h |
| **LSTM** | `LSTM.pth` | 0.837 | 0.0122 | 1.17M | 3.8h |
| **BiLSTM** | `BiLSTM.pth` | 0.843 | 0.0120 | 2.82M | 4.5h |

## 🏗️ Model Architectures

### CNN (Convolutional Neural Network)
```python
Architecture:
├── Input: 21-mer sequence (84 features)
├── Embedding: 84 → 128 dimensions
├── Conv1D: 128 filters, kernel=5, ReLU
├── Conv1D: 128 filters, kernel=5, ReLU
├── Global Max Pooling
├── Dense: 64 units, ReLU, Dropout(0.2)
├── Dense: 32 units, ReLU, Dropout(0.2)
└── Output: 1 unit (linear)
```

**Key Features:**
- 2 convolutional layers with 128 filters each
- 5×5 kernel size for motif detection
- Batch normalization for stable training
- Dropout regularization (20%)

### GRU (Gated Recurrent Unit)
```python
Architecture:
├── Input: 21-mer sequence (84 features)
├── Embedding: 84 → 128 dimensions
├── GRU Layer 1: 128 hidden, Dropout(0.2)
├── GRU Layer 2: 128 hidden, Dropout(0.2)
├── Dense: 64 units, ReLU, Dropout(0.2)
├── Dense: 32 units, ReLU, Dropout(0.2)
└── Output: 1 unit (linear)
```

**Key Features:**
- 2 GRU layers with 128 hidden units each
- Update and reset gates for memory control
- Fewer parameters than LSTM
- Efficient gradient flow

### LSTM (Long Short-Term Memory)
```python
Architecture:
├── Input: 21-mer sequence (84 features)
├── Embedding: 84 → 128 dimensions
├── LSTM Layer 1: 128 hidden, Dropout(0.2)
├── LSTM Layer 2: 128 hidden, Dropout(0.2)
├── Dense: 64 units, ReLU, Dropout(0.2)
├── Dense: 32 units, ReLU, Dropout(0.2)
└── Output: 1 unit (linear)
```

**Key Features:**
- 2 LSTM layers with 128 hidden units each
- Input, forget, output, and candidate gates
- Cell state for long-term memory
- Complex gating mechanism

### BiLSTM (Bidirectional LSTM)
```python
Architecture:
├── Input: 21-mer sequence (84 features)
├── Embedding: 84 → 128 dimensions
├── BiLSTM Layer 1: 128 hidden × 2, Dropout(0.2)
├── BiLSTM Layer 2: 128 hidden × 2, Dropout(0.2)
├── Dense: 128 units, ReLU, Dropout(0.2)
├── Dense: 64 units, ReLU, Dropout(0.2)
└── Output: 1 unit (linear)
```

**Key Features:**
- Bidirectional processing (forward + backward)
- 256 total hidden units per layer (128 × 2)
- Context from both directions
- Enhanced sequence understanding

##  Performance Analysis

### Strengths by Architecture
- **CNN**: Best at detecting local sequence motifs and patterns
- **GRU**: Efficient with good performance-to-parameter ratio
- **LSTM**: Strong long-term dependency modeling
- **BiLSTM**: Superior contextual understanding

### Performance Comparison
```
BiLSTM (0.843) ━━ Best contextual understanding
GRU    (0.837) ━━ Most parameter-efficient
LSTM   (0.837) ━━ Good long-term memory
CNN    (0.793) ━━ Pattern detection focus
```

##  Hyperparameter Details

### Training Configuration
- **Batch Size**: 64 (all models)
- **Learning Rate**: 0.0001 - 0.0002 (optimized per model)
- **Epochs**: 80 - 95 (early stopping)
- **Optimizer**: Adam (β1=0.9, β2=0.999)
- **Loss**: Mean Squared Error (MSE)

### Architecture Parameters
- **Sequence Length**: 21 nucleotides
- **Input Features**: 84 (one-hot encoded)
- **Hidden Dimensions**: 128 units per layer
- **Dropout Rate**: 20% throughout
- **Activation**: ReLU for hidden, Linear for output

##  Model Files & Access

### Direct File Access
```bash
# Model files
CNN.pth      # Convolutional Neural Network
GRU.pth      # Gated Recurrent Unit
LSTM.pth     # Long Short-Term Memory
BiLSTM.pth   # Bidirectional LSTM
```

### Loading Models
```python
import torch

# Load specific model
model = torch.load('BiLSTM.pth')  # Best performing base model
model.eval()

# For inference
with torch.no_grad():
    predictions = model(input_sequence)
```

##  Related Documentation

- **[Complete Hyperparameters](../../docs/hyperparameters/)** - Full specifications for all base models
- **[Training Procedures](../../docs/training_procedures/)** - Training protocols and validation
- **[Model Architectures](../../docs/MODEL_ARCHITECTURES.md)** - Detailed technical specifications
- **[Performance Comparison](../../docs/COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

##  Key Insights

1. **BiLSTM Superiority**: Achieves best performance among base models (0.843 Spearman)
2. **GRU Efficiency**: Matches LSTM performance with 35% fewer parameters
3. **CNN Specialization**: Excels at motif detection but limited sequence context
4. **Bidirectional Advantage**: Forward + backward processing enhances understanding

##  Usage Recommendations

- **For motif analysis**: Use CNN
- **For efficiency**: Use GRU
- **For long dependencies**: Use LSTM
- **For best performance**: Use BiLSTM
- **For research comparison**: Use BiLSTM as baseline

---
**Note**: All base models serve as foundation architectures. For state-of-the-art performance, see hybrid models in parent directory.
