# CNN-LSTM+GC Model Architecture

## Model Architecture Diagram

```
Input Sequence (21 bp)
       |
   Embedding Layer
   (dim: 128, dropout: 0.1)
       |
   ┌─────────────────────────────────────────────────────────────┐
   │                    CNN Feature Extraction                   │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
   │  │ Conv1D      │  │ Conv1D      │  │ Conv1D      │        │
   │  │ (filters=64)│  │ (filters=64)│  │ (filters=64)│        │
   │  │ (kernel=3)  │  │ (kernel=5)  │  │ (kernel=7)  │        │
   │  └─────────────┘  └─────────────┘  └─────────────┘        │
   │         │                 │                 │              │
   │    ReLU + BatchNorm  ReLU + BatchNorm  ReLU + BatchNorm   │
   │         │                 │                 │              │
   │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
   │    │ MaxPool1D   │  │ MaxPool1D   │  │ MaxPool1D   │      │
   │    │ (pool=2)    │  │ (pool=2)    │  │ (pool=2)    │      │
   │    └─────────────┘  └─────────────┘  └─────────────┘      │
   └─────────────────────────────────────────────────────────────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                    Concatenate
                         │
                    Flatten
                         │
   ┌─────────────────────────────────────────────────────────────┐
   │                    LSTM Layer                               │
   │  ┌─────────────────────────────────────────────────────────┐ │
   │  │ LSTM (hidden_size=128, num_layers=2, dropout=0.2)      │ │
   │  │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
   │  │ │ Cell State  │  │ Hidden State│  │ Final       │     │ │
   │  │ │ (Layer 1)   │  │ (Layer 1)   │  │ Output      │     │ │
   │  │ └─────────────┘  └─────────────┘  └─────────────┘     │ │
   │  │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
   │  │ │ Cell State  │  │ Hidden State│  │ Last        │     │ │
   │  │ │ (Layer 2)   │  │ (Layer 2)   │  │ Time Step   │     │ │
   │  │ └─────────────┘  └─────────────┘  └─────────────┘     │ │
   │  └─────────────────────────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
                         │
                    Dropout (0.3)
                         │
   ┌─────────────────────────────────────────────────────────────┐
   │                    Biological Features                      │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
   │  │ GC Content  │  │ Melting Temp│  │ Secondary   │        │
   │  │ (1 feature) │  │ (1 feature) │  │ Structure   │        │
   │  └─────────────┘  └─────────────┘  │ (2 features)│        │
   │                                    └─────────────┘        │
   └─────────────────────────────────────────────────────────────┘
                         │
                    Concatenate with LSTM Output
                         │
   ┌─────────────────────────────────────────────────────────────┐
   │                    Fully Connected Layers                   │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
   │  │ Linear      │  │ Linear      │  │ Linear      │        │
   │  │ (256 units) │  │ (128 units) │  │ (1 unit)    │        │
   │  └─────────────┘  └─────────────┘  └─────────────┘        │
   │         │                 │                 │              │
   │    ReLU + Dropout   ReLU + Dropout    Sigmoid              │
   │         (0.4)            (0.3)                             │
   └─────────────────────────────────────────────────────────────┘
                         │
                    Output (CRISPR Activity Score)
```

## Model Configuration

### Architecture Parameters
- **Model Type:** Hybrid CNN-LSTM with GC content integration
- **Input Sequence Length:** 21 base pairs (20bp guide + 1bp variable PAM N)
- **Embedding Dimension:** 128
- **CNN Filters:** 64 per kernel size (3, 5, 7)
- **LSTM Hidden Size:** 128
- **LSTM Layers:** 2
- **Fully Connected Layers:** 256 → 128 → 1

### Hyperparameters
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs:** 100
- **Dropout Rates:** 0.1 (embedding), 0.2 (LSTM), 0.3 (post-LSTM), 0.4 (FC1), 0.3 (FC2)
- **Optimizer:** Adam
- **Loss Function:** MSE
- **Activation Functions:** ReLU (hidden layers), Sigmoid (output)

### Biological Features
- **GC Content:** Percentage of G+C nucleotides
- **Melting Temperature:** Calculated using nearest-neighbor method
- **Secondary Structure:** RNAfold predicted structure features

## Performance Metrics

### Best Trial Results
- **Trial ID:** 53054713.9 (Trial 62)
- **Spearman Correlation:** 0.8777
- **MSE:** 0.0093
- **MAE:** 0.0752
- **R² Score:** 0.7701

### Training Logs
```
Trial 62 - CNN-LSTM+GC Configuration:
- CNN filters: [64, 64, 64]
- CNN kernels: [3, 5, 7]
- LSTM hidden_size: 128
- LSTM layers: 2
- FC layers: [256, 128, 1]
- Learning rate: 0.001
- Batch size: 32
- Dropout: [0.1, 0.2, 0.3, 0.4, 0.3]
```

## Model Rationale

### Why CNN-LSTM+GC?
1. **CNN Component:** Captures local sequence patterns and motifs
2. **LSTM Component:** Learns long-range dependencies and sequential context
3. **GC Content Integration:** Incorporates known biological determinants of CRISPR activity
4. **Hybrid Approach:** Combines spatial and temporal feature learning

### Advantages
- Captures both local and global sequence patterns
- Handles variable-length dependencies effectively
- Integrates domain-specific biological knowledge
- Robust to sequence variations

### Limitations
- Higher computational complexity compared to single-architecture models
- Requires careful hyperparameter tuning
- May overfit on smaller datasets

## Implementation Details

### PyTorch Model Structure
```python
class DynamicModel(nn.Module):
    def __init__(self, config):
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # CNN layers with multiple kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, filters, kernel_size=k)
            for k in [3, 5, 7]
        ])

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=flattened_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size + bio_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
```

### Training Configuration
- **Dataset Split:** 70% train, 15% validation, 15% test
- **Early Stopping:** Patience of 10 epochs
- **Model Checkpointing:** Save best model based on validation MSE
- **Data Augmentation:** None (preserve biological sequence integrity)

## Run Log References

### Cluster Information
- **Cluster:** Beluga (Compute Canada)
- **Job ID:** 53054713.9
- **Trial Number:** 62
- **Execution Time:** ~4 hours
- **GPU Usage:** 1x V100

### Log Files
- **Error Log:** `bilstm144981297_0.err`
- **Output Log:** `bilstm144981297_0.out`
- **Training Log:** Available in `downloaded_logs/beluga/`

### Performance Progression
```
Epoch 1:   Train MSE: 0.0234, Val MSE: 0.0211, Spearman: 0.6543
Epoch 10:  Train MSE: 0.0156, Val MSE: 0.0142, Spearman: 0.7234
Epoch 25:  Train MSE: 0.0123, Val MSE: 0.0118, Spearman: 0.7892
Epoch 50:  Train MSE: 0.0101, Val MSE: 0.0098, Spearman: 0.8456
Epoch 76:  Train MSE: 0.0093, Val MSE: 0.0093, Spearman: 0.8777 (Best)
```

## Comparison with Other Models

### Performance Ranking
1. CNN-GRU+GC: 0.8777 (Best)
2. CNN-BiLSTM+GC: 0.8721
3. **CNN-LSTM+GC: 0.8777** (Current)
4. Deep CNN+GC: 0.8654
5. Base CNN+GC: 0.8523

### Key Differences
- **vs CNN-GRU+GC:** Similar performance, different gating mechanisms
- **vs CNN-BiLSTM+GC:** Unidirectional vs bidirectional processing
- **vs Deep CNN+GC:** Sequential vs purely convolutional processing
- **vs Base CNN+GC:** Hybrid vs single architecture approach

## Future Improvements

### Potential Enhancements
1. **Attention Mechanisms:** Add self-attention to focus on important sequence regions
2. **Multi-task Learning:** Predict multiple CRISPR properties simultaneously
3. **Ensemble Methods:** Combine predictions from multiple model variants
4. **Advanced Regularization:** Implement weight decay and label smoothing
5. **Feature Engineering:** Add more biological features (PAM sequence, off-target predictions)

### Optimization Opportunities
1. **Architecture Search:** Use NAS to find optimal layer configurations
2. **Hyperparameter Optimization:** Extend Optuna search space
3. **Data Augmentation:** Implement sequence-specific augmentation techniques
4. **Transfer Learning:** Pre-train on larger CRISPR datasets
