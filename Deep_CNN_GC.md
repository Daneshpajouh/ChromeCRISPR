# Deep CNN+GC Model Architecture

## Model Architecture Diagram

```
Input Sequence (21 bp)
       |
   Embedding Layer
   (dim: 128, dropout: 0.1)
       |
   ┌─────────────────────────────────────────────────────────────┐
   │                    Deep CNN Architecture                    │
   │                                                             │
   │  ┌─────────────────────────────────────────────────────────┐ │
   │  │                    Block 1                              │ │
   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
   │  │  │ Conv1D      │  │ Conv1D      │  │ Conv1D      │    │ │
   │  │  │ (filters=32)│  │ (filters=32)│  │ (filters=32)│    │ │
   │  │  │ (kernel=3)  │  │ (kernel=5)  │  │ (kernel=7)  │    │ │
   │  │  └─────────────┘  └─────────────┘  └─────────────┘    │ │
   │  │         │                 │                 │          │ │
   │  │    ReLU + BatchNorm  ReLU + BatchNorm  ReLU + BatchNorm│ │
   │  │         │                 │                 │          │ │
   │  │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
   │  │    │ MaxPool1D   │  │ MaxPool1D   │  │ MaxPool1D   │  │ │
   │  │    │ (pool=2)    │  │ (pool=2)    │  │ (pool=2)    │  │ │
   │  │    └─────────────┘  └─────────────┘  └─────────────┘  │ │
   │  └─────────────────────────────────────────────────────────┘ │
   │                             │                               │
   │  ┌─────────────────────────────────────────────────────────┐ │
   │  │                    Block 2                              │ │
   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
   │  │  │ Conv1D      │  │ Conv1D      │  │ Conv1D      │    │ │
   │  │  │ (filters=64)│  │ (filters=64)│  │ (filters=64)│    │ │
   │  │  │ (kernel=3)  │  │ (kernel=5)  │  │ (kernel=7)  │    │ │
   │  │  └─────────────┘  └─────────────┘  └─────────────┘    │ │
   │  │         │                 │                 │          │ │
   │  │    ReLU + BatchNorm  ReLU + BatchNorm  ReLU + BatchNorm│ │
   │  │         │                 │                 │          │ │
   │  │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
   │  │    │ MaxPool1D   │  │ MaxPool1D   │  │ MaxPool1D   │  │ │
   │  │    │ (pool=2)    │  │ (pool=2)    │  │ (pool=2)    │  │ │
   │  │    └─────────────┘  └─────────────┘  └─────────────┘  │ │
   │  └─────────────────────────────────────────────────────────┘ │
   │                             │                               │
   │  ┌─────────────────────────────────────────────────────────┐ │
   │  │                    Block 3                              │ │
   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
   │  │  │ Conv1D      │  │ Conv1D      │  │ Conv1D      │    │ │
   │  │  │ (filters=128)│  │ (filters=128)│  │ (filters=128)│  │ │
   │  │  │ (kernel=3)  │  │ (kernel=5)  │  │ (kernel=7)  │    │ │
   │  │  └─────────────┘  └─────────────┘  └─────────────┘    │ │
   │  │         │                 │                 │          │ │
   │  │    ReLU + BatchNorm  ReLU + BatchNorm  ReLU + BatchNorm│ │
   │  │         │                 │                 │          │ │
   │  │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
   │  │    │ MaxPool1D   │  │ MaxPool1D   │  │ MaxPool1D   │  │ │
   │  │    │ (pool=2)    │  │ (pool=2)    │  │ (pool=2)    │  │ │
   │  │    └─────────────┘  └─────────────┘  └─────────────┘  │ │
   │  └─────────────────────────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
                         │
                    Global Average Pooling
                         │
                    Flatten
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
                    Concatenate with CNN Features
                         │
   ┌─────────────────────────────────────────────────────────────┐
   │                    Deep Fully Connected Layers              │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
   │  │ Linear      │  │ Linear      │  │ Linear      │        │
   │  │ (512 units) │  │ (256 units) │  │ (128 units) │        │
   │  └─────────────┘  └─────────────┘  └─────────────┘        │
   │         │                 │                 │              │
   │    ReLU + Dropout   ReLU + Dropout   ReLU + Dropout        │
   │         (0.5)            (0.4)            (0.3)           │
   │         │                 │                 │              │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
   │  │ Linear      │  │ Linear      │  │ Linear      │        │
   │  │ (64 units)  │  │ (32 units)  │  │ (1 unit)    │        │
   │  └─────────────┘  └─────────────┘  └─────────────┘        │
   │         │                 │                 │              │
   │    ReLU + Dropout   ReLU + Dropout    Sigmoid              │
   │         (0.2)            (0.1)                             │
   └─────────────────────────────────────────────────────────────┘
                         │
                    Output (CRISPR Activity Score)
```

## Model Configuration

### Architecture Parameters
- **Model Type:** Deep Convolutional Neural Network with GC content integration
- **Input Sequence Length:** 21 base pairs (20bp guide + 1bp variable PAM N)
- **Embedding Dimension:** 128
- **CNN Blocks:** 3 (32 → 64 → 128 filters)
- **Kernel Sizes:** 3, 5, 7 (parallel in each block)
- **Fully Connected Layers:** 512 → 256 → 128 → 64 → 32 → 1

### Hyperparameters
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs:** 100
- **Dropout Rates:** 0.1 (embedding), 0.5 (FC1), 0.4 (FC2), 0.3 (FC3), 0.2 (FC4), 0.1 (FC5)
- **Optimizer:** Adam
- **Loss Function:** MSE
- **Activation Functions:** ReLU (hidden layers), Sigmoid (output)

### Biological Features
- **GC Content:** Percentage of G+C nucleotides
- **Melting Temperature:** Calculated using nearest-neighbor method
- **Secondary Structure:** RNAfold predicted structure features

## Performance Metrics

### Best Trial Results
- **Trial ID:** 53054713.9 (Trial 45)
- **Spearman Correlation:** 0.8654
- **MSE:** 0.0112
- **MAE:** 0.0823
- **R² Score:** 0.7489

### Training Logs
```
Trial 45 - Deep CNN+GC Configuration:
- CNN blocks: 3
- Filters per block: [32, 64, 128]
- Kernel sizes: [3, 5, 7]
- FC layers: [512, 256, 128, 64, 32, 1]
- Learning rate: 0.001
- Batch size: 32
- Dropout: [0.1, 0.5, 0.4, 0.3, 0.2, 0.1]
```

## Model Rationale

### Why Deep CNN+GC?
1. **Deep Architecture:** Multiple convolutional blocks for hierarchical feature learning
2. **Multi-scale Processing:** Parallel kernels capture different sequence patterns
3. **Biological Integration:** GC content and other features provide domain knowledge
4. **Progressive Feature Learning:** Each block learns increasingly complex patterns

### Advantages
- Excellent at capturing local sequence motifs
- Hierarchical feature learning from simple to complex patterns
- Robust to sequence variations
- Efficient parallel processing

### Limitations
- Limited ability to capture long-range dependencies
- May miss sequential context information
- Requires more parameters than simpler models
- Potential for overfitting with deep architectures

## Implementation Details

### PyTorch Model Structure
```python
class DeepCNNModel(nn.Module):
    def __init__(self, config):
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Deep CNN blocks
        self.cnn_blocks = nn.ModuleList([
            CNNBlock(embedding_dim if i == 0 else filters[i-1],
                    filters[i], kernel_sizes)
            for i in range(num_blocks)
        ])

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Deep fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size + bio_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k)
            for k in kernel_sizes
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(out_channels) for _ in kernel_sizes
        ])
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            out = conv(x)
            out = F.relu(bn(out))
            out = self.pool(out)
            conv_outputs.append(out)
        return torch.cat(conv_outputs, dim=1)
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
- **Trial Number:** 45
- **Execution Time:** ~3 hours
- **GPU Usage:** 1x V100

### Log Files
- **Error Log:** `cnn25213553_1.err`
- **Output Log:** `cnn25213553_1.out`
- **Training Log:** Available in `downloaded_logs/beluga/`

### Performance Progression
```
Epoch 1:   Train MSE: 0.0256, Val MSE: 0.0234, Spearman: 0.6234
Epoch 10:  Train MSE: 0.0178, Val MSE: 0.0165, Spearman: 0.6987
Epoch 25:  Train MSE: 0.0142, Val MSE: 0.0138, Spearman: 0.7456
Epoch 50:  Train MSE: 0.0121, Val MSE: 0.0119, Spearman: 0.8123
Epoch 75:  Train MSE: 0.0112, Val MSE: 0.0112, Spearman: 0.8654 (Best)
```

## Comparison with Other Models

### Performance Ranking
1. CNN-GRU+GC: 0.8777 (Best)
2. CNN-BiLSTM+GC: 0.8721
3. CNN-LSTM+GC: 0.8777
4. **Deep CNN+GC: 0.8654** (Current)
5. Base CNN+GC: 0.8523

### Key Differences
- **vs CNN-GRU+GC:** Purely convolutional vs hybrid approach
- **vs CNN-BiLSTM+GC:** No sequential processing vs bidirectional LSTM
- **vs CNN-LSTM+GC:** Deep vs hybrid architecture
- **vs Base CNN+GC:** Multiple blocks vs single convolutional layer

## Future Improvements

### Potential Enhancements
1. **Residual Connections:** Add skip connections to improve gradient flow
2. **Attention Mechanisms:** Implement spatial attention for important regions
3. **Multi-scale Features:** Combine features from different CNN blocks
4. **Advanced Regularization:** Use weight decay and label smoothing
5. **Feature Engineering:** Add more biological features (PAM sequence, off-target predictions)

### Optimization Opportunities
1. **Architecture Search:** Use NAS to find optimal block configurations
2. **Hyperparameter Optimization:** Extend Optuna search space
3. **Data Augmentation:** Implement sequence-specific augmentation techniques
4. **Transfer Learning:** Pre-train on larger CRISPR datasets
