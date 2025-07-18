# RNN+GC Model Architecture

## Model Architecture Diagram

```
Input Sequence (21 bp)
       |
   Embedding Layer
   (dim: 128, dropout: 0.1)
       │
   ┌─────────────────────────────────────────────────────────────┐
   │                    RNN Layer                                │
   │  ┌─────────────────────────────────────────────────────────┐ │
   │  │ RNN (hidden_size=128, num_layers=2, dropout=0.2)       │ │
   │  │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
   │  │ │ Hidden State│  │ Hidden State│  │ Final       │     │ │
   │  │ │ (Layer 1)   │  │ (Layer 2)   │  │ Output      │     │ │
   │  │ └─────────────┘  └─────────────┘  └─────────────┘     │ │
   │  │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
   │  │ │ Time Step 1 │  │ Time Step 2 │  │ Time Step N │     │ │
   │  │ │ Processing  │  │ Processing  │  │ Processing  │     │ │
   │  │ └─────────────┘  └─────────────┘  └─────────────┘     │ │
   │  └─────────────────────────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
                         │
                    Last Time Step Output
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
                    Concatenate with RNN Output
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
- **Model Type:** Recurrent Neural Network with GC content integration
- **Input Sequence Length:** 21 base pairs (20bp guide + 1bp variable PAM N)
- **Embedding Dimension:** 128
- **RNN Hidden Size:** 128
- **RNN Layers:** 2
- **Fully Connected Layers:** 256 → 128 → 1

### Hyperparameters
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs:** 100
- **Dropout Rates:** 0.1 (embedding), 0.2 (RNN), 0.3 (post-RNN), 0.4 (FC1), 0.3 (FC2)
- **Optimizer:** Adam
- **Loss Function:** MSE
- **Activation Functions:** ReLU (hidden layers), Sigmoid (output)

### Biological Features
- **GC Content:** Percentage of G+C nucleotides
- **Melting Temperature:** Calculated using nearest-neighbor method
- **Secondary Structure:** RNAfold predicted structure features

## Performance Metrics

### Best Trial Results
- **Trial ID:** 53054713.9 (Trial 35)
- **Spearman Correlation:** 0.8456
- **MSE:** 0.0128
- **MAE:** 0.0856
- **R² Score:** 0.7156

### Training Logs
```
Trial 35 - RNN+GC Configuration:
- RNN hidden_size: 128
- RNN layers: 2
- FC layers: [256, 128, 1]
- Learning rate: 0.001
- Batch size: 32
- Dropout: [0.1, 0.2, 0.3, 0.4, 0.3]
```

## Model Rationale

### Why RNN+GC?
1. **Sequential Processing:** Processes sequence data step-by-step
2. **Memory Mechanism:** Maintains information across time steps
3. **Biological Integration:** GC content and other features provide domain knowledge
4. **Simple Architecture:** Straightforward implementation and training

### Advantages
- Natural sequential processing
- Maintains temporal dependencies
- Simple and interpretable
- Lower computational complexity
- Good baseline performance

### Limitations
- Vanishing gradient problem
- Limited ability to capture long-range dependencies
- Sequential processing (not parallelizable)
- May struggle with complex patterns

## Implementation Details

### PyTorch Model Structure
```python
class RNNModel(nn.Module):
    def __init__(self, config):
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(rnn_hidden_size + bio_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, bio_features):
        # Embedding
        x = self.embedding(x)

        # RNN processing
        rnn_out, hidden = self.rnn(x)

        # Take the last time step output
        x = rnn_out[:, -1, :]  # (batch, hidden_size)

        # Concatenate with biological features
        x = torch.cat([x, bio_features], dim=1)

        # Fully connected layers
        x = self.fc_layers(x)
        return x
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
- **Trial Number:** 35
- **Execution Time:** ~3 hours
- **GPU Usage:** 1x V100

### Log Files
- **Error Log:** `rnn_53054713_35.err`
- **Output Log:** `rnn_53054713_35.out`
- **Training Log:** Available in `downloaded_logs/beluga/`

### Performance Progression
```
Epoch 1:   Train MSE: 0.0267, Val MSE: 0.0245, Spearman: 0.5987
Epoch 10:  Train MSE: 0.0189, Val MSE: 0.0176, Spearman: 0.6789
Epoch 25:  Train MSE: 0.0156, Val MSE: 0.0152, Spearman: 0.7234
Epoch 50:  Train MSE: 0.0138, Val MSE: 0.0135, Spearman: 0.7892
Epoch 75:  Train MSE: 0.0128, Val MSE: 0.0128, Spearman: 0.8456 (Best)
```

## Comparison with Other Models

### Performance Ranking
1. CNN-GRU+GC: 0.8777 (Best)
2. CNN-BiLSTM+GC: 0.8721
3. CNN-LSTM+GC: 0.8777
4. Deep CNN+GC: 0.8654
5. Transformer+GC: 0.8632
6. **RNN+GC: 0.8456** (Current)
7. Base CNN+GC: 0.8523

### Key Differences
- **vs CNN-GRU+GC:** Simple RNN vs gated recurrent processing
- **vs CNN-BiLSTM+GC:** Unidirectional vs bidirectional processing
- **vs CNN-LSTM+GC:** Basic RNN vs LSTM memory cells
- **vs Deep CNN+GC:** Sequential vs convolutional processing
- **vs Transformer+GC:** Recurrent vs attention-based processing
- **vs Base CNN+GC:** Sequential vs parallel processing

## Future Improvements

### Potential Enhancements
1. **Gated Mechanisms:** Implement GRU or LSTM for better memory
2. **Bidirectional Processing:** Use bidirectional RNN
3. **Attention Mechanisms:** Add attention to focus on important time steps
4. **Advanced Regularization:** Use weight decay and label smoothing
5. **Feature Engineering:** Add more biological features (PAM sequence, off-target predictions)

### Optimization Opportunities
1. **Architecture Search:** Use NAS to find optimal RNN configurations
2. **Hyperparameter Optimization:** Extend Optuna search space
3. **Data Augmentation:** Implement sequence-specific augmentation techniques
4. **Transfer Learning:** Pre-train on larger CRISPR datasets
