# CNN+BiLSTM+GC Model Architecture

## Overview

The CNN+BiLSTM+GC model is the best performing architecture in the ChromeCRISPR framework, achieving a Spearman correlation of **0.8768** and MSE of **0.0095**. This hybrid model combines convolutional neural networks for local feature extraction with bidirectional LSTM for sequential modeling, enhanced by GC content and other biological features.

## Architecture Diagram

```
Input DNA Sequence (21 bp)
         │
    Embedding Layer
    (4 → 128 dim)
         │
    ┌─────────────────────────────────────────────────────────────┐
    │                    CNN Feature Extraction                   │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
    │  │ Conv1D      │  │ Conv1D      │  │ Conv1D      │        │
    │  │ (64 filters)│  │ (64 filters)│  │ (64 filters)│        │
    │  │ kernel=5    │  │ kernel=5    │  │ kernel=5    │        │
    │  └─────────────┘  └─────────────┘  └─────────────┘        │
    │         │                 │                 │              │
    │    ReLU + BatchNorm + Dropout (0.1)                       │
    └─────────────────────────────────────────────────────────────┘
                         │
    ┌─────────────────────────────────────────────────────────────┐
    │                Bidirectional LSTM Processing                │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
    │  │ BiLSTM      │  │ BiLSTM      │  │ BiLSTM      │        │
    │  │ Layer 1     │  │ Layer 2     │  │ Last Time   │        │
    │  │ (128 units) │  │ (128 units) │  │ Step Output │        │
    │  └─────────────┘  └─────────────┘  └─────────────┘        │
    │         │                 │                 │              │
    │    Dropout (0.18) + Concatenation (256 dim)                │
    └─────────────────────────────────────────────────────────────┘
                         │
    ┌─────────────────────────────────────────────────────────────┐
    │                    Biofeature Integration                   │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
    │  │ GC Content  │  │ Melting     │  │ Enthalpy    │        │
    │  │ (1 dim)     │  │ Temp (1 dim)│  │ Duplex      │        │
    │  └─────────────┘  └─────────────┘  │ (1 dim)     │        │
    │         │                 │        └─────────────┘        │
    │         └─────────────────┼───────────────────────────────┘
    │                           │
    │  ┌─────────────┐  ┌─────────────┐                        │
    │  │ Entropy     │  │ Free Energy │                        │
    │  │ Duplex      │  │ RNA (1 dim) │                        │
    │  │ (1 dim)     │  └─────────────┘                        │
    │  └─────────────┘                                         │
    └─────────────────────────────────────────────────────────────┘
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
- **Model Type:** CNN-BiLSTM Hybrid with GC content integration
- **Input Sequence Length:** 21 base pairs (20bp guide + 1bp variable PAM N)
- **Embedding Dimension:** 128
- **CNN Filters:** 64 per layer, kernel size 5
- **BiLSTM Hidden Size:** 128 (256 total with bidirectional)
- **Fully Connected Layers:** 256 → 128 → 1

### Hyperparameters (Best Run)
- Learning rate: 0.0011062631134035848
- Batch size: 64
- Weight decay: 1.614789571500063e-05
- Epochs: 113
- Dropout: 0.18458170507780688
- Activation: ReLU
- Optimizer: Adam
- Scheduler: Plateau
- BiLSTM layers: 2, hidden size: 128
- CNN layers: 2, 64 filters, kernel size 5
- Pooling: Last time step
- Biofeatures: GC Content, Td, Enthalpy_Duplex, Entropy_Duplex, Free_Energy_RNA

### Performance
- **Spearman correlation:** 0.8768
- **MSE:** 0.0095

### Rationale
- Combines local motif extraction (CNN) with bidirectional sequential modeling (BiLSTM).
- GC content and biofeatures are added at the final layer to modulate predictions based on known biological relevance.
- Bidirectional processing captures both forward and backward sequence dependencies.
- Outperformed all other models in the study.

### Run Log Details
- Trained on 85% of data, tested on 15%.
- 5-fold cross-validation for hyperparameter tuning.
- Training time: ~20s/iteration on NVIDIA V100 GPU.
- Cluster: Beluga
- Job ID: 53054713.9, Trial 74
- Log: `bio_hdynamic53054713.9_trial_74.err`

### Copy-Paste for Manuscript
> The ChromeCRISPR CNN-BiLSTM+GC hybrid model consists of an embedding layer, two 1D convolutional layers (64 filters, kernel size 5), followed by a 2-layer bidirectional LSTM (128 hidden units), with GC content and biofeatures concatenated before the final fully connected layers. The model achieved a Spearman correlation of 0.8768 and MSE of 0.0095, outperforming all other tested architectures.

## Comparison with Other Models

### Performance Ranking
1. **CNN+BiLSTM+GC: 0.8768** (Best - Current)
2. CNN+GRU+GC: 0.8756
3. CNN+LSTM+GC: 0.8701
4. Deep CNN+GC: 0.8732
5. Base CNN+GC: 0.8523

### Key Differences
- **vs CNN+GRU+GC:** Bidirectional vs unidirectional processing
- **vs CNN+LSTM+GC:** Bidirectional vs unidirectional LSTM
- **vs Deep CNN+GC:** Hybrid vs purely convolutional approach
- **vs Base CNN+GC:** Complex hybrid vs simple CNN architecture

## Future Improvements

### Potential Enhancements
1. **Attention Mechanisms:** Add attention to focus on important sequence regions
2. **Advanced Regularization:** Use weight decay and label smoothing
3. **Feature Engineering:** Add more biological features (PAM sequence, off-target predictions)
4. **Architecture Search:** Use NAS to find optimal layer configurations
5. **Transfer Learning:** Pre-train on larger CRISPR datasets

### Optimization Opportunities
1. **Hyperparameter Optimization:** Extend Optuna search space
2. **Data Augmentation:** Implement sequence-specific augmentation techniques
3. **Ensemble Methods:** Combine multiple model predictions
4. **Interpretability:** Add attention visualization for model interpretability
