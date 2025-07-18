# CNN-GRU+GC Model Architecture

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
   │                    GRU Layer                                │
   │  ┌─────────────────────────────────────────────────────────┐ │
   │  │ GRU (hidden_size=128, num_layers=2, dropout=0.2)       │ │
   │  │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
   │  │ │ Hidden State│  │ Hidden State│  │ Hidden State│     │ │
   │  │ │ (Layer 1)   │  │ (Layer 2)   │  │ (Final)     │     │ │
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
                    Concatenate with GRU Output
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

---

## Hyperparameters (Best Run)
- Learning rate: 0.00013
- Batch size: 64
- Weight decay: 0.00012
- Epochs: 82
- Dropout: 0.19
- Activation: LeakyReLU
- Optimizer: RAdam
- Scheduler: Cosine
- BatchNorm: None
- GRU layers: 2, hidden size: 128
- CNN layers: 1, 64 filters, kernel size 3
- Pooling: Last time step
- Biofeatures: GC Content, Td, Enthalpy_Duplex, Entropy_Duplex, Free_Energy_RNA

---

## Performance
- **Spearman correlation:** 0.8777
- **MSE:** 0.0093

---

## Rationale
- Combines local motif extraction (CNN) with sequential dependency modeling (GRU).
- GC content is added at the final layer to modulate predictions based on known biological relevance.
- Outperformed all other models in the study.

---

## Run Log Details
- Trained on 85% of data, tested on 15%.
- 5-fold cross-validation for hyperparameter tuning.
- Training time: ~20s/iteration on NVIDIA V100 GPU.
- Cluster: Beluga
- Job ID: 53054713.9, Trial 62
- Log: `bio_hdynamic53054713.9_trial_62.err`

---

## Copy-Paste for Manuscript
> The ChromeCRISPR CNN-GRU+GC hybrid model consists of an embedding layer, a 1D convolutional layer (64 filters, kernel size 3), followed by a 2-layer GRU (128 hidden units), with GC content concatenated before the final fully connected layers. The model achieved a Spearman correlation of 0.8777 and MSE of 0.0093, outperforming all other tested architectures.
