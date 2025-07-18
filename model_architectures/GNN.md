# GNN Model Architecture

## Model Architecture Diagram

```
Input Sequence (21 bp)
       |
   Embedding Layer
   (dim: 128, dropout: 0.1)
       |
   Graph Construction
   (k-nearest neighbors, similarity)
       |
   ┌─────────────────────────────────────────────────────────────┐
   │                    Graph Neural Network                     │
   │                                                             │
   │  ┌─────────────────────────────────────────────────────────┐ │
   │  │                    GNN Layer 1                          │ │
   │  │  ┌─────────────────────────────────────────────────────┐ │ │
   │  │  │                Graph Convolution                     │ │ │
   │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │ │
   │  │  │  │ Node        │  │ Edge        │  │ Message     │ │ │ │
   │  │  │  │ Features    │  │ Features    │  │ Passing     │ │ │ │
   │  │  │  │ (128)       │  │ (64)        │  │ (Aggregate) │ │ │ │
   │  │  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │ │
   │  │  │         │                 │                 │        │ │ │
   │  │  │         └───────── Update ─────────────────┘        │ │ │
   │  │  │                         │                            │ │ │
   │  │  │                   ReLU + BatchNorm                  │ │ │
   │  │  └─────────────────────────────────────────────────────┘ │ │
   │  │                         │                                │ │
   │  │                   Dropout                               │ │
   │  └─────────────────────────────────────────────────────────┘ │
   │                         │                                    │
   │  ┌─────────────────────────────────────────────────────────┐ │
   │  │                    GNN Layer N                          │ │
   │  │  (Repeat structure for N GNN layers)                    │ │
   │  └─────────────────────────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
       |
   Global Pooling
   (Mean/Max/Sum)
       |
   Fully Connected Layer(s)
       |
   Output (Regression/Classification)
```
