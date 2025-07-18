# CNN+GRU Model Architecture

## Model Architecture Diagram

```
Input Sequence (23 bp)
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
   │      MaxPool           MaxPool           MaxPool          │
   │         │                 │                 │              │
   │      ────────────── Concatenate ──────────────            │
   └─────────────────────────────────────────────────────────────┘
       |
   GRU Layer(s)
       |
   Last Hidden State
       |
   Fully Connected Layer(s)
       |
   Output (Regression/Classification)
```
