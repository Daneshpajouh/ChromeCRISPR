# LSTM Model Architecture

## Model Architecture Diagram

```
Input Sequence (21 bp)
       |
   Embedding Layer
   (dim: 128, dropout: 0.1)
       |
   ┌─────────────────────────────────────────────────────────────┐
   │                        LSTM Layer(s)                        │
   │  ┌─────────────┐  ┌─────────────┐  ...  ┌─────────────┐    │
   │  │ LSTM Cell   │→ │ LSTM Cell   │→ ...→ │ LSTM Cell   │    │
   │  │ (hidden=128)│  │ (hidden=128)│      │ (hidden=128)│    │
   │  └─────────────┘  └─────────────┘      └─────────────┘    │
   │         │                 │                 │              │
   │      Dropout           Dropout           Dropout           │
   └─────────────────────────────────────────────────────────────┘
       |
   Last Hidden State
       |
   Fully Connected Layer(s)
       |
   Output (Regression/Classification)
```
