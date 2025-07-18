# BiLSTM Model Architecture

## Model Architecture Diagram

```
Input Sequence (21 bp)
       |
   Embedding Layer
   (dim: 128, dropout: 0.1)
       |
   ┌─────────────────────────────────────────────────────────────┐
   │                      BiLSTM Layer(s)                        │
   │  ┌─────────────┐  ┌─────────────┐  ...  ┌─────────────┐    │
   │  │ BiLSTM Cell │→ │ BiLSTM Cell │→ ...→ │ BiLSTM Cell │    │
   │  │ (hidden=128)│  │ (hidden=128)│      │ (hidden=128)│    │
   │  └─────────────┘  └─────────────┘      └─────────────┘    │
   │         │                 │                 │              │
   │      Dropout           Dropout           Dropout           │
   └─────────────────────────────────────────────────────────────┘
       |
   Concatenate Forward & Backward States
       |
   Fully Connected Layer(s)
       |
   Output (Regression/Classification)
```
