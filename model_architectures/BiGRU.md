# BiGRU Model Architecture

## Model Architecture Diagram

```
Input Sequence (21 bp)
       |
   Embedding Layer
   (dim: 128, dropout: 0.1)
       |
   ┌─────────────────────────────────────────────────────────────┐
   │                      BiGRU Layer(s)                         │
   │  ┌─────────────┐  ┌─────────────┐  ...  ┌─────────────┐    │
   │  │ BiGRU Cell  │→ │ BiGRU Cell  │→ ...→ │ BiGRU Cell  │    │
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
