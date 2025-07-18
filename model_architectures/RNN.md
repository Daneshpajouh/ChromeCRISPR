# RNN Model Architecture

## Model Architecture Diagram

```
Input Sequence (21 bp)
       |
   Embedding Layer
   (dim: 128, dropout: 0.1)
       |
   ┌─────────────────────────────────────────────────────────────┐
   │                        RNN Layer(s)                         │
   │  ┌─────────────┐  ┌─────────────┐  ...  ┌─────────────┐    │
   │  │ RNN Cell    │→ │ RNN Cell    │→ ...→ │ RNN Cell    │    │
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
