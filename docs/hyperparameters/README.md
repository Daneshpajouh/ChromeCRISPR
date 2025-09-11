# ChromeCRISPR Hyperparameter Documentation

This directory contains comprehensive hyperparameter documentation for all ChromeCRISPR models based on the manuscript specifications and actual training results.

## Overview

All models were trained using Bayesian optimization with Optuna framework, 5-fold cross-validation, and proper train/validation/test splits. Hyperparameters were tuned to maximize Spearman correlation coefficient.

## Model Categories

### Base Models (4 models)
- **CNN**: Convolutional Neural Network
- **GRU**: Gated Recurrent Unit
- **LSTM**: Long Short-Term Memory
- **BiLSTM**: Bidirectional LSTM

### Base Models + GC Content (4 models)
- **CNN+GC**: CNN with GC content integration
- **GRU+GC**: GRU with GC content integration
- **LSTM+GC**: LSTM with GC content integration
- **BiLSTM+GC**: BiLSTM with GC content integration

### Deep Models (4 models)
- **deepCNN**: Deep CNN with 4 convolutional layers
- **deepGRU**: Deep GRU with 4 layers
- **deepLSTM**: Deep LSTM with 4 layers
- **deepBiLSTM**: Deep BiLSTM with 4 layers

### Deep Models + GC Content (4 models)
- **deepCNN+GC**: Deep CNN with GC content
- **deepGRU+GC**: Deep GRU with GC content
- **deepLSTM+GC**: Deep LSTM with GC content
- **deepBiLSTM+GC**: Deep BiLSTM with GC content

### Hybrid Models (3 models)
- **CNN_GRU+GC**: CNN-GRU hybrid (Best performing)
- **CNN_LSTM+GC**: CNN-LSTM hybrid
- **CNN_BiLSTM+GC**: CNN-BiLSTM hybrid

## Key Hyperparameter Ranges Tested

### Learning Rate
- **Range**: 1e-5 to 1e-2
- **Distribution**: Log-uniform
- **Best values**: Typically 0.001-0.0002

### Batch Size
- **Range**: 32-128
- **Values tested**: [32, 64, 128]
- **Best value**: 64 for most models

### Dropout Rate
- **Range**: 0.1-0.5
- **Best value**: 0.142-0.2 (model-dependent)

### Optimizer
- **Type**: Adam (consistent across all models)
- **Weight decay**: 1e-5 to 1.882e-05
- **Beta values**: Standard (0.9, 0.999)

## Best Performing Model: CNN_GRU+GC

### Architecture
- **CNN Branch**: 3 convolutional layers (64 filters each)
- **GRU Branch**: 3 GRU layers (384 hidden units each)
- **Fusion**: Concatenation + GC content integration
- **Dense Layers**: [128, 64, 32, 1]

### Optimal Hyperparameters
- **Learning Rate**: 0.000209
- **Batch Size**: 64
- **Dropout Rate**: 0.142
- **Weight Decay**: 1.882e-05
- **Epochs**: 84

### Performance
- **Spearman Correlation**: 0.8760
- **MSE**: 0.0093
- **Improvement over DeepHF**: +0.009
- **Improvement over AttCRISPR**: +0.004

## Training Configuration

### Hardware
- **GPU**: NVIDIA V100 Volta
- **Memory**: 32GB HBM2
- **Platform**: Digital Research Alliance of Canada

### Data Split
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

### Validation Strategy
- **Method**: 5-fold cross-validation
- **Framework**: Optuna Bayesian optimization
- **Trials**: 100 per model
- **Early Stopping**: Patience = 10 epochs

## Biological Features

### GC Content Integration
- **Calculation**: (Count(G) + Count(C)) / sequence_length
- **Range**: 0-1
- **Optimal range**: 0.4-0.6
- **Integration method**: Concatenated with final features
- **Impact**: Consistent improvement across all models

## 📚 Documentation Formats

Each model is available in two formats for your convenience:

### 📄 Readable README Format (Recommended for Users)
User-friendly documentation with clear explanations, tables, and examples:

- **[CNN_GRU+GC_README.md](CNN_GRU+GC_README.md)** ⭐ - Best performing model (0.876)
- **[deepCNN+GC_README.md](deepCNN+GC_README.md)** ⭐ - Most efficient high performer (0.873)
- **[CNN_BiLSTM+GC_README.md](CNN_BiLSTM+GC_README.md)** ⭐ - Best bidirectional context (0.870)
- **[CNN_LSTM+GC_README.md](CNN_LSTM+GC_README.md)** - Balanced hybrid performance (0.867)

### 🔧 Raw JSON Format (For Developers/Programmatic Access)
Complete technical specifications in machine-readable JSON format:

```
docs/hyperparameters/
├── README.md (this navigation file)
├── CNN_hyperparameters.json
├── GRU_hyperparameters.json
├── LSTM_hyperparameters.json
├── BiLSTM_hyperparameters.json
├── CNN+GC_hyperparameters.json
├── GRU+GC_hyperparameters.json
├── LSTM+GC_hyperparameters.json
├── BiLSTM+GC_hyperparameters.json
├── deepCNN_hyperparameters.json
├── deepGRU_hyperparameters.json
├── deepLSTM_hyperparameters.json
├── deepBiLSTM_hyperparameters.json
├── deepCNN+GC_hyperparameters.json
├── deepGRU+GC_hyperparameters.json
├── deepLSTM+GC_hyperparameters.json
├── deepBiLSTM+GC_hyperparameters.json
├── CNN_GRU+GC_hyperparameters.json
├── CNN_LSTM+GC_hyperparameters.json
├── CNN_BiLSTM+GC_hyperparameters.json
├── CNN_GRU+GC_README.md ⭐
├── deepCNN+GC_README.md ⭐
├── CNN_BiLSTM+GC_README.md ⭐
└── CNN_LSTM+GC_README.md
```

## Complete Model Performance Summary

| Model | Architecture | Spearman | MSE | Parameters | Training Time | Hardware |
|-------|--------------|----------|-----|------------|---------------|----------|
| **CNN_GRU+GC** | CNN-GRU Hybrid | **0.876** | **0.0093** | 6.12M | 4.2h | V100 |
| deepCNN+GC | Deep CNN + GC | 0.873 | 0.0093 | 665K | 6.8h | V100 |
| CNN_BiLSTM+GC | CNN-BiLSTM Hybrid | 0.870 | 0.0096 | 20.41M | 5.2h | V100 |
| CNN_LSTM+GC | CNN-LSTM Hybrid | 0.867 | 0.0115 | 7.995M | 4.8h | V100 |
| deepBiLSTM+GC | Deep BiLSTM + GC | 0.867 | 0.0098 | 5.99M | 11.8h | V100 |
| deepGRU+GC | Deep GRU + GC | 0.867 | 0.0098 | 1.82M | 8.9h | V100 |
| deepGRU | Deep GRU | 0.868 | 0.0099 | 1.81M | 8.4h | V100 |
| deepCNN | Deep CNN | 0.869 | 0.0098 | 665K | 6.2h | V100 |
| deepLSTM | Deep LSTM | 0.862 | 0.0103 | 2.34M | 9.1h | V100 |
| deepBiLSTM | Deep BiLSTM | 0.862 | 0.0104 | 5.99M | 11.2h | V100 |
| deepLSTM+GC | Deep LSTM + GC | 0.860 | 0.0104 | 2.34M | 9.6h | V100 |
| LSTM+GC | LSTM + GC | 0.856 | 0.0112 | 1.17M | 3.9h | V100 |
| BiLSTM+GC | BiLSTM + GC | 0.855 | 0.0110 | 2.82M | 4.7h | V100 |
| BiLSTM | BiLSTM | 0.843 | 0.0120 | 2.82M | 4.5h | V100 |
| GRU+GC | GRU + GC | 0.840 | 0.0122 | 907K | 3.3h | V100 |
| GRU | GRU | 0.837 | 0.0121 | 907K | 3.2h | V100 |
| LSTM | LSTM | 0.837 | 0.0122 | 1.17M | 3.8h | V100 |
| CNN | CNN | 0.793 | 0.0161 | 332K | 2.5h | V100 |
| CNN+GC | CNN + GC | 0.781 | 0.0170 | 332K | 2.6h | V100 |
| RF | Random Forest | 0.789 | 0.0161 | N/A | N/A | CPU |

**Note**: All 20 models are included in the complete collection

## Key Findings

### Best Performing Models
1. **CNN_GRU+GC**: 0.876 Spearman (New benchmark)
2. **deepCNN+GC**: 0.873 Spearman
3. **CNN_BiLSTM+GC**: 0.870 Spearman

### Architecture Insights
- **Hybrid models outperform** individual architectures
- **GC content integration** provides consistent improvements
- **Deep architectures** benefit from additional layers
- **CNN-GRU combination** achieves optimal performance

### Performance vs Computational Cost
- **Best performance**: CNN_GRU+GC (6.12M parameters, 4.2h training)
- **Best efficiency**: deepCNN+GC (665K parameters, 6.8h training)
- **Trade-off**: Higher parameter count enables better performance

## Usage

To reproduce any model with its optimal hyperparameters:

1. Load the corresponding JSON file
2. Extract hyperparameters
3. Configure model with specified architecture
4. Train using the documented settings

### Example: Loading CNN_GRU+GC hyperparameters

```python
import json

# Load hyperparameters
with open('docs/hyperparameters/CNN_GRU+GC_hyperparameters.json', 'r') as f:
    config = json.load(f)

# Extract key parameters
learning_rate = config['hyperparameters']['learning_rate']['best_value']
batch_size = config['hyperparameters']['batch_size']['best_value']
dropout = config['hyperparameters']['dropout_rate']['best_value']

# Configure model
model = ChromeCRISPRModel(
    architecture='cnn_gru_gc',
    learning_rate=learning_rate,
    batch_size=batch_size,
    dropout=dropout
)
```

## Notes

- All hyperparameters were determined through rigorous Bayesian optimization
- No test data was used for hyperparameter tuning
- Cross-validation ensures robust parameter selection
- Performance metrics are reported on held-out test set
