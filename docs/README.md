# ChromeCRISPR Documentation

Complete technical documentation for all 20 models, hyperparameters, training procedures, and performance analysis.

## Documentation Overview

| Section | Description | Files |
|---------|-------------|-------|
| **[Hyperparameters](hyperparameters/)** | Complete specifications for all 20 models | 20 JSON files + README |
| **[Training Procedures](training_procedures/)** | Detailed training protocols and methodologies | README |
| **[Model Architectures](../docs/MODEL_ARCHITECTURES.md)** | Technical architecture descriptions | Single comprehensive file |
| **[Performance Analysis](../docs/COMPREHENSIVE_MODEL_DOCUMENTATION.md)** | Comparative analysis and benchmarking | Single analysis file |

## Quick Access Guide

### Best Model Information
```bash
BEST MODEL: CNN_GRU+GC
├── Performance: 0.876 Spearman, 0.0093 MSE
├── File: ../models/chromecrispr_hybrid_models/CNN_GRU+GC.pth
├── Hyperparameters: hyperparameters/CNN_GRU+GC_hyperparameters.json
├── Architecture: MODEL_ARCHITECTURES.md (Hybrid Models section)
```

### Find Model Information Quickly
1. **Performance Rankings**: See `hyperparameters/README.md`
2. **Architecture Details**: See `MODEL_ARCHITECTURES.md`
3. **Training Procedures**: See `training_procedures/README.md`
4. **Comparative Analysis**: See `COMPREHENSIVE_MODEL_DOCUMENTATION.md`

## Model Categories & Access

### Hybrid Models (Best Performance)
| Model | Performance | File Location | Hyperparameters |
|-------|-------------|---------------|----------------|
| **CNN_GRU+GC** | 0.876 | `../models/chromecrispr_hybrid_models/` | [View](hyperparameters/CNN_GRU+GC_hyperparameters.json) |
| CNN_BiLSTM+GC | 0.870 | `../models/chromecrispr_hybrid_models/` | [View](hyperparameters/CNN_BiLSTM+GC_hyperparameters.json) |
| CNN_LSTM+GC | 0.867 | `../models/chromecrispr_hybrid_models/` | [View](hyperparameters/CNN_LSTM+GC_hyperparameters.json) |

### Deep Models + GC Content
| Model | Performance | File Location | Hyperparameters |
|-------|-------------|---------------|----------------|
| deepCNN+GC | 0.873 | `../models/deep_models_with_gc/` | [View](hyperparameters/deepCNN+GC_hyperparameters.json) |
| deepBiLSTM+GC | 0.867 | `../models/deep_models_with_gc/` | [View](hyperparameters/deepBiLSTM+GC_hyperparameters.json) |
| deepGRU+GC | 0.867 | `../models/deep_models_with_gc/` | [View](hyperparameters/deepGRU+GC_hyperparameters.json) |
| deepLSTM+GC | 0.860 | `../models/deep_models_with_gc/` | [View](hyperparameters/deepLSTM+GC_hyperparameters.json) |

### Deep Models (No GC)
| Model | Performance | File Location | Hyperparameters |
|-------|-------------|---------------|----------------|
| deepCNN | 0.869 | `../models/deep_models/` | [View](hyperparameters/deepCNN_hyperparameters.json) |
| deepGRU | 0.868 | `../models/deep_models/` | [View](hyperparameters/deepGRU_hyperparameters.json) |
| deepLSTM | 0.862 | `../models/deep_models/` | [View](hyperparameters/deepLSTM_hyperparameters.json) |
| deepBiLSTM | 0.862 | `../models/deep_models/` | [View](hyperparameters/deepBiLSTM_hyperparameters.json) |

### Base Models + GC Content
| Model | Performance | File Location | Hyperparameters |
|-------|-------------|---------------|----------------|
| LSTM+GC | 0.856 | `../models/base_models_with_gc/` | [View](hyperparameters/LSTM+GC_hyperparameters.json) |
| BiLSTM+GC | 0.855 | `../models/base_models_with_gc/` | [View](hyperparameters/BiLSTM+GC_hyperparameters.json) |
| GRU+GC | 0.840 | `../models/base_models_with_gc/` | [View](hyperparameters/GRU+GC_hyperparameters.json) |
| CNN+GC | 0.781 | `../models/base_models_with_gc/` | [View](hyperparameters/CNN+GC_hyperparameters.json) |

### Base Models (No GC)
| Model | Performance | File Location | Hyperparameters |
|-------|-------------|---------------|----------------|
| BiLSTM | 0.843 | `../models/base_models/` | [View](hyperparameters/BiLSTM_hyperparameters.json) |
| GRU | 0.837 | `../models/base_models/` | [View](hyperparameters/GRU_hyperparameters.json) |
| LSTM | 0.837 | `../models/base_models/` | [View](hyperparameters/LSTM_hyperparameters.json) |
| CNN | 0.793 | `../models/base_models/` | [View](hyperparameters/CNN_hyperparameters.json) |
| Random Forest | 0.789 | `../models/base_models/` | [View](hyperparameters/RF_hyperparameters.json) |

## Technical Documentation

### Hyperparameter Documentation
- **20 Complete JSON Files**: Every model has full hyperparameter specifications
- **Bayesian Optimization Results**: Search spaces and optimal configurations
- **Training Details**: Hardware, memory, timing, and convergence information
- **Performance Metrics**: Spearman correlation, MSE, cross-validation scores

### Training Procedures
- **Data Preprocessing**: Sequence encoding, GC content calculation, normalization
- **Cross-Validation**: 5-fold nested validation strategy
- **Optimization Protocol**: Bayesian optimization with Optuna
- **Hardware Requirements**: GPU specifications and memory usage

### Model Architectures
- **Layer-by-Layer Specifications**: Complete technical details
- **Parameter Counts**: Exact parameter numbers for each component
- **Input/Output Dimensions**: Data flow through each architecture
- **GC Integration Methods**: Biological feature incorporation techniques

## Performance Analysis

### Benchmark Comparisons
```
ChromeCRISPR Results:
CNN_GRU+GC    (0.876) ━━ NEW BENCHMARK
deepCNN+GC    (0.873) ━━ Second Best
CNN_BiLSTM+GC (0.870) ━━ Third Best

Previous State-of-the-Art:
DeepHF         (0.867) ━━ Surpassed
AttCRISPR      (0.872) ━━ Surpassed
```

### Key Performance Insights
1. **Hybrid Advantage**: CNN-RNN combinations outperform individual architectures
2. **GRU Superiority**: GRU performs better than LSTM in hybrid models
3. **GC Content Impact**: Consistent improvement across most model types
4. **Depth Benefits**: Deeper models generally perform better than base models

## Quick Start Guide

### Using the Best Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load best performing model
model = ChromeCRISPRModel.load_from_file('../models/chromecrispr_hybrid_models/CNN_GRU+GC.pth')

# Make predictions
predictions = model.predict(your_sequence_data)
```

### Loading Model Hyperparameters
```python
import json

# Load hyperparameters for any model
with open('hyperparameters/CNN_GRU+GC_hyperparameters.json', 'r') as f:
    config = json.load(f)

learning_rate = config['hyperparameters']['learning_rate']['best_value']
batch_size = config['hyperparameters']['batch_size']['best_value']
dropout = config['hyperparameters']['dropout_rate']['best_value']
```

### Comparing Model Performance
```python
from src.evaluation import ModelEvaluator

# Compare any set of models
models = [
    '../models/chromecrispr_hybrid_models/CNN_GRU+GC.pth',
    '../models/deep_models_with_gc/deepCNN+GC.pth'
]

evaluator = ModelEvaluator()
results = evaluator.compare_models(test_data, models)
```

## Documentation Structure

```
docs/
├── hyperparameters/           # 20 JSON files + performance table
│   ├── CNN_GRU+GC_hyperparameters.json
│   ├── deepCNN+GC_hyperparameters.json
│   ├── [all other model hyperparameters...]
│   └── README.md              # Complete performance ranking table
├── training_procedures/       # Training protocols and methodologies
│   └── README.md
├── MODEL_ARCHITECTURES.md     # Technical architecture details
├── COMPREHENSIVE_MODEL_DOCUMENTATION.md  # Comparative analysis
└── README.md                  # This navigation file
```

## Finding Specific Information

### For Researchers
- **Performance Rankings**: `hyperparameters/README.md`
- **Architecture Comparisons**: `MODEL_ARCHITECTURES.md`
- **Training Details**: `training_procedures/README.md`
- **Benchmark Analysis**: `COMPREHENSIVE_MODEL_DOCUMENTATION.md`

### For Developers
- **Model Loading**: Check specific hyperparameter JSON files
- **API Usage**: See example code in individual README files
- **Integration**: Review architecture specifications
- **Optimization**: Compare hyperparameter configurations

## Advanced Usage

### Custom Model Training
```python
# Use hyperparameters from documentation
config = load_hyperparameters('hyperparameters/CNN_GRU+GC_hyperparameters.json')

model = ChromeCRISPRModel(
    architecture='cnn_gru_gc',
    learning_rate=config['learning_rate'],
    batch_size=config['batch_size'],
    dropout=config['dropout_rate']
)

model.train(X_train, y_train)
```

### Ensemble Methods
```python
# Combine multiple top models
models = [
    load_model('CNN_GRU+GC.pth'),
    load_model('deepCNN+GC.pth'),
    load_model('CNN_BiLSTM+GC.pth')
]

ensemble_predictions = ensemble_predict(models, test_data)
```

## Support & Resources

- **Model Files**: Located in `../models/`
- **Source Code**: Available in `../src/` directory
- **Examples**: See usage examples in each model directory README
- **Issues**: Report bugs or request features via GitHub Issues

## Key Achievements

1. **New Benchmark**: CNN_GRU+GC achieves 0.876 Spearman (surpasses DeepHF 0.867, AttCRISPR 0.872)
2. **Complete Documentation**: All 20 models fully specified with hyperparameters
3. **Hybrid Innovation**: Novel CNN-RNN fusion architectures
4. **Biological Integration**: GC content enhances domain-specific performance
5. **Publication Ready**: Comprehensive technical documentation

---
**Navigation**: Use this file as your central hub for accessing all ChromeCRISPR documentation, models, and technical specifications.
