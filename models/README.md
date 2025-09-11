# ChromeCRISPR: Complete Model Collection (20 Models)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17058362.svg)](https://doi.org/10.5281/zenodo.17058362)

## Best Model: CNN_GRU+GC
- **Performance**: 0.876 Spearman, 0.0093 MSE
- **Location**: `chromecrispr_hybrid_models/CNN_GRU+GC.pth`
- **Hyperparameters**: [View Details](../docs/hyperparameters/CNN_GRU+GC_hyperparameters.json)

## Model Performance Overview

| Model | Spearman | MSE | Architecture | Parameters |
|-------|----------|-----|--------------|------------|
| **CNN_GRU+GC** | **0.876** | **0.0093** | Hybrid CNN-GRU + GC | 6.12M |
| deepCNN+GC | 0.873 | 0.0093 | Deep CNN + GC | 665K |
| CNN_BiLSTM+GC | 0.870 | 0.0096 | Hybrid CNN-BiLSTM + GC | 20.41M |
| CNN_LSTM+GC | 0.867 | 0.0115 | Hybrid CNN-LSTM + GC | 7.995M |
| deepBiLSTM+GC | 0.867 | 0.0098 | Deep BiLSTM + GC | 5.99M |
| deepGRU+GC | 0.867 | 0.0098 | Deep GRU + GC | 1.82M |
| deepGRU | 0.868 | 0.0099 | Deep GRU | 1.81M |
| deepCNN | 0.869 | 0.0098 | Deep CNN | 665K |
| deepLSTM | 0.862 | 0.0103 | Deep LSTM | 2.34M |
| deepBiLSTM | 0.862 | 0.0104 | Deep BiLSTM | 5.99M |
| deepLSTM+GC | 0.860 | 0.0104 | Deep LSTM + GC | 2.34M |
| LSTM+GC | 0.856 | 0.0112 | LSTM + GC | 1.17M |
| BiLSTM+GC | 0.855 | 0.0110 | BiLSTM + GC | 2.82M |
| BiLSTM | 0.843 | 0.0120 | BiLSTM | 2.82M |
| GRU+GC | 0.840 | 0.0122 | GRU + GC | 907K |
| GRU | 0.837 | 0.0121 | GRU | 907K |
| LSTM | 0.837 | 0.0122 | LSTM | 1.17M |
| CNN | 0.793 | 0.0161 | CNN | 332K |
| CNN+GC | 0.781 | 0.0170 | CNN + GC | 332K |
| RF | 0.789 | 0.0161 | Random Forest | N/A |

## Model Organization

### Base Models (5 models)
Standard implementations of individual architectures:

| Model | File | Hyperparameters | Performance |
|-------|------|----------------|--------------|
| **Random Forest** | `base_models/RF.joblib` | [View](../docs/hyperparameters/RF_hyperparameters.json) | 0.789 Spearman |
| **CNN** | `base_models/CNN.pth` | [View](../docs/hyperparameters/CNN_hyperparameters.json) | 0.793 Spearman |
| **GRU** | `base_models/GRU.pth` | [View](../docs/hyperparameters/GRU_hyperparameters.json) | 0.837 Spearman |
| **LSTM** | `base_models/LSTM.pth` | [View](../docs/hyperparameters/LSTM_hyperparameters.json) | 0.837 Spearman |
| **BiLSTM** | `base_models/BiLSTM.pth` | [View](../docs/hyperparameters/BiLSTM_hyperparameters.json) | 0.843 Spearman |

### Base Models + GC Content (4 models)
Base models with biological GC content integration:

| Model | File | Hyperparameters | Performance | GC Impact |
|-------|------|----------------|--------------|-----------|
| **CNN+GC** | `base_models_with_gc/CNN+GC.pth` | [View](../docs/hyperparameters/CNN+GC_hyperparameters.json) | 0.781 Spearman | -0.012 |
| **GRU+GC** | `base_models_with_gc/GRU+GC.pth` | [View](../docs/hyperparameters/GRU+GC_hyperparameters.json) | 0.840 Spearman | +0.003 |
| **LSTM+GC** | `base_models_with_gc/LSTM+GC.pth` | [View](../docs/hyperparameters/LSTM+GC_hyperparameters.json) | 0.856 Spearman | +0.019 |
| **BiLSTM+GC** | `base_models_with_gc/BiLSTM+GC.pth` | [View](../docs/hyperparameters/BiLSTM+GC_hyperparameters.json) | 0.855 Spearman | +0.012 |

### Deep Models (4 models)
Enhanced versions with additional layers:

| Model | File | Hyperparameters | Performance |
|-------|------|----------------|--------------|
| **deepCNN** | `deep_models/deepCNN.pth` | [View](../docs/hyperparameters/deepCNN_hyperparameters.json) | 0.869 Spearman |
| **deepGRU** | `deep_models/deepGRU.pth` | [View](../docs/hyperparameters/deepGRU_hyperparameters.json) | 0.868 Spearman |
| **deepLSTM** | `deep_models/deepLSTM.pth` | [View](../docs/hyperparameters/deepLSTM_hyperparameters.json) | 0.862 Spearman |
| **deepBiLSTM** | `deep_models/deepBiLSTM.pth` | [View](../docs/hyperparameters/deepBiLSTM_hyperparameters.json) | 0.862 Spearman |

### Deep Models + GC Content (4 models)
Deep models with biological feature integration:

| Model | File | Hyperparameters | Performance | GC Impact |
|-------|------|----------------|--------------|-----------|
| **deepCNN+GC** | `deep_models_with_gc/deepCNN+GC.pth` | [View](../docs/hyperparameters/deepCNN+GC_hyperparameters.json) | 0.873 Spearman | +0.004 |
| **deepGRU+GC** | `deep_models_with_gc/deepGRU+GC.pth` | [View](../docs/hyperparameters/deepGRU+GC_hyperparameters.json) | 0.867 Spearman | -0.001 |
| **deepLSTM+GC** | `deep_models_with_gc/deepLSTM+GC.pth` | [View](../docs/hyperparameters/deepLSTM+GC_hyperparameters.json) | 0.860 Spearman | -0.002 |
| **deepBiLSTM+GC** | `deep_models_with_gc/deepBiLSTM+GC.pth` | [View](../docs/hyperparameters/deepBiLSTM+GC_hyperparameters.json) | 0.867 Spearman | +0.005 |

### ChromeCRISPR Hybrid Models (3 models)
Novel hybrid architectures combining CNN and RNN components:

| Model | File | Hyperparameters | Performance |
|-------|------|----------------|--------------|
| **CNN_GRU+GC** | `chromecrispr_hybrid_models/CNN_GRU+GC.pth` | [View](../docs/hyperparameters/CNN_GRU+GC_hyperparameters.json) | **0.876 Spearman** |
| **CNN_LSTM+GC** | `chromecrispr_hybrid_models/CNN_LSTM+GC.pth` | [View](../docs/hyperparameters/CNN_LSTM+GC_hyperparameters.json) | 0.867 Spearman |
| **CNN_BiLSTM+GC** | `chromecrispr_hybrid_models/CNN_BiLSTM+GC.pth` | [View](../docs/hyperparameters/CNN_BiLSTM+GC_hyperparameters.json) | 0.870 Spearman |

## Quick Model Access

### Best Performing Models
```bash
# Top 3 Models
1. CNN_GRU+GC     (0.876) - chromecrispr_hybrid_models/
2. deepCNN+GC     (0.873) - deep_models_with_gc/
3. CNN_BiLSTM+GC  (0.870) - chromecrispr_hybrid_models/
```

### Most Efficient Models
```bash
# Best performance per parameter
1. deepCNN+GC     (0.873, 665K params) - Most efficient
2. deepCNN        (0.869, 665K params) - Good balance
3. CNN_GRU+GC     (0.876, 6.12M params) - Best performance
```

## Model Specifications

### Input Format
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding (84 features)
- **GC Content**: Single feature (0-1 range)
- **Total Features**: 85 (with GC) or 84 (without GC)

### Hardware Requirements
- **Training**: NVIDIA V100 Volta GPU (32GB HBM2)
- **Memory Usage**: 0.8GB - 31GB depending on model
- **Training Time**: 2.5 - 11.8 hours per model

### Data Requirements
- **Dataset**: DeepHF (60,000 sgRNAs from 20,000 genes)
- **Split**: 70% train, 15% validation, 15% test
- **Cross-validation**: 5-fold nested validation

## Usage Examples

### Load and Use Best Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load best performing model
model = ChromeCRISPRModel.load_from_file('chromecrispr_hybrid_models/CNN_GRU+GC.pth')

# Make predictions
predictions = model.predict(your_sequence_data)
```

### Compare Multiple Models
```python
from src.evaluation import ModelEvaluator

models = [
    'chromecrispr_hybrid_models/CNN_GRU+GC.pth',
    'deep_models_with_gc/deepCNN+GC.pth',
    'chromecrispr_hybrid_models/CNN_BiLSTM+GC.pth'
]

evaluator = ModelEvaluator()
results = evaluator.compare_models(models, test_data)
```

## Documentation Links

- **[Complete Hyperparameter Documentation](../docs/hyperparameters/)** - All 19 models with full specifications
- **[Training Procedures](../docs/training_procedures/)** - Detailed training protocols
- **[Model Architectures](../docs/MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../docs/COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

## Model Categories Legend

- **Best Model**: Highest performance across all metrics
- **Top Performer**: Among top 3 models
- **Efficient**: Best performance per parameter ratio
- **Hybrid**: ChromeCRISPR novel architecture
- **GC Integration**: Includes biological GC content feature

## Support

For questions about specific models or usage:
- **Documentation**: Check hyperparameter files for complete specifications
- **Issues**: [GitHub Issues](../../issues)
- **Contact**: amir_dp@sfu.ca

---
**Note**: All models are trained and validated using identical protocols. Performance metrics represent Spearman correlation on held-out test set with 5-fold cross-validation.
