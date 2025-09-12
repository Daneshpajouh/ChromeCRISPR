# BiLSTM Model: Bidirectional LSTM Baseline

**Performance**: 0.843 Spearman correlation | **Rank**: #11 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | BiLSTM |
| **Architecture Type** | Bidirectional Long Short-Term Memory |
| **Spearman Correlation** | 0.843 |
| **Mean Squared Error** | 0.0120 |
| **Total Parameters** | 2,819,769 |
| **Training Time** | 4.5 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **Embedding Dimension**: 128
- **Total Input Features**: 84

### BiLSTM Architecture
```
Input (84 features) → Embedding (128) → BiLSTM Layers → Dense Layers → 1 output

BiLSTM Layers:
├── BiLSTM Layer 1: 128 hidden units × 2 directions, Dropout(0.2)
└── BiLSTM Layer 2: 128 hidden units × 2 directions, Dropout(0.2)
```

### Dense Layers
```
BiLSTM Output (256) → Dense Layers → 1 output

Dense Layers:
├── Dense Layer 1: 128 units, ReLU, Dropout(0.2)
├── Dense Layer 2: 64 units, ReLU, Dropout(0.2)
├── Dense Layer 3: 32 units, ReLU, Dropout(0.2)
└── Output Layer: 1 unit, Linear activation
```

## Optimal Hyperparameters

### Training Configuration
| Parameter | Optimal Value | Search Range | Tuning Method |
|-----------|---------------|--------------|----------------|
| **Learning Rate** | 0.001 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Epochs** | 100 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **BiLSTM Hidden Size** | All layers | 128 |
| **BiLSTM Layers** | Count | 2 |
| **Directions** | Per layer | 2 (bidirectional) |
| **Total Hidden Features** | Output | 256 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |
| **Dense Units** | Layer 3 | 32 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.841 | 0.0122 |
| Fold 2 | 0.845 | 0.0118 |
| Fold 3 | 0.842 | 0.0120 |
| Fold 4 | 0.844 | 0.0119 |
| Fold 5 | 0.844 | 0.0119 |
| **Mean** | **0.843** | **0.0120** |
| **Std** | **0.002** | **0.0002** |

### Performance by Metric
- **Spearman Correlation**: 0.843 (95% CI: 0.840-0.846)
- **Mean Squared Error**: 0.0120
- **Test Set Size**: 9,000 samples
- **Cross-Validation Stability**: Standard deviation = 0.0029

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100
- **Memory Usage**: 12GB
- **Platform**: Digital Research Alliance of Canada

### Training Configuration
- **Framework**: PyTorch 1.12.0
- **Optimizer**: Adam (β1=0.9, β2=0.999, weight_decay=1e-5)
- **Optimizer Selection**: Fixed as Adam (not part of hyperparameter search)
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience=10 epochs
- **Model Checkpointing**: Save best model

### Training Performance
- **Total Training Time**: 4.5 hours
- **GPU Memory Usage**: 12GB peak
- **Convergence Epoch**: 100 (completed)
- **Final Training Loss**: 0.0115
- **Final Validation Loss**: 0.0120

## Bidirectional Processing Analysis

### Forward vs Backward Processing
- **Forward Pass**: Processes sequence from 5' to 3' end
- **Backward Pass**: Processes sequence from 3' to 5' end
- **Combined Output**: Concatenated features from both directions
- **Total Features**: 256 (128 × 2 directions)

### Sequence Context Enhancement
- **PAM Context**: Bidirectional information around PAM site
- **Seed Region**: Full context for specificity determination
- **Position Dependencies**: Both upstream and downstream effects
- **Global Sequence**: Complete sequence representation

## Hyperparameter Tuning

### Bayesian Optimization Setup
- **Framework**: Optuna
- **Trials**: 100 total
- **Best Trial**: Trial 52
- **Objective**: Maximize Spearman correlation
- **Validation**: 5-fold cross-validation

### Search Spaces
```python
learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
```

### Optimization Results
- **Best Trial Score**: 0.8432 (trial 52)
- **Total Trials Evaluated**: 100
- **Optimization Time**: 8 hours
- **Convergence**: Stable after 60 trials

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the BiLSTM model
model = ChromeCRISPRModel.load_from_file('models/base_models/BiLSTM.pth')
model.eval()

# Prepare input sequence
sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCG"  # 21-mer
X = model.preprocess_sequence(sequence)

# Make prediction
with torch.no_grad():
    prediction = model(X)
    activity_score = prediction.item()
    print(f"Predicted CRISPR activity: {activity_score:.4f}")
```

### Custom Training
```python
from src.training import Trainer
from src.models import ChromeCRISPRModel

# Configure BiLSTM model
config = {
    'architecture': 'bilstm',
    'learning_rate': 0.001,
    'batch_size': 64,
    'dropout_rate': 0.2,
    'hidden_size': 128,
    'num_layers': 2,
    'bidirectional': True,
    'epochs': 100,
    'early_stopping_patience': 10
}

# Train model
trainer = Trainer()
model = trainer.train(
    model_type='bilstm',
    X_train=X_train,
    y_train=y_train,
    **config
)

# Evaluate
predictions = model.predict(X_test)
spearman = calculate_spearman(predictions, y_test)
print(f"Spearman correlation: {spearman:.4f}")  # Expected: ~0.843
```

## Model Specifications Summary

### Architecture Summary
- **Input**: 21-mer DNA sequence (84 features)
- **BiLSTM Branch**: 2 bidirectional layers (128 hidden × 2 directions each) → 256 features
- **Dense**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 2.82M

### Key Features
1. **Bidirectional Processing**: Forward + backward sequence analysis
2. **Enhanced Context**: Complete sequence representation
3. **Memory Capacity**: Superior long-range dependency modeling
4. **Contextual Richness**: Most comprehensive baseline model

### Performance Characteristics
- **Best Baseline Model**: Highest performance among base architectures
- **Stable Training**: Consistent convergence across folds
- **Computational Efficiency**: Good performance-to-resource ratio
- **Sequence Understanding**: Superior contextual awareness

## Comparative Analysis

### vs Other Base Models
| Model | Spearman | MSE | Parameters | Training Time |
|-------|----------|-----|------------|---------------|
| **BiLSTM** | **0.843** | **0.0120** | **2.82M** | **4.5h** |
| LSTM | 0.837 | 0.0122 | 1.17M | 3.8h |
| GRU | 0.837 | 0.0121 | 907K | 3.2h |
| CNN | 0.793 | 0.0161 | 125K | 2.5h |

### Strengths
- **Contextual Superiority**: Best sequence understanding
- **Bidirectional Advantage**: Full positional context
- **Stability**: Consistent performance across validation folds
- **Scalability**: Good foundation for larger architectures

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/BiLSTM_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**BiLSTM provides the strongest baseline performance with 0.843 Spearman correlation, offering comprehensive bidirectional sequence analysis for CRISPR applications.**
