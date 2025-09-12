# BiLSTM+GC Model: Bidirectional LSTM with GC Content Integration

**Performance**: 0.855 Spearman correlation | **Rank**: #8 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | BiLSTM+GC |
| **Architecture Type** | Bidirectional LSTM + GC Content Integration |
| **Spearman Correlation** | 0.855 |
| **Mean Squared Error** | 0.0110 |
| **Total Parameters** | 272,129 |
| **Training Time** | 4.7 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **GC Content**: 1 feature (calculated as (G+C)/21)
- **Embedding Dimension**: 128
- **Total Input Features**: 85

### BiLSTM Architecture
```
Input (85 features) → Embedding (128) → BiLSTM Layers → Dense Layers → 1 output

BiLSTM Layers:
├── BiLSTM Layer 1: 128 hidden units × 2 directions, Dropout(0.2)
└── BiLSTM Layer 2: 128 hidden units × 2 directions, Dropout(0.2)
```

### Biological Integration
```
BiLSTM Output (256) + GC Content (1) = 257 features → Dense Layers → 1 output

Dense Layers:
├── Dense Layer 1: 128 units, ReLU, Dropout(0.2)
├── Dense Layer 2: 64 units, ReLU, Dropout(0.2)
└── Output Layer: 1 unit, Linear activation
```

## Optimal Hyperparameters

### Training Configuration
| Parameter | Optimal Value | Search Range | Tuning Method |
|-----------|---------------|--------------|----------------|
| **Learning Rate** | 0.000142 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.2 | 0.1-0.5 | Bayesian Optimization |
| **Weight Decay** | 2.67e-05 | Auto-tuned | Bayesian Optimization |
| **Epochs** | 88 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **BiLSTM Hidden Size** | All layers | 128 |
| **BiLSTM Layers** | Count | 2 |
| **Directions** | Per layer | 2 (bidirectional) |
| **Total Hidden Features** | Output | 256 |
| **GC Integration** | Method | Concatenation |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.854 | 0.0112 |
| Fold 2 | 0.856 | 0.0110 |
| Fold 3 | 0.855 | 0.0111 |
| Fold 4 | 0.856 | 0.0110 |
| Fold 5 | 0.855 | 0.0111 |
| **Mean** | **0.855** | **0.0110** |
| **Std** | **0.001** | **0.0001** |

### Performance by Metric
- **Spearman Correlation**: 0.855 (95% CI: 0.852-0.858)
- **Mean Squared Error**: 0.0110
- **Test Set Size**: 8,341 samples
- **Improvement over BiLSTM**: +0.012 Spearman (1.4%), -0.0010 MSE

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100 Volta
- **Memory Usage**: 1.4GB
- **Platform**: Digital Research Alliance of Canada

### Training Configuration
- **Framework**: PyTorch 1.12.0
- **Optimizer**: Adam (β1=0.9, β2=0.999, weight_decay=2.67e-05)
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience=10 epochs
- **Model Checkpointing**: Save best model

### Training Performance
- **Total Training Time**: 4.7 hours
- **GPU Memory Usage**: 1.4GB peak
- **Convergence Epoch**: 91 (best epoch)
- **Final Training Loss**: 0.0105
- **Final Validation Loss**: 0.0110

## Biological Integration

### GC Content Feature
- **Calculation**: (Count(G) + Count(C)) / 21
- **Range**: 0.0 to 1.0 (continuous)
- **Optimal Range**: 0.4 to 0.6 for CRISPR efficiency
- **Integration Method**: Concatenated with BiLSTM final hidden state
- **Impact**: +0.012 improvement over base BiLSTM
- **Statistical Significance**: p < 0.05

### Biological Relevance
- **Sequence Thermodynamics**: GC content affects DNA duplex stability
- **CRISPR Binding**: Influences Cas9-gRNA complex formation
- **Context Enhancement**: Provides biological context for bidirectional processing
- **Stability Correlation**: GC-rich regions have different binding characteristics

## Bidirectional Processing with Biological Context

### Enhanced Context Awareness
- **Forward Pass**: Processes sequence from 5' to 3' with GC context
- **Backward Pass**: Processes sequence from 3' to 5' with GC context
- **Biological Integration**: GC content informs both directional processing
- **Combined Output**: Bidirectional features + biological context

### Sequence Understanding
- **PAM-Proximal Context**: Bidirectional information around PAM site
- **Seed Region Enhancement**: GC context for specificity determination
- **Global Sequence Context**: Full biological sequence representation
- **Thermodynamic Awareness**: GC content informs stability predictions

## Hyperparameter Tuning

### Bayesian Optimization Setup
- **Framework**: Optuna
- **Trials**: 100 total
- **Best Trial**: Trial 45
- **Objective**: Maximize Spearman correlation
- **Validation**: 5-fold cross-validation

### Search Spaces
```python
learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
```

### Optimization Results
- **Best Trial Score**: 0.8551 (trial 45)
- **Total Trials Evaluated**: 100
- **Optimization Time**: 9 hours
- **Convergence**: Stable after 70 trials

## Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the BiLSTM+GC model
model = ChromeCRISPRModel.load_from_file('models/base_models_with_gc/BiLSTM+GC.pth')
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

### Custom Training with GC Integration
```python
from src.training import Trainer
from src.models import ChromeCRISPRModel

# Configure BiLSTM+GC model
config = {
    'architecture': 'bilstm_gc',
    'learning_rate': 0.000142,
    'batch_size': 64,
    'dropout_rate': 0.2,
    'weight_decay': 2.67e-05,
    'hidden_size': 128,
    'num_layers': 2,
    'bidirectional': True,
    'use_gc_content': True,
    'epochs': 100,
    'early_stopping_patience': 10
}

# Train model
trainer = Trainer()
model = trainer.train(
    model_type='bilstm_gc',
    X_train=X_train,
    y_train=y_train,
    **config
)

# Evaluate
predictions = model.predict(X_test)
spearman = calculate_spearman(predictions, y_test)
print(f"Spearman correlation: {spearman:.4f}")  # Expected: ~0.855
```

## Model Specifications Summary

### Architecture Summary
- **Input**: 21-mer DNA sequence (84 features) + GC content (1 feature)
- **BiLSTM Branch**: 2 bidirectional layers (128 hidden × 2 directions each) → 256 features
- **Biological Fusion**: BiLSTM features + GC content → 257 features
- **Dense**: 2 layers (128→64) → 1 output
- **Total Parameters**: 272K

### Key Features
1. **Bidirectional Processing**: Forward + backward sequence analysis
2. **GC Content Integration**: Biological feature enhancement
3. **Contextual Richness**: Most comprehensive sequence understanding
4. **Biological Awareness**: Thermodynamic and stability considerations

### Performance Characteristics
- **Top Performer**: 8th best overall model
- **Biological Enhancement**: Significant improvement with GC integration
- **Stable Training**: Consistent performance across folds
- **Resource Efficient**: Lower memory usage than deeper models

## Comparative Analysis

### Performance Comparison
| Model | Spearman | MSE | Parameters | GC Integration |
|-------|----------|-----|------------|----------------|
| **BiLSTM+GC** | **0.855** | **0.0110** | **272K** | **Yes** |
| BiLSTM | 0.843 | 0.0120 | 2.82M | No |
| LSTM+GC | 0.856 | 0.0112 | 1.17M | Yes |
| GRU+GC | 0.840 | 0.0122 | 907K | Yes |

### GC Content Impact
- **Without GC**: BiLSTM = 0.843
- **With GC**: BiLSTM+GC = 0.855
- **Improvement**: +0.012 (1.4% increase)
- **Statistical Significance**: p < 0.05

### Strengths
- **Biological Integration**: Effective use of GC content
- **Bidirectional Advantage**: Full sequence context
- **Parameter Efficiency**: Lower parameter count than base BiLSTM
- **Performance Consistency**: Stable across validation folds

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/BiLSTM+GC_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**BiLSTM+GC combines bidirectional sequence processing with biological GC content integration, achieving 0.855 Spearman correlation with significant improvement over the base model.**
