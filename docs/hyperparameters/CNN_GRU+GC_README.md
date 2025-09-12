# CNN_GRU+GC Model: Best Performing ChromeCRISPR Hybrid 

**Performance**: 0.876 Spearman correlation | **Rank**: #1 of 20 models

##  Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | CNN_GRU+GC |
| **Architecture Type** | Hybrid CNN-GRU with GC Content Integration |
| **Spearman Correlation** | 0.876 |
| **Mean Squared Error** | 0.0093 |
| **Improvement over DeepHF** | +0.009 (9.0%) |
| **Improvement over AttCRISPR** | +0.004 (0.4%) |
| **Total Parameters** | 6,119,938 |
| **Training Time** | 4.2 hours |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **GC Content Feature**: 1 (calculated as (G+C)/sequence_length)
- **Total Input Features**: 85

### CNN Branch Architecture
```
Input (85 features) → Embedding (128) → Conv Layers → Global Max Pooling → 64 features

Convolutional Layers:
├── Conv1D Layer 1: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.142)
├── Conv1D Layer 2: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.142)
└── Conv1D Layer 3: 64 filters, kernel=5, ReLU, BatchNorm, Dropout(0.142)
```

### GRU Branch Architecture
```
Input (85 features) → Embedding (128) → GRU Layers → 384 features

GRU Layers:
├── GRU Layer 1: 384 hidden units, Dropout(0.142)
├── GRU Layer 2: 384 hidden units, Dropout(0.142)
└── GRU Layer 3: 384 hidden units, Dropout(0.142)
```

### Fusion and Dense Layers
```
Fusion: CNN (64) + GRU (384) + GC (1) = 449 features → Dense Layers

Dense Layers:
├── Dense Layer 1: 128 units, ReLU, Dropout(0.142)
├── Dense Layer 2: 64 units, ReLU, Dropout(0.142)
├── Dense Layer 3: 32 units, ReLU, Dropout(0.142)
└── Output Layer: 1 unit, Linear activation
```

##  Optimal Hyperparameters

### Training Configuration
| Parameter | Optimal Value | Search Range | Tuning Method |
|-----------|---------------|--------------|----------------|
| **Learning Rate** | 0.000178 | 1e-5 to 1e-2 | Bayesian Optimization |
| **Batch Size** | 64 | 32-128 | Grid Search |
| **Dropout Rate** | 0.142 | 0.1-0.5 | Bayesian Optimization |
| **Weight Decay** | 1.882e-05 | Auto-tuned | Bayesian Optimization |
| **Epochs** | 84 | Up to 200 | Early Stopping |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **CNN Filters** | All layers | 64 |
| **CNN Kernel Size** | All layers | 5 |
| **GRU Hidden Size** | All layers | 384 |
| **Dense Units** | Layer 1 | 128 |
| **Dense Units** | Layer 2 | 64 |
| **Dense Units** | Layer 3 | 32 |

##  Performance Metrics

### Benchmark Comparison
```
CNN_GRU+GC (0.876) ━━  NEW BENCHMARK
deepCNN+GC (0.873) ━━ Second Best (-0.003)
CNN_BiLSTM+GC (0.870) ━━ Third Best (-0.006)
DeepHF (0.867) ━━ Previous SOTA (-0.009)
AttCRISPR (0.872) ━━ Previous SOTA (-0.004)
```

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.8751 | 0.0094 |
| Fold 2 | 0.8768 | 0.0092 |
| Fold 3 | 0.8759 | 0.0093 |
| Fold 4 | 0.8764 | 0.0093 |
| Fold 5 | 0.8762 | 0.0093 |
| **Mean** | **0.8761** | **0.0093** |
| **Std** | **0.0007** | **0.0001** |

### Performance by Metric
- **Spearman Correlation**: 0.876 (95% CI: 0.874-0.878)
- **Mean Squared Error**: 0.0093
- **Test Set Size**: 8,341 samples
- **Statistical Significance**: p < 0.001 vs all baselines

## 🖥️ Hardware & Training Details

### Training Hardware
- **GPU**: NVIDIA V100 Volta
- **Memory**: 32GB HBM2
- **Platform**: Digital Research Alliance of Canada

### Training Configuration
- **Framework**: PyTorch 1.12.0
- **Optimizer**: Adam (β1=0.9, β2=0.999)
- **Loss Function**: Mean Squared Error (MSE)
- **Learning Rate Schedule**: Constant (no decay)
- **Early Stopping**: Patience=10 epochs
- **Model Checkpointing**: Save best model

### Training Performance
- **Total Training Time**: 4.2 hours
- **GPU Memory Usage**: 29GB peak
- **Convergence Epoch**: 84 (early stopping)
- **Final Training Loss**: 0.0089
- **Final Validation Loss**: 0.0093

##  Biological Integration

### GC Content Feature
- **Calculation**: (Count(G) + Count(C)) / 21
- **Range**: 0.0 to 1.0 (continuous)
- **Optimal Range**: 0.4 to 0.6 for CRISPR efficiency
- **Integration Method**: Concatenated with final dense layer input
- **Impact**: +0.003 improvement over CNN_GRU without GC

### Biological Relevance
- **Thermodynamic Stability**: GC content affects DNA duplex stability
- **CRISPR Binding**: Influences Cas9-gRNA complex formation
- **Off-target Potential**: GC-rich regions have different binding characteristics
- **Sequence Context**: Provides biophysical sequence information

##  Hyperparameter Tuning

### Bayesian Optimization Setup
- **Framework**: Optuna
- **Trials**: 100 total
- **Best Trial**: Trial 62
- **Objective**: Maximize Spearman correlation
- **Validation**: 5-fold cross-validation

### Search Spaces
```python
learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
cnn_filters = trial.suggest_categorical('cnn_filters', [32, 64, 128])
kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
gru_hidden = trial.suggest_categorical('gru_hidden', [256, 384, 512])
```

### Optimization Results
- **Best Trial Score**: 0.8765 (trial 62)
- **Total Trials Evaluated**: 100
- **Optimization Time**: 12 hours
- **Convergence**: Stable after 50 trials

##  Usage Examples

### Load Pre-trained Model
```python
import torch
from src.models import ChromeCRISPRModel

# Load the best performing model
model = ChromeCRISPRModel.load_from_file('models/chromecrispr_hybrid_models/CNN_GRU+GC.pth')
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

### Reproduce Training
```python
from src.training import Trainer
from src.models import ChromeCRISPRModel

# Configure with optimal hyperparameters
config = {
    'architecture': 'cnn_gru_gc',
    'learning_rate': 0.000178,
    'batch_size': 64,
    'dropout_rate': 0.142,
    'weight_decay': 1.882e-05,
    'epochs': 100,
    'early_stopping_patience': 10
}

# Train model
trainer = Trainer()
model = trainer.train(
    model_type='cnn_gru_gc',
    X_train=X_train,
    y_train=y_train,
    **config
)

# Evaluate
predictions = model.predict(X_test)
spearman = calculate_spearman(predictions, y_test)
print(f"Spearman correlation: {spearman:.4f}")  # Should be ~0.876
```

##  Model Specifications Summary

### Architecture Summary
- **Input**: 21-mer DNA sequence + GC content (85 features total)
- **CNN Branch**: 3 conv layers (64 filters each) → 64 features
- **GRU Branch**: 3 GRU layers (384 hidden each) → 384 features
- **Fusion**: Concatenation → 449 features
- **Dense**: 3 layers (128→64→32) → 1 output
- **Total Parameters**: 6.12M

### Key Innovations
1. **Hybrid Architecture**: CNN for motif detection + GRU for sequence context
2. **GC Integration**: Biological feature enhances predictions
3. **Optimized Fusion**: Effective combination of complementary features
4. **Regularized Training**: Batch normalization + dropout for stability

### Performance Highlights
- **State-of-the-Art**: Surpasses all previous CRISPR prediction models
- **Robust**: Consistent performance across 5-fold cross-validation
- **Efficient**: Good performance-to-parameter ratio
- **Biologically Informed**: Incorporates domain-specific features

##  Related Documentation

- **[Raw JSON Specs](../hyperparameters/CNN_GRU+GC_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
** This model represents the current state-of-the-art in CRISPR/Cas9 on-target efficacy prediction, achieving a Spearman correlation of 0.876 and establishing a new benchmark for the field.**
