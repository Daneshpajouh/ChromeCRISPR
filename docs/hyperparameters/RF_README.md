# RF Model: Random Forest Baseline

**Performance**: 0.789 Spearman correlation | **Rank**: #18 of 20 models

## Model Overview

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | RF |
| **Architecture Type** | Random Forest Ensemble |
| **Spearman Correlation** | 0.789 |
| **Mean Squared Error** | 0.0161 |
| **Total Parameters** | N/A (tree-based) |
| **Training Time** | Instantaneous |

## 🏗️ Architecture Details

### Input Processing
- **Sequence Length**: 21 nucleotides (20 bases + PAM)
- **Encoding**: One-hot encoding + GC content
- **Base Features**: 84 (4 nucleotides × 21 positions)
- **GC Content**: 1 feature (calculated as (G+C)/21)
- **Total Input Features**: 85

### Random Forest Architecture
```
Input (85 features) → Random Forest Ensemble → Prediction

Ensemble Details:
├── 100 Decision Trees (n_estimators=100)
├── Max Depth: Unlimited (default)
├── Min Samples Split: 2
├── Min Samples Leaf: 1
├── Bootstrap Sampling: Enabled
└── Random State: 42 (reproducibility)
```

## Optimal Hyperparameters

### Training Configuration
| Parameter | Optimal Value | Search Range | Tuning Method |
|-----------|---------------|--------------|----------------|
| **n_estimators** | 100 | 50-500 | Grid Search |
| **max_depth** | None | 10-None | Grid Search |
| **min_samples_split** | 2 | 2-10 | Grid Search |
| **min_samples_leaf** | 1 | 1-4 | Grid Search |
| **max_features** | auto | auto/sqrt/log2 | Grid Search |

### Architecture Parameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **Number of Trees** | n_estimators | 100 |
| **Max Tree Depth** | max_depth | None |
| **Min Split Samples** | min_samples_split | 2 |
| **Min Leaf Samples** | min_samples_leaf | 1 |
| **Feature Selection** | max_features | auto |

## Performance Metrics

### Cross-Validation Results
| Fold | Spearman | MSE |
|------|----------|-----|
| Fold 1 | 0.781 | 0.0164 |
| Fold 2 | 0.788 | 0.0158 |
| Fold 3 | 0.792 | 0.0162 |
| Fold 4 | 0.786 | 0.0167 |
| Fold 5 | 0.791 | 0.0159 |
| **Mean** | **0.788** | **0.0162** |
| **Std** | **0.004** | **0.0003** |

### Performance by Metric
- **Spearman Correlation**: 0.789 (95% CI: 0.785-0.793)
- **Mean Squared Error**: 0.0161
- **Test Set Size**: 9,000 samples

## 🖥️ Hardware & Training Details

### Training Hardware
- **Platform**: CPU-based training
- **Memory Usage**: 2GB RAM
- **Training Time**: 45 seconds
- **Library**: scikit-learn 1.3.0

### Training Configuration
- **Algorithm**: RandomForestRegressor
- **Criterion**: MSE (default)
- **Bootstrap**: True
- **OOB Score**: False
- **n_jobs**: None (single-threaded)

## Feature Importance Analysis

### Top Sequence Positions
1. **Position 20** (PAM proximal): Highest importance
2. **GC Content**: Strong biological signal
3. **Position 1-5** (5' end): Guide RNA folding
4. **Central positions** (10-15): Target recognition

### Biological Insights
- **PAM-Proximal Effects**: Critical for Cas9 binding
- **GC Content Influence**: Affects duplex stability
- **5' End Importance**: RNA secondary structure
- **Central Sequence**: Target specificity determinants

## Hyperparameter Tuning

### Grid Search Setup
- **Framework**: scikit-learn GridSearchCV
- **Trials**: 36 parameter combinations
- **Best Trial**: n_estimators=100, max_depth=None
- **Objective**: Maximize Spearman correlation
- **Validation**: 5-fold cross-validation

### Search Spaces
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}
```

## Usage Examples

### Load Pre-trained Model
```python
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the RF model
rf_model = joblib.load('models/base_models/RF.joblib')

# Prepare input sequence (flatten for RF)
sequence_features = preprocess_sequence("ATCGATCGATCGATCGATCGATCGATCGATCGATCG")
X = sequence_features.reshape(1, -1)

# Make prediction
prediction = rf_model.predict(X)
activity_score = prediction[0]
print(f"Predicted CRISPR activity: {activity_score:.4f}")
```

### Feature Importance Analysis
```python
# Get feature importances
feature_importances = rf_model.feature_importances_

# Analyze top features
top_features = np.argsort(feature_importances)[-10:][::-1]
for idx in top_features:
    print(f"Feature {idx}: {feature_importances[idx]:.4f}")
```

## Model Specifications Summary

### Architecture Summary
- **Input**: 21-mer DNA sequence + GC content (85 features)
- **Algorithm**: Random Forest ensemble
- **Number of Trees**: 100 decision trees
- **Feature Selection**: Auto (sqrt(n_features))
- **Output**: Single regression value

### Key Features
1. **Ensemble Learning**: Multiple decision trees reduce overfitting
2. **Feature Importance**: Built-in interpretability
3. **Robust to Noise**: Handles outliers well
4. **No Feature Scaling**: Tree-based algorithm
5. **Fast Inference**: Quick prediction times

## Performance Comparison

### vs Deep Learning Models
- **RF (0.789)** vs **CNN_GRU+GC (0.876)**: -0.087 difference
- **Computational Cost**: 45 seconds vs 4.2 hours
- **Memory Usage**: 2GB vs 29GB
- **Interpretability**: High vs Low

### Use Cases
- **Baseline Comparison**: Traditional ML benchmark
- **Feature Analysis**: Biological insight generation
- **Resource Constraints**: CPU-only environments
- **Interpretability**: Feature importance analysis

## Related Documentation

- **[Raw JSON Specs](../hyperparameters/RF_hyperparameters.json)** - Complete technical specifications
- **[Training Procedures](../training_procedures/)** - Detailed training protocols
- **[Model Architectures](../MODEL_ARCHITECTURES.md)** - Technical architecture details
- **[Performance Analysis](../COMPREHENSIVE_MODEL_DOCUMENTATION.md)** - Comparative analysis

---
**Random Forest provides a robust traditional ML baseline with 0.789 Spearman correlation, offering interpretability and computational efficiency as a comparison point for deep learning approaches.**
