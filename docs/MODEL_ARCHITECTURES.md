# ChromeCRISPR Model Architectures

This document provides comprehensive text-based descriptions of all ChromeCRISPR model architectures, replacing visual diagrams with detailed specifications.

## Overview

ChromeCRISPR consists of three main categories of models:
1. **Base Models** - Standard implementations of individual architectures
2. **Deep Models** - Enhanced versions with additional layers
3. **Hybrid Models** - CNN-RNN combinations (ChromeCRISPR core)

## Base Models

### 1. Random Forest (RF)
- **Type**: Ensemble learning method
- **Implementation**: scikit-learn RandomForestRegressor
- **Parameters**: 100 estimators
- **Input**: Flattened one-hot encoded sgRNA sequences (84 features) + GC content
- **Output**: Single regression value (activity prediction)

### 2. Convolutional Neural Network (CNN)
- **Type**: Convolutional neural network
- **Implementation**: PyTorch
- **Architecture**:
  - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
  - Embedding Layer: 84 → 128 dimensions
  - Conv1D Layer 1: 128 filters, kernel_size=3, stride=1, padding=1, ReLU activation
  - Conv1D Layer 2: 128 filters, kernel_size=3, stride=1, padding=1, ReLU activation
  - Flatten Layer
  - Dense Layer 1: 64 units with batch normalization
  - Dense Layer 2: 1 unit (output)
- **Batch Size**: 64
- **Input Processing**: Sequence embedding to 128-dimensional tensor

### 3. Gated Recurrent Unit (GRU)
- **Type**: Recurrent neural network
- **Implementation**: PyTorch
- **Architecture**:
  - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
  - Embedding Layer: 84 → 128 dimensions
  - GRU Layer 1: 128 hidden units, bidirectional=False
  - GRU Layer 2: 128 hidden units, bidirectional=False
  - Dense Layer 1: 64 units with batch normalization
  - Dense Layer 2: 1 unit (output)
- **Sequence Processing**: Processes 21-mer sequences sequentially
- **Memory Management**: Gated mechanism for long-term dependencies

### 4. Long Short-Term Memory (LSTM)
- **Type**: Recurrent neural network
- **Implementation**: PyTorch
- **Architecture**:
  - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
  - Embedding Layer: 84 → 128 dimensions
  - LSTM Layer 1: 128 hidden units, bidirectional=False
  - LSTM Layer 2: 128 hidden units, bidirectional=False
  - Dense Layer 1: 64 units with batch normalization
  - Dense Layer 2: 1 unit (output)
- **Sequence Processing**: Processes 21-mer sequences sequentially
- **Memory Management**: Cell state and hidden state for long-term dependencies

### 5. Bidirectional Long Short-Term Memory (BiLSTM)
- **Type**: Bidirectional recurrent neural network
- **Implementation**: PyTorch
- **Architecture**:
  - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
  - Embedding Layer: 84 → 128 dimensions
  - BiLSTM Layer 1: 128 hidden units, bidirectional=True
  - BiLSTM Layer 2: 128 hidden units, bidirectional=True
  - Dense Layer 1: 64 units with batch normalization
  - Dense Layer 2: 1 unit (output)
- **Sequence Processing**: Processes sequences in both forward and reverse directions
- **Output Concatenation**: Forward and backward hidden states combined

## Base Models with GC Content

### 1. CNN+GC
- **Architecture**: Same as base CNN
- **GC Integration**: GC content added as single feature in final dense layer
- **Input**: 84 sequence features + 1 GC content feature = 85 total features

### 2. GRU+GC
- **Architecture**: Same as base GRU
- **GC Integration**: GC content concatenated with final GRU output before dense layers

### 3. LSTM+GC
- **Architecture**: Same as base LSTM
- **GC Integration**: GC content concatenated with final LSTM output before dense layers

### 4. BiLSTM+GC
- **Architecture**: Same as base BiLSTM
- **GC Integration**: GC content concatenated with final BiLSTM output before dense layers

## Deep Models

### 1. Deep CNN
- **Architecture**: Enhanced version of base CNN
- **Layers**:
  - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
  - Embedding Layer: 84 → 128 dimensions
  - Conv1D Layer 1: 128 filters, kernel_size=3, stride=1, padding=1, ReLU activation
  - Conv1D Layer 2: 128 filters, kernel_size=3, stride=1, padding=1, ReLU activation
  - Conv1D Layer 3: 128 filters, kernel_size=3, stride=1, padding=1, ReLU activation
  - Flatten Layer
  - Dense Layer 1: 128 units with batch normalization
  - Dense Layer 2: 64 units with batch normalization
  - Dense Layer 3: 32 units with batch normalization
  - Dense Layer 4: 1 unit (output)

### 2. Deep GRU
- **Architecture**: Enhanced version of base GRU
- **Layers**:
  - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
  - Embedding Layer: 84 → 128 dimensions
  - GRU Layer 1: 128 hidden units, bidirectional=False
  - GRU Layer 2: 128 hidden units, bidirectional=False
  - GRU Layer 3: 128 hidden units, bidirectional=False
  - Dense Layer 1: 128 units with batch normalization
  - Dense Layer 2: 64 units with batch normalization
  - Dense Layer 3: 32 units with batch normalization
  - Dense Layer 4: 1 unit (output)

### 3. Deep LSTM
- **Architecture**: Enhanced version of base LSTM
- **Layers**:
  - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
  - Embedding Layer: 84 → 128 dimensions
  - LSTM Layer 1: 128 hidden units, bidirectional=False
  - LSTM Layer 2: 128 hidden units, bidirectional=False
  - LSTM Layer 3: 128 hidden units, bidirectional=False
  - Dense Layer 1: 128 units with batch normalization
  - Dense Layer 2: 64 units with batch normalization
  - Dense Layer 3: 32 units with batch normalization
  - Dense Layer 4: 1 unit (output)

### 4. Deep BiLSTM
- **Architecture**: Enhanced version of base BiLSTM
- **Layers**:
  - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
  - Embedding Layer: 84 → 128 dimensions
  - BiLSTM Layer 1: 128 hidden units, bidirectional=True
  - BiLSTM Layer 2: 128 hidden units, bidirectional=True
  - BiLSTM Layer 3: 128 hidden units, bidirectional=True
  - Dense Layer 1: 128 units with batch normalization
  - Dense Layer 2: 64 units with batch normalization
  - Dense Layer 3: 32 units with batch normalization
  - Dense Layer 4: 1 unit (output)

## Deep Models with GC Content

### 1. Deep CNN+GC
- **Architecture**: Same as Deep CNN
- **GC Integration**: GC content added as single feature in final dense layer

### 2. Deep GRU+GC
- **Architecture**: Same as Deep GRU
- **GC Integration**: GC content concatenated with final GRU output before dense layers

### 3. Deep LSTM+GC
- **Architecture**: Same as Deep LSTM
- **GC Integration**: GC content concatenated with final LSTM output before dense layers

### 4. Deep BiLSTM+GC
- **Architecture**: Same as Deep BiLSTM
- **GC Integration**: GC content concatenated with final BiLSTM output before dense layers

## ChromeCRISPR Hybrid Models

### 1. CNN_GRU+GC (Best Performing Model)
- **Type**: Hybrid CNN-GRU architecture
- **Architecture**:
  - **CNN Branch**:
    - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
    - Embedding Layer: 84 → 128 dimensions
    - Conv1D Layer 1: 128 filters, kernel_size=3, stride=1, padding=1, ReLU activation
    - Conv1D Layer 2: 128 filters, kernel_size=3, stride=1, padding=1, ReLU activation
    - Conv1D Layer 3: 128 filters, kernel_size=3, stride=1, padding=1, ReLU activation
    - Flatten Layer
    - Dense Layer: 128 units with batch normalization
  - **GRU Branch**:
    - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
    - Embedding Layer: 84 → 128 dimensions
    - GRU Layer 1: 128 hidden units, bidirectional=False
    - GRU Layer 2: 128 hidden units, bidirectional=False
    - GRU Layer 3: 128 hidden units, bidirectional=False
    - Dense Layer: 128 units with batch normalization
  - **Fusion Layer**: Concatenate CNN and GRU outputs (256 total features)
  - **GC Integration**: Add GC content feature (257 total features)
  - **Final Layers**:
    - Dense Layer 1: 128 units with batch normalization
    - Dense Layer 2: 64 units with batch normalization
    - Dense Layer 3: 32 units with batch normalization
    - Dense Layer 4: 1 unit (output)
- **Performance**: Spearman = 0.876, MSE = 0.0093

### 2. CNN_LSTM+GC
- **Type**: Hybrid CNN-LSTM architecture
- **Architecture**:
  - **CNN Branch**: Same as CNN_GRU+GC
  - **LSTM Branch**:
    - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
    - Embedding Layer: 84 → 128 dimensions
    - LSTM Layer 1: 128 hidden units, bidirectional=False
    - LSTM Layer 2: 128 hidden units, bidirectional=False
    - LSTM Layer 3: 128 hidden units, bidirectional=False
    - Dense Layer: 128 units with batch normalization
  - **Fusion Layer**: Concatenate CNN and LSTM outputs (256 total features)
  - **GC Integration**: Add GC content feature (257 total features)
  - **Final Layers**: Same as CNN_GRU+GC
- **Performance**: Spearman = 0.867, MSE = 0.0115

### 3. CNN_BiLSTM+GC
- **Type**: Hybrid CNN-BiLSTM architecture
- **Architecture**:
  - **CNN Branch**: Same as CNN_GRU+GC
  - **BiLSTM Branch**:
    - Input: One-hot encoded sgRNA (21×4 matrix, flattened to 84 features)
    - Embedding Layer: 84 → 128 dimensions
    - BiLSTM Layer 1: 128 hidden units, bidirectional=True
    - BiLSTM Layer 2: 128 hidden units, bidirectional=True
    - BiLSTM Layer 3: 128 hidden units, bidirectional=True
    - Dense Layer: 128 units with batch normalization
  - **Fusion Layer**: Concatenate CNN and BiLSTM outputs (256 total features)
  - **GC Integration**: Add GC content feature (257 total features)
  - **Final Layers**: Same as CNN_GRU+GC
- **Performance**: Spearman = 0.870, MSE = 0.0096

## Model Selection Rationale

### Why Hybrid Models?
1. **CNN Strength**: Feature extraction from spatial patterns in sequences
2. **RNN Strength**: Temporal dependency modeling in sequential data
3. **Combination Benefit**: Captures both local motifs and long-range dependencies

### Why This Order (CNN → RNN)?
1. **Feature Extraction First**: CNN extracts local patterns and motifs
2. **Sequence Processing Second**: RNN processes the extracted features sequentially
3. **Biological Relevance**: sgRNA sequences have both local binding motifs and sequential dependencies

### Pipeline Details
1. **Input**: 21-mer sgRNA sequences (one-hot encoded) + GC content
2. **CNN Processing**: Extracts local sequence patterns and motifs
3. **RNN Processing**: Models sequential dependencies and context
4. **Fusion**: Combines both representations
5. **GC Integration**: Adds biological feature at final stage
6. **Output**: Single regression value (activity prediction)

## Training Specifications

### Hyperparameter Tuning
- **Method**: Nested 5-fold cross-validation with Bayesian search
- **Data Split**: 85% training/validation, 15% testing
- **Validation**: 5-fold cross-validation on training set

### Training Environment
- **Hardware**: NVIDIA V100 Volta GPUs with 32GB HBM2 memory
- **System**: Digital Research Alliance of Canada superclusters
- **Memory**: 4GB RAM, 2 CPU cores
- **Training Time**: ~20 seconds per iteration

### Optimization
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam optimizer with learning rate tuning
- **Regularization**: Batch normalization, dropout (where applicable)
- **Early Stopping**: Based on validation loss to prevent overfitting

## Performance Summary

| Model | Spearman Correlation | MSE | Status |
|-------|---------------------|-----|--------|
| CNN_GRU+GC | 0.876 | 0.0093 | **Best Model** |
| CNN_BiLSTM+GC | 0.870 | 0.0096 | Second Best |
| CNN_LSTM+GC | 0.867 | 0.0115 | Third Best |
| Deep CNN+GC | 0.873 | 0.0093 | Baseline |
| Deep GRU+GC | 0.867 | 0.0098 | Baseline |
| Deep LSTM+GC | 0.860 | 0.0104 | Baseline |
| Deep BiLSTM+GC | 0.867 | 0.0098 | Baseline |

## Key Insights

1. **Hybrid Advantage**: CNN-RNN combinations outperform individual architectures
2. **GRU Superiority**: GRU performs better than LSTM in hybrid models
3. **GC Content Impact**: Consistent improvement across all models
4. **Depth Benefits**: Deeper models generally perform better than base models
5. **Bidirectional Trade-off**: BiLSTM shows mixed results in hybrid combinations

This comprehensive architecture documentation provides all necessary details for understanding and reproducing the ChromeCRISPR models without requiring visual diagrams.
