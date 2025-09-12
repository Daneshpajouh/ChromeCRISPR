# ChromeCRISPR Training Procedures

This document outlines the complete training procedures used for all ChromeCRISPR models, ensuring reproducibility and consistency.

## Data Preprocessing Pipeline

### 1. Sequence Encoding
- **Method**: One-hot encoding
- **Input**: 21-mer sgRNA sequences (20 nucleotides + PAM)
- **Output**: 84-dimensional feature vector (21 × 4 nucleotides)
- **Framework**: Custom implementation

### 2. Sequence Embedding
- **Method**: Dense embedding layer
- **Input dimension**: 84
- **Output dimension**: 128
- **Activation**: None (linear)

### 3. GC Content Calculation
- **Formula**: GC_content = (Count(G) + Count(C)) / sequence_length
- **Range**: 0-1
- **Integration**: Concatenated with final features before dense layers
- **Biological significance**: Correlates with sgRNA efficiency

### 4. Data Normalization
- **Method**: StandardScaler (optional)
- **Applied to**: All numerical features
- **Purpose**: Stabilize training

## Hyperparameter Optimization

### Bayesian Optimization Setup
- **Framework**: Optuna
- **Optimization target**: Maximize Spearman correlation
- **Trials**: 100 per model
- **Timeout**: 48 hours per model

### Cross-Validation Strategy
- **Method**: 5-fold cross-validation
- **Validation metric**: Spearman correlation on validation fold
- **Data split**: 70% train, 15% validation, 15% test
- **Stratification**: None (regression task)

### Search Spaces

#### Learning Rate
```python
"learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True)
```

#### Batch Size
```python
"batch_size": trial.suggest_categorical("batch_size", [32, 64, 128])
```

#### Dropout Rate
```python
"dropout": trial.suggest_float("dropout", 0.1, 0.5)
```

#### Architecture Parameters
```python
"cnn_filters": trial.suggest_categorical("cnn_filters", [32, 64, 128])
"kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7])
"hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 384, 512])
"num_layers": trial.suggest_categorical("num_layers", [1, 2, 3, 4])
```

## Training Protocol

### 1. Model Initialization
- **Weight initialization**: Kaiming normal for CNN, Xavier for RNN
- **Bias initialization**: Zeros
- **Batch normalization**: Enabled where applicable

### 2. Optimizer Configuration
- **Type**: Adam
- **Learning rate**: Optimized per model
- **Weight decay**: 1e-5 to 1.882e-05
- **Betas**: (0.9, 0.999)
- **AMSGrad**: Disabled

### 3. Loss Function
- **Type**: Mean Squared Error (MSE)
- **Reduction**: Mean
- **Target**: Minimize MSE

### 4. Early Stopping
- **Monitor**: Validation Spearman correlation
- **Mode**: Maximize
- **Patience**: 10 epochs
- **Min delta**: 0.001
- **Restore best weights**: Enabled

### 5. Learning Rate Scheduling
- **Method**: ReduceLROnPlateau
- **Monitor**: Validation loss
- **Factor**: 0.5
- **Patience**: 5 epochs
- **Min LR**: 1e-7

## Hardware Configuration

### Compute Resources
- **GPU**: NVIDIA V100 Volta
- **Memory**: 32GB HBM2
- **CUDA Version**: 11.2+
- **PyTorch Version**: 1.11+

### Memory Management
- **Batch size**: Optimized per model (32-128)
- **Gradient accumulation**: None
- **Mixed precision**: Disabled
- **Memory optimization**: Gradient checkpointing for large models

### Parallel Processing
- **Data loading**: 4 worker processes
- **Pin memory**: Enabled
- **Non-blocking**: Enabled

## Training Loop

### Forward Pass
```python
# Model forward pass
outputs = model(inputs, gc_content)
loss = criterion(outputs.squeeze(), targets)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Metrics Calculation
```python
# Spearman correlation
spearman_corr, _ = spearmanr(outputs.detach().cpu().numpy(),
                            targets.detach().cpu().numpy())

# MSE loss
mse_loss = F.mse_loss(outputs.squeeze(), targets)
```

### Logging
- **Frequency**: Every 10 batches
- **Metrics**: Loss, Spearman correlation, learning rate
- **Hardware**: GPU memory usage, training time
- **Progress**: Epoch progress, ETA

## Model Checkpointing

### Best Model Saving
- **Criterion**: Highest validation Spearman correlation
- **Format**: PyTorch state_dict (.pth)
- **Additional data**: Hyperparameters, training metadata
- **Frequency**: After each epoch

### Checkpoint Strategy
- **Save top-k**: 3 best models
- **Naming convention**: `{model_name}_trial_{trial_num}_epoch_{epoch}_Sp_{spearman:.4f}.pth`
- **Metadata**: JSON file with hyperparameters and performance

## Validation and Testing

### Validation During Training
- **Frequency**: End of each epoch
- **Metrics**: Spearman correlation, MSE
- **Purpose**: Early stopping and model selection

### Final Testing
- **Data**: Held-out test set (15% of total data)
- **Metrics**: Spearman correlation, MSE, R² score
- **Statistical tests**: Paired t-tests for comparisons
- **Confidence intervals**: 95% bootstrap confidence intervals

## Reproducibility

### Random Seeds
- **PyTorch**: `torch.manual_seed(42)`
- **CUDA**: `torch.cuda.manual_seed(42)`
- **NumPy**: `np.random.seed(42)`
- **Python**: `random.seed(42)`

### Environment
- **Python version**: 3.9+
- **Package versions**: Locked in requirements.txt
- **OS**: Linux (Compute Canada clusters)

## Error Handling and Recovery

### Training Interruptions
- **Checkpoint loading**: Automatic recovery from latest checkpoint
- **State restoration**: Optimizer state, epoch number, best metrics
- **Resume capability**: Seamless continuation of interrupted training

### Memory Issues
- **Gradient clipping**: Max norm = 1.0
- **Batch size reduction**: Automatic fallback for OOM errors
- **Model pruning**: Remove unnecessary layers if needed

## Performance Monitoring

### Real-time Metrics
- **Training loss**: Exponential moving average
- **Validation metrics**: Per-epoch tracking
- **Learning rate**: Current and scheduled values
- **GPU utilization**: Memory and compute usage

### Long-term Tracking
- **Experiment logging**: MLflow integration
- **Hyperparameter sweeps**: Optuna dashboard
- **Model comparisons**: Automated performance reports

## Quality Assurance

### Data Integrity Checks
- **Input validation**: Sequence length, nucleotide composition
- **Output range**: Activity scores between 0-1
- **NaN detection**: Automatic removal of invalid samples

### Model Validation
- **Gradient flow**: Check for vanishing/exploding gradients
- **Weight distributions**: Monitor for unhealthy patterns
- **Overfitting detection**: Train/validation performance gap

## Deployment Preparation

### Model Export
- **Format**: PyTorch state_dict
- **Metadata**: Hyperparameters, performance metrics, training details
- **Documentation**: Model card with usage instructions

### Inference Optimization
- **TorchScript**: Optional JIT compilation
- **ONNX export**: Cross-platform compatibility
- **Quantization**: Reduced precision for deployment
