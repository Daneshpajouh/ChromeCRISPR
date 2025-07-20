# ChromeCRISPR: Complete Model Documentation

## Project Overview
ChromeCRISPR is a deep learning framework for predicting CRISPR guide RNA efficiency using hybrid CNN-RNN architectures. The system combines convolutional neural networks (CNNs) with recurrent neural networks (RNNs) to capture both local sequence patterns and long-range dependencies.

## Best Performing Model: CNN-GRU+GC
- **Performance**: 0.876 Spearman correlation, 0.0093 MSE
- **File**: BestModel_Bio_53054713.9_trial_62_epoch_73_Sp_0.8765.pth
- **Architecture**: CNN → GRU → MLP Mixer → Fully Connected Layers

## Complete Model List (20 Models)

### Base Models (5 models)

#### RF: Random Forest
- **Description**: Ensemble learning method with 100 estimators
- **Performance**: 0.7550 Spearman, 0.0197 MSE

#### CNN: Convolutional Neural Network
- **Description**: 2 convolutional layers, 128 filters each, kernel size 3
- **Performance**: 0.7925 Spearman, 0.0161 MSE

#### GRU: Gated Recurrent Unit
- **Description**: RNN variant for sequential data processing
- **Performance**: 0.8368 Spearman, 0.0121 MSE

#### LSTM: Long Short-Term Memory
- **Description**: RNN variant for capturing long-term dependencies
- **Performance**: 0.8371 Spearman, 0.0122 MSE

#### BiLSTM: Bidirectional LSTM
- **Description**: LSTM processing data in both directions
- **Performance**: 0.8432 Spearman, 0.0120 MSE

### Base Models + GC Content (4 models)

#### CNN+GC: CNN with GC Content
- **Description**: CNN with GC content as additional feature
- **Performance**: 0.7810 Spearman, 0.0170 MSE

#### GRU+GC: GRU with GC Content
- **Description**: GRU with GC content as additional feature
- **Performance**: 0.8401 Spearman, 0.0122 MSE

#### LSTM+GC: LSTM with GC Content
- **Description**: LSTM with GC content as additional feature
- **Performance**: 0.8564 Spearman, 0.0112 MSE

#### BiLSTM+GC: BiLSTM with GC Content
- **Description**: BiLSTM with GC content as additional feature
- **Performance**: 0.8550 Spearman, 0.0110 MSE

### Deep Models (4 models)

#### deepCNN: Deep CNN
- **Description**: CNN with additional layers for enhanced feature extraction
- **Performance**: 0.8694 Spearman, 0.0098 MSE

#### deepGRU: Deep GRU
- **Description**: GRU with additional layers for enhanced sequence processing
- **Performance**: 0.8684 Spearman, 0.0099 MSE

#### deepLSTM: Deep LSTM
- **Description**: LSTM with additional layers for enhanced sequence processing
- **Performance**: 0.8620 Spearman, 0.0103 MSE

#### deepBiLSTM: Deep BiLSTM
- **Description**: BiLSTM with additional layers for enhanced sequence processing
- **Performance**: 0.8617 Spearman, 0.0104 MSE

### Deep Models + GC Content (4 models)

#### deepCNN+GC: Deep CNN with GC Content
- **Description**: Deep CNN with GC content as additional feature
- **Performance**: 0.8728 Spearman, 0.0093 MSE

#### deepGRU+GC: Deep GRU with GC Content
- **Description**: Deep GRU with GC content as additional feature
- **Performance**: 0.8668 Spearman, 0.0098 MSE

#### deepLSTM+GC: Deep LSTM with GC Content
- **Description**: Deep LSTM with GC content as additional feature
- **Performance**: 0.8602 Spearman, 0.0104 MSE

#### deepBiLSTM+GC: Deep BiLSTM with GC Content
- **Description**: Deep BiLSTM with GC content as additional feature
- **Performance**: 0.8671 Spearman, 0.0098 MSE

### ChromeCRISPR Hybrid Models (3 models)

#### CNN_GRU+GC: CNN-GRU Hybrid with GC
- **Description**: Best performing model: CNN followed by GRU with GC content
- **Performance**: 0.8760 Spearman, 0.0093 MSE

#### CNN_LSTM+GC: CNN-LSTM Hybrid with GC
- **Description**: CNN followed by LSTM with GC content
- **Performance**: 0.8668 Spearman, 0.0115 MSE

#### CNN_BiLSTM+GC: CNN-BiLSTM Hybrid with GC
- **Description**: CNN followed by BiLSTM with GC content
- **Performance**: 0.8700 Spearman, 0.0096 MSE


## Available Model Files
The repository contains 11 trained model files:


### deepLSTM_GC_trial_59.pth
- **Total Parameters**: 650,407
- **Layer Types**: RNN, FC, CNN
- **File Size**: 2.50 MB

### deepCNN_GC_trial_112.pth
- **Total Parameters**: 536,772
- **Layer Types**: RNN, FC, MLP_Mixer, CNN
- **File Size**: 2.06 MB

### BiLSTM_GC_trial_55.pth
- **Total Parameters**: 422,104
- **Layer Types**: RNN, FC, MLP_Mixer, CNN
- **File Size**: 1.63 MB

### BestModel_Bio_53054713.9_trial_62_epoch_73_Sp_0.8765.pth
- **Total Parameters**: 369,729
- **Layer Types**: RNN, FC, MLP_Mixer, CNN
- **File Size**: 1.42 MB

### LSTM_GC_trial_103.pth
- **Total Parameters**: 372,484
- **Layer Types**: RNN, FC, MLP_Mixer, CNN
- **File Size**: 1.44 MB

### deepGRU_GC_trial_68.pth
- **Total Parameters**: 393,672
- **Layer Types**: RNN, FC, MLP_Mixer, CNN
- **File Size**: 1.52 MB

### deepBiLSTM_GC_trial_54.pth
- **Total Parameters**: 935,594
- **Layer Types**: RNN, FC, MLP_Mixer, CNN
- **File Size**: 3.58 MB

### CNN_LSTM_GC_trial_18.pth
- **Total Parameters**: 444,263
- **Layer Types**: RNN, FC, MLP_Mixer, CNN
- **File Size**: 1.71 MB

### CNN_GC_trial_39.pth
- **Total Parameters**: 585,507
- **Layer Types**: RNN, FC, CNN
- **File Size**: 2.26 MB

### GRU_GC_trial_44.pth
- **Total Parameters**: 746,378
- **Layer Types**: RNN, FC, MLP_Mixer, CNN
- **File Size**: 2.86 MB

### CNN_BiLSTM_GC_trial_111.pth
- **Total Parameters**: 295,975
- **Layer Types**: RNN, FC, CNN
- **File Size**: 1.15 MB

## Performance Summary

### Top 5 Models by Spearman Correlation:
1. **CNN_GRU+GC**: 0.8760 (Best model)
2. **deepCNN+GC**: 0.8728
3. **deepCNN**: 0.8694
4. **CNN_BiLSTM+GC**: 0.8700
5. **deepBiLSTM+GC**: 0.8671

### Top 5 Models by MSE:
1. **CNN_GRU+GC**: 0.0093 (Best model)
2. **deepCNN+GC**: 0.0093
3. **CNN_BiLSTM+GC**: 0.0096
4. **deepBiLSTM+GC**: 0.0098
5. **deepGRU+GC**: 0.0098

## Data Split
- **Training + Validation**: 85%
- **Test**: 15%
- **No test data used for hyperparameter tuning**

## Input/Output
- **Input**: 21-mer sgRNA sequences (one-hot encoded)
- **Output**: Single value (efficiency score)
- **Biological Features**: GC Content, Enthalpy Duplex, Entropy Duplex, Free Energy RNA

## Training Details
- **Optimizer**: Adam
- **Learning Rate**: Optimized per model (typically ~0.0002)
- **Batch Size**: 64
- **Epochs**: Variable (typically 50-100)
- **Dropout Rate**: Optimized per model (typically ~0.14)

## Usage
```python
import torch
from models.dynamic_model import ChromeCRISPRModel

# Load the best model
model = ChromeCRISPRModel()
model.load_state_dict(torch.load('models/BestModel_Bio_53054713.9_trial_62_epoch_73_Sp_0.8765.pth'))
model.eval()

# Make predictions
# (Implementation details in the model files)
```

## Model Architecture Details
Detailed architecture information for each model file is available in the JSON documentation.
