# ChromeCRISPR: Deep Learning Framework for CRISPR Guide RNA Efficiency Prediction

ChromeCRISPR is a comprehensive deep learning framework for predicting CRISPR guide RNA efficiency. The system uses hybrid architectures combining CNN, RNN (GRU/LSTM/BiLSTM), and MLP Mixer with biological features to achieve state-of-the-art performance.

## 🏆 Best Performance
**CNN-GRU+GC Model**: 0.876 Spearman correlation, 0.0093 MSE

## 🎯 Key Features
- **20 Different Model Architectures** evaluated and documented
- **Hybrid CNN-RNN Models** for optimal sequence processing
- **Biological Feature Integration** (GC content, enthalpy, entropy, free energy)
- **State-of-the-art Performance** outperforming DeepHF and AttCRISPR
- **Comprehensive Evaluation** with statistical significance testing

## 🏗️ Model Architecture

### Best Model: CNN-GRU+GC (Trial 62)
- **Architecture**: CNN → GRU → MLP Mixer → Fully Connected Layers
- **Performance**: 0.876 Spearman correlation, 0.0093 MSE
- **Parameters**: 369,729 total parameters
- **File**: `BestModel_Bio_53054713.9_trial_62_epoch_73_Sp_0.8765.pth`

## 📊 Complete Model Performance

### ChromeCRISPR Hybrid Models (Best Performing)
| Model | Spearman | MSE | Description |
|-------|----------|-----|-------------|
| **CNN_GRU+GC** | **0.876** | **0.0093** | **Best model: CNN → GRU with GC** |
| CNN_BiLSTM+GC | 0.870 | 0.0096 | CNN → BiLSTM with GC |
| CNN_LSTM+GC | 0.867 | 0.0115 | CNN → LSTM with GC |

### Deep Models + GC Content
| Model | Spearman | MSE | Description |
|-------|----------|-----|-------------|
| deepCNN+GC | 0.873 | 0.0093 | Deep CNN with GC content |
| deepBiLSTM+GC | 0.867 | 0.0098 | Deep BiLSTM with GC |
| deepGRU+GC | 0.867 | 0.0098 | Deep GRU with GC |
| deepLSTM+GC | 0.860 | 0.0104 | Deep LSTM with GC |

### Deep Models
| Model | Spearman | MSE | Description |
|-------|----------|-----|-------------|
| deepCNN | 0.869 | 0.0098 | Deep CNN architecture |
| deepGRU | 0.868 | 0.0099 | Deep GRU architecture |
| deepLSTM | 0.862 | 0.0103 | Deep LSTM architecture |
| deepBiLSTM | 0.862 | 0.0104 | Deep BiLSTM architecture |

### Base Models + GC Content
| Model | Spearman | MSE | Description |
|-------|----------|-----|-------------|
| LSTM+GC | 0.856 | 0.0112 | LSTM with GC content |
| BiLSTM+GC | 0.855 | 0.0110 | BiLSTM with GC content |
| GRU+GC | 0.840 | 0.0122 | GRU with GC content |
| CNN+GC | 0.781 | 0.0170 | CNN with GC content |

### Base Models
| Model | Spearman | MSE | Description |
|-------|----------|-----|-------------|
| BiLSTM | 0.843 | 0.0120 | Bidirectional LSTM |
| LSTM | 0.837 | 0.0122 | Long Short-Term Memory |
| GRU | 0.837 | 0.0121 | Gated Recurrent Unit |
| CNN | 0.793 | 0.0161 | Convolutional Neural Network |
| RF | 0.755 | 0.0197 | Random Forest (100 estimators) |

## 📁 Available Model Files

The repository contains **11 trained model files** representing different architectures:

- `BestModel_Bio_53054713.9_trial_62_epoch_73_Sp_0.8765.pth` (Best model: 1.42 MB)
- `CNN_LSTM_GC_trial_18.pth` (CNN-LSTM hybrid: 1.71 MB)
- `CNN_BiLSTM_GC_trial_111.pth` (CNN-BiLSTM hybrid: 1.15 MB)
- `deepCNN_GC_trial_112.pth` (Deep CNN: 2.06 MB)
- `deepGRU_GC_trial_68.pth` (Deep GRU: 1.52 MB)
- `deepLSTM_GC_trial_59.pth` (Deep LSTM: 2.50 MB)
- `deepBiLSTM_GC_trial_54.pth` (Deep BiLSTM: 3.58 MB)
- `CNN_GC_trial_39.pth` (CNN with GC: 2.26 MB)
- `GRU_GC_trial_44.pth` (GRU with GC: 2.86 MB)
- `LSTM_GC_trial_103.pth` (LSTM with GC: 1.44 MB)
- `BiLSTM_GC_trial_55.pth` (BiLSTM with GC: 1.63 MB)

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/your-username/ChromeCRISPR.git
cd ChromeCRISPR
pip install -r requirements.txt
```

### Basic Usage
```python
import torch
from models.dynamic_model import ChromeCRISPRModel

# Load the best performing model
model = ChromeCRISPRModel()
model.load_state_dict(torch.load('models/BestModel_Bio_53054713.9_trial_62_epoch_73_Sp_0.8765.pth'))
model.eval()

# Make predictions on sgRNA sequences
# (Implementation details in the model files)
```

## 📋 Data Specifications

### Input
- **Sequence**: 21-mer sgRNA sequences (one-hot encoded)
- **Biological Features**: GC Content, Enthalpy Duplex, Entropy Duplex, Free Energy RNA
- **Format**: PyTorch tensors

### Output
- **Prediction**: Single efficiency score (0-1 range)
- **Format**: Float tensor

### Data Split
- **Training + Validation**: 85%
- **Test**: 15%
- **No test data used for hyperparameter tuning**

## ⚙️ Training Configuration

### Best Model (CNN-GRU+GC) Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.00020972671691680056
- **Batch Size**: 64
- **Epochs**: 84
- **Dropout Rate**: 0.14201131516203347
- **Weight Decay**: 1.882255599576252e-05
- **Gradient Clipping**: 3.848456063450914

### Model Architecture Details
- **CNN Layers**: 2 layers with 128 filters each, kernel size 5
- **RNN**: 2-layer GRU with 128 hidden units
- **MLP Mixer**: Token and channel mixing for enhanced feature processing
- **Fully Connected**: 3 layers (128→64→32→1)

## 📈 Performance Comparison

### vs. State-of-the-Art Models
| Model | Spearman | MSE | Reference |
|-------|----------|-----|-----------|
| **ChromeCRISPR CNN_GRU+GC** | **0.876** | **0.0093** | **This work** |
| AttCRISPR StAC+Bio | 0.872 | Not reported | Xiao et al. 2021 |
| DeepHF RNN+Bio | 0.867 | 0.0094 | Wang et al. 2019 |
| AttCRISPR EnAC+Bio | 0.868 | Not reported | Xiao et al. 2021 |

## 📚 Documentation

- **Complete Model Documentation**: `docs/COMPREHENSIVE_MODEL_DOCUMENTATION.md`
- **Model Architecture Details**: `docs/REAL_MODEL_ARCHITECTURE.md`
- **JSON Architecture Data**: `docs/COMPREHENSIVE_MODEL_DOCUMENTATION.json`

## 🔬 Research Details

### Model Categories Evaluated
1. **Base Models** (5): RF, CNN, GRU, LSTM, BiLSTM
2. **Base Models + GC** (4): CNN+GC, GRU+GC, LSTM+GC, BiLSTM+GC
3. **Deep Models** (4): deepCNN, deepGRU, deepLSTM, deepBiLSTM
4. **Deep Models + GC** (4): deepCNN+GC, deepGRU+GC, deepLSTM+GC, deepBiLSTM+GC
5. **ChromeCRISPR Hybrids** (3): CNN_GRU+GC, CNN_LSTM+GC, CNN_BiLSTM+GC

### Statistical Analysis
- **One-way ANOVA** for group comparisons
- **Tukey's HSD** for post-hoc analysis
- **5-fold cross-validation** for robust evaluation
- **Statistical significance**: p < 0.05 threshold

## 📄 Citation

If you use ChromeCRISPR in your research, please cite:

```bibtex
@article{chromeCRISPR2024,
  title={ChromeCRISPR: Hybrid CNN-RNN Models for CRISPR Guide RNA Efficiency Prediction},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions and support, please open an issue on GitHub or contact the maintainers.
