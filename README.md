# ChromeCRISPR: High Efficacy Hybrid Machine Learning Models for CRISPR/Cas On-Target Predictions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17058362.svg)](https://doi.org/10.5281/zenodo.17058362)

## Quick Navigation

| Section | Description |
|---------|-------------|
| **[Model Architectures](#model-architectures)** | Complete collection of 19 models |
| **[Performance Summary](#performance-summary)** | Benchmark results and comparisons |
| **[Documentation](#documentation)** | Hyperparameters and training procedures |
| **[Installation](#installation)** | Setup and requirements |
| **[Usage](#usage)** | Model training and evaluation |

## Overview

ChromeCRISPR is a collection of novel hybrid machine learning models that combine Convolutional Neural Networks (CNNs) with Recurrent Neural Networks (RNNs) to achieve high efficacy for CRISPR/Cas on-target predictions. Our best model, CNN-GRU hybrid with GC content, establishes new benchmarks for predictive accuracy in CRISPR/Cas9 efficacy predictions.

## Key Features

- **Hybrid Architecture**: Combines CNN feature extraction with RNN sequence processing
- **GC Content Integration**: Incorporates biological features for improved predictions
- **State-of-the-Art Performance**: Outperforms DeepHF and AttCRISPR models
- **Comprehensive Evaluation**: Multiple model architectures and configurations tested
- **Publication Quality**: Research paper with detailed methodology and results

## Performance Summary

### Best Model Performance

Our best model, **CNN_GRU+GC**, achieves:
- **Spearman Correlation**: 0.876
- **Mean Squared Error**: 0.0093

### Benchmark Comparison

| Model | Spearman | MSE | Status |
|-------|----------|-----|--------|
| **CNN_GRU+GC** | **0.876** | **0.0093** | **New Benchmark** |
| deepCNN+GC | 0.873 | 0.0093 | Second Best |
| CNN_BiLSTM+GC | 0.870 | 0.0096 | Third Best |
| deepBiLSTM+GC | 0.867 | 0.0098 | Baseline |
| deepGRU+GC | 0.867 | 0.0098 | Baseline |
| **DeepHF** | 0.867 | 0.0094 | Previous SOTA |
| **AttCRISPR** | 0.872 | - | Previous SOTA |

*For complete performance comparison of all 19 models, see [Hyperparameter Documentation](docs/hyperparameters/)*

### Key Performance Insights

1. **Hybrid Advantage**: CNN-RNN combinations significantly outperform individual architectures
2. **GRU Superiority**: GRU performs better than LSTM in hybrid models
3. **GC Content Impact**: Consistent improvement across all model types
4. **Depth Benefits**: Deeper models generally perform better than base models
5. **New Benchmark**: CNN_GRU+GC establishes new state-of-the-art performance

## Model Architectures

### Complete Model Collection (19 Models)

All models are organized in the `models/` directory with proper categorization.

#### Base Models (4 models)
- **CNN**: Convolutional neural network with 2 conv layers (128 filters each)
- **GRU**: Gated recurrent unit with 2 layers (128 hidden units each)
- **LSTM**: Long short-term memory with 2 layers (128 hidden units each)
- **BiLSTM**: Bidirectional LSTM with 2 layers (128 hidden units each)

#### Base Models + GC Content (4 models)
- **CNN+GC**: CNN with GC content integration
- **GRU+GC**: GRU with GC content integration
- **LSTM+GC**: LSTM with GC content integration
- **BiLSTM+GC**: BiLSTM with GC content integration

#### Deep Models (4 models)
Enhanced versions with additional layers for deeper feature extraction:
- **deepCNN**: 4 convolutional layers + 4 dense layers
- **deepGRU**: 4 GRU layers + 4 dense layers
- **deepLSTM**: 4 LSTM layers + 4 dense layers
- **deepBiLSTM**: 4 BiLSTM layers + 4 dense layers

#### Deep Models + GC Content (4 models)
- **deepCNN+GC**: Deep CNN with GC content integration
- **deepGRU+GC**: Deep GRU with GC content integration
- **deepLSTM+GC**: Deep LSTM with GC content integration
- **deepBiLSTM+GC**: Deep BiLSTM with GC content integration

#### ChromeCRISPR Hybrid Models (3 models)
Novel hybrid architectures combining CNN and RNN components:
- **CNN_GRU+GC**: CNN + GRU fusion with GC content (**Best performing: 0.876 Spearman**)
- **CNN_LSTM+GC**: CNN + LSTM fusion with GC content
- **CNN_BiLSTM+GC**: CNN + BiLSTM fusion with GC content

### Directory Structure

```
models/
├── base_models/                    # 4 base models
├── base_models_with_gc/           # 4 base models + GC
├── deep_models/                   # 4 deep models
├── deep_models_with_gc/           # 4 deep models + GC
└── chromecrispr_hybrid_models/    # 3 hybrid models
```

**Note**: Random Forest model not included (not found in available files)

## Dataset

We use the DeepHF dataset containing:
- ~60,000 unique sgRNAs from 20,000 human genes
- 21-mer sequences (20 nucleotides + PAM)
- Activity values as indel frequencies (0-1 range)
- GC content as biological feature

## Installation

```bash
# Clone the repository
git clone https://github.com/Daneshpajouh/ChromeCRISPR.git
cd ChromeCRISPR

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Model Training

```python
from src.models import ChromeCRISPRModel
from src.data import DataLoader

# Load data
data_loader = DataLoader()
X_train, y_train, X_test, y_test = data_loader.load_deephf_data()

# Train model
model = ChromeCRISPRModel(architecture='cnn_gru_gc')
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

### Model Comparison

```python
from src.evaluation import ModelEvaluator

# Compare all models
evaluator = ModelEvaluator()
results = evaluator.compare_models(X_test, y_test)
evaluator.plot_results(results)
```

## Documentation

### Complete Technical Documentation

| Document | Description |
|----------|-------------|
| **[Hyperparameter Documentation](docs/hyperparameters/)** | Complete hyperparameter specifications for all 19 models |
| **[Training Procedures](docs/training_procedures/)** | Detailed training protocols and methodologies |
| **[Model Architectures](docs/MODEL_ARCHITECTURES.md)** | Comprehensive architecture descriptions |
| **[Performance Analysis](docs/COMPREHENSIVE_MODEL_DOCUMENTATION.md)** | Detailed performance analysis and comparisons |

### Key Documentation Files

#### Hyperparameter Documentation (`docs/hyperparameters/`)
- **19 JSON files** with complete hyperparameter specifications
- **Bayesian optimization results** and search spaces
- **Performance metrics** for each hyperparameter configuration
- **Training details** including hardware and timing
- **Comprehensive README** with performance summary table

#### Training Procedures (`docs/training_procedures/`)
- **Data preprocessing pipeline** with exact steps
- **Hyperparameter optimization protocol** using Optuna
- **Cross-validation strategy** and evaluation metrics
- **Hardware configuration** and resource requirements
- **Model checkpointing** and reproducibility procedures

#### Model Architectures (`docs/MODEL_ARCHITECTURES.md`)
- **Detailed layer-by-layer specifications** for all models
- **Parameter counts** and computational complexity
- **Input/output dimensions** and data flow
- **GC content integration methods** and biological relevance

### Quick Access to Best Model

```bash
# Best performing model: CNN_GRU+GC
# Location: models/chromecrispr_hybrid_models/CNN_GRU+GC.pth
# Performance: 0.876 Spearman, 0.0093 MSE
# Hyperparameters: docs/hyperparameters/CNN_GRU+GC_hyperparameters.json
```

## Project Structure

```
ChromeCRISPR/
├── models/                          # All 19 trained models
│   ├── base_models/                 # 4 base models
│   ├── base_models_with_gc/         # 4 base models + GC content
│   ├── deep_models/                 # 4 deep models
│   ├── deep_models_with_gc/         # 4 deep models + GC content
│   └── chromecrispr_hybrid_models/  # 3 hybrid models
├── docs/                            # Comprehensive documentation
│   ├── hyperparameters/             # 19 hyperparameter JSON files
│   ├── training_procedures/         # Training protocols
│   ├── MODEL_ARCHITECTURES.md       # Architecture details
│   └── COMPREHENSIVE_MODEL_DOCUMENTATION.md
├── src/                             # Source code
│   ├── models/                      # Model implementations
│   ├── evaluation/                  # Performance evaluation
│   └── training/                    # Training utilities
├── requirements.txt                 # Python dependencies
├── setup.py                         # Installation script
├── LICENSE                          # MIT License
├── AUTHORS                          # Author information
├── DATASET_REFERENCE.md             # Dataset citations
└── README.md                        # This file
```

## Model Architecture Details

For comprehensive details about all model architectures, see [docs/MODEL_ARCHITECTURES.md](docs/MODEL_ARCHITECTURES.md).

### Key Architecture Features

1. **Input Processing**: One-hot encoding of 21-mer sequences (84 features)
2. **Sequence Embedding**: 84 → 128 dimensions
3. **CNN Branch**: 3 conv layers with 128 filters each
4. **RNN Branch**: 3 recurrent layers (GRU/LSTM/BiLSTM) with 128 units each
5. **Fusion**: Concatenation of CNN and RNN outputs (256 features)
6. **GC Integration**: Addition of GC content feature (257 total features)
7. **Final Layers**: 3 dense layers (128, 64, 32 units) + output layer

## Training Specifications

- **Hyperparameter Tuning**: Nested 5-fold cross-validation with Bayesian search
- **Data Split**: 85% training/validation, 15% testing
- **Hardware**: NVIDIA V100 Volta GPUs with 32GB HBM2 memory
- **Training Time**: ~20 seconds per iteration
- **Optimization**: Adam optimizer with MSE loss function


## Citation

If you use ChromeCRISPR in your research, please cite our paper:

```bibtex
@article{daneshpajouh2024chromecrispr,
  title={ChromeCRISPR: A High Efficacy Hybrid Machine Learning Model for CRISPR/Cas On-Target Predictions},
  author={Daneshpajouh, Amirhossein and Fowler, Megan and Wiese, Kay C.},
  journal={BioMed Central},
  year={2024}
}
```

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Digital Research Alliance of Canada for computational resources
- Natural Sciences and Engineering Research Council of Canada (NSERC) for funding
- Simon Fraser University for research support

## Contact

For questions or support, please contact:
- Amirhossein Daneshpajouh: amir_dp@sfu.ca
- Megan Fowler: megan_fowler_2@sfu.ca
- Kay C. Wiese: wiese@sfu.ca

## Related Publications

- [DeepHF: Optimized CRISPR Guide RNA Design via Deep Learning](https://doi.org/10.1038/s41467-019-12281-8)
- [AttCRISPR: Attention-based deep learning for CRISPR/Cas9 guide RNA design](https://doi.org/10.1093/bioinformatics/btab127)

---

**Note**: This repository contains the complete implementation and documentation for ChromeCRISPR. All model architectures are described in detail in the documentation, and the code is fully functional for reproducing our results.
