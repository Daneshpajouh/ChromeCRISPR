# ChromeCRISPR: High Efficacy Hybrid Machine Learning Models for CRISPR/Cas On-Target Predictions

## Overview

ChromeCRISPR is a collection of novel hybrid machine learning models that combine Convolutional Neural Networks (CNNs) with Recurrent Neural Networks (RNNs) to achieve high efficacy for CRISPR/Cas on-target predictions. Our best model, CNN-GRU hybrid with GC content, establishes new benchmarks for predictive accuracy in CRISPR/Cas9 efficacy predictions.

## Key Features

- **Hybrid Architecture**: Combines CNN feature extraction with RNN sequence processing
- **GC Content Integration**: Incorporates biological features for improved predictions
- **State-of-the-Art Performance**: Outperforms DeepHF and AttCRISPR models
- **Comprehensive Evaluation**: Multiple model architectures and configurations tested
- **Publication Quality**: Research paper with detailed methodology and results

## Performance

Our best model, **CNN_GRU+GC**, achieves:
- **Spearman Correlation**: 0.876
- **Mean Squared Error**: 0.0093

This outperforms previous state-of-the-art models:
- DeepHF RNN + Bio: Spearman = 0.867, MSE = 0.0094
- AttCRISPR StAC + Bio: Spearman = 0.872

## Model Architectures

### Base Models
- **Random Forest (RF)**: Ensemble learning with 100 estimators
- **CNN**: Convolutional neural network with 2 conv layers (128 filters each)
- **GRU**: Gated recurrent unit with 2 layers (128 hidden units each)
- **LSTM**: Long short-term memory with 2 layers (128 hidden units each)
- **BiLSTM**: Bidirectional LSTM with 2 layers (128 hidden units each)

### Deep Models
Enhanced versions of base models with 3 specialized layers and 3 dense layers:
- **Deep CNN**: 3 conv layers + 3 dense layers (128, 64, 32 units)
- **Deep GRU**: 3 GRU layers + 3 dense layers
- **Deep LSTM**: 3 LSTM layers + 3 dense layers
- **Deep BiLSTM**: 3 BiLSTM layers + 3 dense layers

### ChromeCRISPR Hybrid Models
Our novel hybrid architectures combining CNN and RNN components:
- **CNN_GRU+GC**: CNN + GRU fusion with GC content (Best performing)
- **CNN_LSTM+GC**: CNN + LSTM fusion with GC content
- **CNN_BiLSTM+GC**: CNN + BiLSTM fusion with GC content

## Dataset

We use the DeepHF dataset containing:
- **~60,000 unique sgRNAs** from 20,000 human genes
- **21-mer sequences** (20 nucleotides + PAM)
- **Activity values** as indel frequencies (0-1 range)
- **GC content** as biological feature

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

## Project Structure

```
ChromeCRISPR/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data loading and preprocessing
│   ├── evaluation/        # Evaluation metrics and plotting
│   └── utils/             # Utility functions
├── docs/                  # Documentation
│   └── MODEL_ARCHITECTURES.md  # Detailed architecture descriptions
├── config/                # Configuration files
├── results/               # Model results and figures
├── requirements.txt       # Python dependencies
└── README.md             # This file
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

## Results Summary

| Model | Spearman | MSE | Status |
|-------|----------|-----|--------|
| CNN_GRU+GC | 0.876 | 0.0093 | **Best** |
| CNN_BiLSTM+GC | 0.870 | 0.0096 | Second |
| CNN_LSTM+GC | 0.867 | 0.0115 | Third |
| Deep CNN+GC | 0.873 | 0.0093 | Baseline |
| Deep GRU+GC | 0.867 | 0.0098 | Baseline |

## Key Insights

1. **Hybrid Advantage**: CNN-RNN combinations outperform individual architectures
2. **GRU Superiority**: GRU performs better than LSTM in hybrid models
3. **GC Content Impact**: Consistent improvement across all models
4. **Depth Benefits**: Deeper models generally perform better than base models
5. **Bidirectional Trade-off**: BiLSTM shows mixed results in hybrid combinations

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
