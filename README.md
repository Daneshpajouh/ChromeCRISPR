# ChromeCRISPR: Hybrid Machine Learning Model for CRISPR/Cas9 On-Target Activity Prediction

## Authors

**Amirhossein Daneshpajouh†**, **Megan Fowler†**, and **Kay C. Wiese***

*Correspondence: wiese@sfu.ca

School of Computing Science, Simon Fraser University, 8888 University Dr W, Burnaby, Canada

†Equal contributor

## Overview

ChromeCRISPR is a comprehensive hybrid machine learning framework for predicting CRISPR/Cas9 on-target activity. The system combines multiple deep learning architectures including CNN, RNN variants (LSTM, GRU, BiLSTM), Transformers, and hybrid models to achieve state-of-the-art performance in CRISPR activity prediction.

## Key Features

- **Multiple Model Architectures**: CNN, LSTM, GRU, BiLSTM, Transformer, and hybrid combinations
- **GC Content Integration**: Specialized models incorporating GC content features
- **Biological Feature Enhancement**: Models with additional biological features
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Comprehensive Evaluation**: Multiple performance metrics and cross-validation
- **HPC-Ready**: Optimized for high-performance computing clusters

## Performance Summary

| Model | Spearman Correlation | MSE | Architecture |
|-------|---------------------|-----|--------------|
| CNN-GRU+GC | 0.8768 | 0.0095 | Hybrid CNN-RNN with GC |
| CNN-BiLSTM+GC | 0.8756 | 0.0098 | Hybrid CNN-RNN with GC |
| Deep CNN+GC | 0.8732 | 0.0101 | Deep CNN with GC |
| CNN-LSTM+GC | 0.8701 | 0.0105 | Hybrid CNN-RNN with GC |

## Repository Structure

```
ChromeCRISPR/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data processing
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Utility functions
├── data/                  # Datasets
├── models/                # Trained models
├── results/               # Results and outputs
├── config/                # Configuration files
├── docs/                  # Documentation
```

## Installation

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- CUDA (for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/ChromeCRISPR.git
cd ChromeCRISPR
```

2. Create and activate virtual environment:
```bash
python -m venv chromecrispr_env
source chromecrispr_env/bin/activate  # On Windows: chromecrispr_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```python
from src.models.dynamic_model import DynamicModel
from src.training.trainer import Trainer

# Initialize model
model = DynamicModel(
    model_type='cnn_gru',
    input_size=21,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    use_gc=True
)

# Train model
trainer = Trainer(model, train_loader, val_loader)
trainer.train(epochs=100, lr=0.001)
```

### Making Predictions

```python
from src.models.dynamic_model import DynamicModel

# Load trained model
model = DynamicModel.load_from_checkpoint('models/best_model.pth')

# Make predictions
predictions = model.predict(test_sequences)
```

## Model Architectures

The repository includes comprehensive documentation for all model architectures:

- [Core Models](model_architectures/README.md#core-models)
- [Hybrid Models](model_architectures/README.md#hybrid-models)
- [Advanced Models](model_architectures/README.md#advanced-models)
- [Feature-Enhanced Models](model_architectures/README.md#feature-enhanced-models)

## Data Format

### Input Format

The model expects CRISPR guide RNA sequences in the following format:

```python
# Sequence data (21 nucleotides - 20bp guide + 1bp variable PAM N)
sequence = "GTCGCCCCGCCCCGCCCCGCC"

# GC content (optional)
gc_content = 0.87

# Additional features (optional)
features = {
    'position': 123,
    'chromosome': 'chr1',
    'strand': '+'
}
```

### Output Format

The model outputs activity scores between 0 and 1:

```python
activity_score = 0.847  # Higher values indicate higher predicted activity
```

## Training Configuration

### Hyperparameter Tuning

The system uses Optuna for automated hyperparameter optimization:

```python
from src.training.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(
    model_type='cnn_gru',
    n_trials=100,
    timeout=3600
)

best_params = tuner.optimize()
```

### Training Parameters

Key training parameters can be configured in `config/training_config.yaml`:

```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10

model:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  use_gc: true
```

## Evaluation Metrics

The system evaluates models using multiple metrics:

- **Spearman Correlation**: Measures rank correlation between predicted and actual activities
- **Pearson Correlation**: Measures linear correlation
- **Mean Squared Error (MSE)**: Measures prediction accuracy
- **R² Score**: Measures explained variance
- **Mean Absolute Error (MAE)**: Measures average absolute prediction error

## HPC Deployment

### Cluster Configuration

The system is optimized for HPC clusters with SLURM job scheduling:

```bash
# Submit training job
sbatch scripts/train_model.sh

# Submit hyperparameter tuning job
sbatch scripts/hyperparameter_tuning.sh
```

### Multi-GPU Training

For multi-GPU training on HPC clusters:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model)
```

## Results and Analysis

### Performance Comparison

Detailed performance comparisons for all models are available in:
- [Model Performance Summary](model_architectures/MANUSCRIPT_MODELS_SUMMARY.md)
- [Complete Results Analysis](results/performance_analysis.md)

### Model Interpretability

The system includes tools for model interpretability:

```python
from src.evaluation.interpretability import ModelInterpreter

interpreter = ModelInterpreter(model)
attention_weights = interpreter.get_attention_weights(sequence)
feature_importance = interpreter.get_feature_importance()
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use ChromeCRISPR in your research, please cite:

```bibtex
@article{chromecrispr2024,
  title={ChromeCRISPR: A Hybrid Machine Learning Framework for CRISPR/Cas9 On-Target Activity Prediction},
  author={Daneshpajouh, Amirhossein and Fowler, Megan and Wiese, Kay C.},
  journal={Journal Name},
  year={2024},
  doi={10.xxxx/xxxxx}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and support:
- **Primary Contact**: Kay C. Wiese (wiese@sfu.ca)
- **Technical Support**: Amirhossein Daneshpajouh (amir_dp@sfu.ca)
- **Research Inquiries**: Megan Fowler (mfa69@sfu.ca)
- GitHub Issues: [Create an issue](https://github.com/Daneshpajouh/ChromeCRISPR/issues)

## Acknowledgments

- Compute Canada for HPC resources
- PyTorch team for the deep learning framework
- The CRISPR research community for datasets and insights
