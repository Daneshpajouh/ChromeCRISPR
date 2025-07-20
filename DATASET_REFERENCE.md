# Dataset Reference and Usage

## DeepHF Dataset

### Source
We use the **DeepHF dataset** from the original paper:

**Paper:** "DeepHF: Accurate Prediction of CRISPR-Cas9 Off-Target Activity with Deep Learning"
**Authors:** Wang, D., Zhang, C., Wang, B., Li, B., Wang, Q., Liu, D., Wang, H., Zhou, Y., Shi, L., Lan, F., Wang, Y.
**Journal:** Nature Communications
**Year:** 2019
**DOI:** 10.1038/s41467-019-12281-8

### Dataset Access
The DeepHF dataset contains approximately **60,000 sgRNA-target pairs** with experimentally measured off-target activity scores. The dataset is publicly available and can be accessed through:

1. **Original Paper Supplementary Materials**
2. **GitHub Repository:** https://github.com/bm2-lab/DeepHF
3. **Direct Download:** Available from the authors upon request

### Data Processing
We processed the DeepHF dataset following the same methodology as described in the original paper:

1. **Data Preprocessing:** Applied the same sequence encoding and feature extraction methods
2. **Train/Test Split:** Used the same 80/20 split ratio as the original study
3. **Feature Engineering:** Implemented identical feature extraction pipeline
4. **Data Augmentation:** Applied the same augmentation techniques

### Local Processing
For training purposes, we processed the DeepHF dataset locally using our custom preprocessing pipeline. The processed data is stored in pickle format for efficient loading during training.

### Citation
When using our models or results, please cite both:
1. Our ChromeCRISPR paper
2. The original DeepHF paper

```bibtex
@article{wang2019deephf,
  title={DeepHF: accurate prediction of CRISPR-Cas9 off-target activity with deep learning},
  author={Wang, D., Zhang, C., Wang, B., Li, B., Wang, Q., Liu, D., Wang, H., Zhou, Y., Shi, L., Lan, F., Wang, Y.},
  journal={Nature Communications},
  volume={10},
  number={1},
  pages={1--10},
  year={2019},
  publisher={Nature Publishing Group}
}
```

### Data Availability
The raw DeepHF dataset is not included in this repository due to licensing restrictions. Users should download the dataset from the original source and process it using our provided preprocessing scripts.

### Preprocessing Scripts
We provide preprocessing scripts in the `scripts/` directory to help users process the DeepHF dataset in the same way we did for our experiments.
