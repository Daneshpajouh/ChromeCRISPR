"""
ChromeCRISPR: Hybrid Machine Learning Model for CRISPR/Cas9 On-Target Activity Prediction

A comprehensive deep learning framework for predicting CRISPR/Cas9 on-target activity
using multiple architectures including CNN, RNN variants, Transformers, and hybrid models.
"""

__version__ = "1.0.0"
__author__ = "Amirhossein Daneshpajouh, Megan Fowler, Kay C. Wiese"
__email__ = "amir_dp@sfu.ca"

from .models.dynamic_model import DynamicModel
from .training.trainer import Trainer
from .evaluation.metrics import EvaluationMetrics
from .data.dataset import CRISPRDataset
from .utils.config import Config

__all__ = [
    "DynamicModel",
    "Trainer",
    "EvaluationMetrics",
    "CRISPRDataset",
    "Config"
]
