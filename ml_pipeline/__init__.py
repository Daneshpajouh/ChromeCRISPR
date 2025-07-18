"""
GeneX ML/DL/AI Pipeline Module

This module provides comprehensive machine learning, deep learning, and AI capabilities
for the GeneX Phase 1 research project. It replaces regex-based approaches with
sophisticated ML techniques for data extraction, analysis, and knowledge discovery.

Key Components:
- NLP/Transformer-based text analysis
- Deep learning models for classification
- AI-powered knowledge extraction
- Cluster-optimized processing
- Automated feature engineering
"""

# from .ai_analyzer import AIAnalyzer
# Remove OpenAI dependency for open-source pipeline

from .deep_learning_models import DeepLearningModels
from .nlp_processor import NLPProcessor
# from .knowledge_extractor import KnowledgeExtractor  # Removed, does not exist
from .cluster_executor import ClusterExecutor
# from .ml_pipeline_manager import MLPipelineManager  # Removed to avoid openai dependency

__all__ = [
    'DeepLearningModels',
    'NLPProcessor',
    'KnowledgeExtractor',
    'ClusterExecutor',
    'MLPipelineManager'
]
