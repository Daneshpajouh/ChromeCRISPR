"""
GeneX Phase 1 - Enhanced API Clients Module

This module provides comprehensive API clients for scientific literature mining
with advanced features including async support, bulk operations, caching,
rate limiting, and detailed analytics.
"""

from .base_client import BaseAPIClient
from .pubmed_client import PubMedClient
from .semantic_scholar_client import SemanticScholarClient
from .crossref_client import CrossRefClient
from .ncbi_client import NCBIClient
from .ensembl_client import EnsemblClient

__all__ = [
    'BaseAPIClient',
    'PubMedClient',
    'SemanticScholarClient',
    'CrossRefClient',
    'NCBIClient',
    'EnsemblClient'
]

# Version information
__version__ = "2.0.0"
__author__ = "GeneX Research Team"
__description__ = "Enhanced API clients for scientific literature mining"
