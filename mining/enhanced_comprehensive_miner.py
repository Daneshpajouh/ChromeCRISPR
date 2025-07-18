"""
Enhanced Comprehensive Gene Editing Data Miner
Implements multi-metric outcome framework and comprehensive feature engineering
Based on the technical blueprint for large-scale gene editing database construction
"""

import asyncio
import json
import logging
import time
import re
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import requests
import aiohttp
from dataclasses import dataclass

from ..api_clients.pubmed_client import PubMedClient
from ..api_clients.semantic_scholar_client import SemanticScholarClient
from ..api_clients.crossref_client import CrossRefClient
from ..api_clients.ncbi_client import NCBIClient
from ..api_clients.ensembl_client import EnsemblClient
from ..utils.config import Config
from ..utils.logger import setup_logger
from .enhanced_data_structures import (
    MultiMetricOutcomes, ComprehensiveFeatures, EnhancedDataPoint,
    MVDPCriteria, DataAcquisitionConfig, FeatureExtractionConfig
)

logger = setup_logger(__name__)

@dataclass
class DataPoint:
    """Standardized data point structure"""
    domain: str
    source: str
    paper_id: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    efficiency_score: Optional[float] = None
    experimental_conditions: Optional[Dict[str, Any]] = None
    extracted_at: Optional[str] = None

class EnhancedComprehensiveMiner:
    """
    Enhanced comprehensive data miner for gene editing domains.

    Implements async-first architecture with proper error handling,
    rate limiting, and comprehensive data extraction.
    """

    def __init__(self, config: Config):
        self.config = config
        self.session_id = f"enhanced_comprehensive_{int(time.time())}"

        # Initialize statistics tracking first
        self.stats = {
            'total_data_points': 0,
            'domains_processed': 0,
            'errors': [],
            'start_time': time.time()
        }

        # Initialize API clients with shared session
        self.clients = {}
        self.session = None
        self._initialize_clients()

        # Initialize domain configurations
        self.domain_configs = {
            'crispr': {
                'data_sources': ['literature', 'encode', 'geo'],
                'search_terms': [
                    'CRISPR-Cas9 efficiency indel frequency',
                    'CRISPR-Cas9 on-target off-target ratio',
                    'CRISPR-Cas9 repair profile spectrum',
                    'CRISPR-Cas9 cell survival rate',
                    'CRISPR-Cas12 efficiency',
                    'CRISPR-Cas13 efficiency'
                ],
                'encode_search_terms': ['CRISPR', 'Cas9', 'gene editing'],
                'geo_search_terms': ['CRISPR[Title/Abstract] AND "gene editing"[Title/Abstract]']
            },
            'prime_editing': {
                'data_sources': ['literature', 'encode', 'geo'],
                'search_terms': [
                    'Prime editing efficiency',
                    'pegRNA design optimization',
                    'Prime editing precision',
                    'Prime editing product purity',
                    'Prime editing target site accessibility'
                ],
                'encode_search_terms': ['Prime editing', 'pegRNA'],
                'geo_search_terms': ['Prime editing[Title/Abstract]']
            },
            'base_editing': {
                'data_sources': ['literature', 'encode', 'geo'],
                'search_terms': [
                    'Base editing efficiency',
                    'Cytosine base editor efficiency',
                    'Adenine base editor efficiency',
                    'Base editing bystander effects',
                    'Base editing window precision'
                ],
                'encode_search_terms': ['Base editing', 'CBE', 'ABE'],
                'geo_search_terms': ['Base editing[Title/Abstract]']
            }
        }

        logger.info(f"Enhanced Comprehensive Miner initialized with session ID: {self.session_id}")

    async def __aenter__(self):
        """Async context manager entry"""
        # Create shared aiohttp session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)

        # Reinitialize clients with session
        self._initialize_clients()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            logger.info("Closed aiohttp session")

    def _initialize_clients(self):
        """Initialize API clients with proper configuration"""
        try:
            # Use Config methods to get API configurations
            self.clients['pubmed'] = PubMedClient(self.config.get_api_config('pubmed'), self.session)
            self.clients['semantic_scholar'] = SemanticScholarClient(self.config.get_api_config('semantic_scholar'), self.session)
            self.clients['crossref'] = CrossRefClient(self.config.get_api_config('crossref'), self.session)
            self.clients['ncbi'] = NCBIClient(self.config.get_api_config('ncbi_gene'), self.session)
            self.clients['ensembl'] = EnsemblClient(self.config.get_api_config('ensembl'), self.session)

            logger.info(f"Initialized {len(self.clients)} API clients")

        except Exception as e:
            logger.error(f"Error initializing API clients: {e}")
            if hasattr(self, 'stats'):
                self.stats['errors'].append(f"Client initialization error: {e}")

    async def _mine_literature_for_experiments(self, domain: str, config: Dict) -> List[Dict]:
        """
        Mine literature for experimental data and metadata using async methods
        """
        experiments = []

        try:
            search_terms = config['search_terms']

            for term in search_terms:
                # Use PubMed client async method
                if 'pubmed' in self.clients:
                    try:
                        papers = await self.clients['pubmed'].search_papers_async(term, max_results=100)
                        for paper in papers:
                            experiments.append({
                                'source': 'PubMed',
                                'paper': paper,
                                'search_term': term,
                                'domain': domain
                            })
                    except Exception as e:
                        logger.error(f"PubMed mining failed for term '{term}': {e}")
                        self.stats['errors'].append(f"PubMed error for {term}: {e}")

                # Use Semantic Scholar client async method
                if 'semantic_scholar' in self.clients:
                    try:
                        papers = await self.clients['semantic_scholar'].search_papers_async(term, limit=100)
                        for paper in papers:
                            experiments.append({
                                'source': 'Semantic Scholar',
                                'paper': paper,
                                'search_term': term,
                                'domain': domain
                            })
                    except Exception as e:
                        logger.error(f"Semantic Scholar mining failed for term '{term}': {e}")
                        self.stats['errors'].append(f"Semantic Scholar error for {term}: {e}")

            logger.info(f"Mined {len(experiments)} literature experiments for domain: {domain}")

        except Exception as e:
            logger.error(f"Literature mining failed for domain {domain}: {e}")
            self.stats['errors'].append(f"Literature mining error for {domain}: {e}")

        return experiments

    async def _acquire_encode_data(self, domain: str, config: Dict) -> List[Dict]:
        """
        Acquire ENCODE data using aiohttp for async HTTP requests
        """
        experiments = []

        try:
            import aiohttp
            import asyncio

            # ENCODE API endpoints - use the proper API endpoint
            encode_base_url = "https://www.encodeproject.org"
            search_endpoint = "/api/v1/search/"

            # Search terms for gene editing
            search_terms = config.get('encode_search_terms', [
                'CRISPR', 'gene editing', 'genome editing',
                'Prime editing', 'base editing', 'Cas9', 'Cas12'
            ])

            # Headers for JSON API requests
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'User-Agent': 'GeneX-Miner/1.0'
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                for term in search_terms:
                    try:
                        # Build search URL with proper API endpoint
                        search_url = f"{encode_base_url}{search_endpoint}"
                        params = {
                            'type': 'Experiment',
                            'searchTerm': term,
                            'limit': 50,
                            'format': 'json'
                        }

                        async with session.get(search_url, params=params, headers=headers) as response:
                            if response.status == 200:
                                try:
                                    data = await response.json()
                                    if '@graph' in data:
                                        for experiment in data['@graph']:
                                            experiments.append({
                                                'source': 'ENCODE',
                                                'experiment': experiment,
                                                'search_term': term,
                                                'domain': domain
                                            })
                                    elif 'results' in data:
                                        for experiment in data['results']:
                                            experiments.append({
                                                'source': 'ENCODE',
                                                'experiment': experiment,
                                                'search_term': term,
                                                'domain': domain
                                            })
                                except aiohttp.ContentTypeError:
                                    # If JSON parsing fails, try alternative endpoint
                                    logger.warning(f"ENCODE returned non-JSON for term '{term}', trying alternative endpoint")
                                    continue
                            elif response.status == 404:
                                logger.warning(f"ENCODE API returned status 404 for term '{term}'")
                            else:
                                logger.warning(f"ENCODE API returned status {response.status} for term '{term}'")

                    except Exception as e:
                        logger.error(f"ENCODE data acquisition failed for term '{term}': {e}")
                        self.stats['errors'].append(f"ENCODE error for {term}: {e}")

            logger.info(f"Acquired {len(experiments)} ENCODE experiments for domain: {domain}")

        except Exception as e:
            logger.error(f"ENCODE data acquisition failed for domain {domain}: {e}")
            self.stats['errors'].append(f"ENCODE acquisition error for {domain}: {e}")

        return experiments

    async def _acquire_geo_data(self, domain: str, config: Dict) -> List[Dict]:
        """
        Acquire GEO data using aiohttp for async HTTP requests
        """
        experiments = []

        try:
            import aiohttp
            import asyncio

            # GEO API endpoints
            geo_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

            # Search terms for gene editing
            search_terms = config.get('geo_search_terms', [
                'CRISPR[Title/Abstract] AND "gene editing"[Title/Abstract]',
                'Prime editing[Title/Abstract]',
                'Base editing[Title/Abstract]',
                'Cas9[Title/Abstract] AND "genome editing"[Title/Abstract]'
            ])

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                for term in search_terms:
                    try:
                        # Search for GEO datasets
                        search_url = f"{geo_base_url}/esearch.fcgi"
                        params = {
                            'db': 'gds',
                            'term': term,
                            'retmode': 'json',
                            'retmax': 50
                        }

                        async with session.get(search_url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                                    for gse_id in data['esearchresult']['idlist']:
                                        experiments.append({
                                            'source': 'GEO',
                                            'gse_id': gse_id,
                                            'search_term': term,
                                            'domain': domain
                                        })
                            else:
                                logger.warning(f"GEO API returned status {response.status} for term '{term}'")

                    except Exception as e:
                        logger.error(f"GEO data acquisition failed for term '{term}': {e}")
                        self.stats['errors'].append(f"GEO error for {term}: {e}")

            logger.info(f"Acquired {len(experiments)} GEO experiments for domain: {domain}")

        except Exception as e:
            logger.error(f"GEO data acquisition failed for domain {domain}: {e}")
            self.stats['errors'].append(f"GEO acquisition error for {domain}: {e}")

        return experiments

    async def _mine_domain_data(self, domain: str, config: Dict) -> Dict:
        """
        Mine data for a specific domain using async methods
        """
        domain_data = {
            'domain': domain,
            'experiments': [],
            'outcomes': [],
            'features': [],
            'mvdp_data': [],
            'stats': {}
        }

        try:
            logger.info(f"Starting async mining for domain: {domain}")

            # Create async tasks for all data sources
            tasks = []

            # Literature mining task
            if 'literature' in config.get('data_sources', []):
                tasks.append(self._mine_literature_for_experiments(domain, config))

            # ENCODE data acquisition task
            if 'encode' in config.get('data_sources', []):
                tasks.append(self._acquire_encode_data(domain, config))

            # GEO data acquisition task
            if 'geo' in config.get('data_sources', []):
                tasks.append(self._acquire_geo_data(domain, config))

            # Execute all tasks concurrently
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Task {i} failed for domain {domain}: {result}")
                        self.stats['errors'].append(f"Task {i} error for {domain}: {result}")
                    else:
                        domain_data['experiments'].extend(result)

            # Process experiments to extract outcomes and features
            domain_data['outcomes'] = self._extract_outcomes(domain_data['experiments'], domain)
            domain_data['features'] = self._extract_features(domain_data['experiments'], domain)
            domain_data['mvdp_data'] = self._apply_mvdp_criteria(domain_data['experiments'], domain)

            # Calculate domain statistics
            domain_data['stats'] = self._calculate_domain_statistics(domain_data)

            logger.info(f"Completed mining for domain: {domain} - {len(domain_data['experiments'])} experiments")

        except Exception as e:
            logger.error(f"Domain mining failed for {domain}: {e}")
            self.stats['errors'].append(f"Domain mining error for {domain}: {e}")
            domain_data['error'] = str(e)

        return domain_data

    def _extract_outcomes(self, experiments: List[Dict], domain: str) -> List[Dict]:
        """Extract outcome metrics from experiments"""
        outcomes = []

        for exp in experiments:
            try:
                if exp['source'] == 'PubMed' and 'paper' in exp:
                    paper = exp['paper']

                    # Extract efficiency scores based on domain
                    efficiency_score = None
                    if domain == 'crispr':
                        efficiency_score = self._extract_crispr_efficiency(paper)
                    elif domain == 'prime_editing':
                        efficiency_score = self._extract_prime_editing_efficiency(paper)
                    elif domain == 'base_editing':
                        efficiency_score = self._extract_base_editing_efficiency(paper)

                    if efficiency_score is not None:
                        outcomes.append({
                            'experiment_id': paper.get('pmid', 'unknown'),
                            'efficiency_score': efficiency_score,
                            'domain': domain,
                            'source': exp['source']
                        })

            except Exception as e:
                logger.error(f"Error extracting outcomes from experiment: {e}")

        return outcomes

    def _extract_features(self, experiments: List[Dict], domain: str) -> List[Dict]:
        """Extract features from experiments"""
        features = []

        for exp in experiments:
            try:
                if exp['source'] == 'PubMed' and 'paper' in exp:
                    paper = exp['paper']

                    feature = {
                        'experiment_id': paper.get('pmid', 'unknown'),
                        'title': paper.get('title', ''),
                        'abstract': paper.get('abstract', ''),
                        'year': paper.get('year'),
                        'journal': paper.get('journal', ''),
                        'authors': paper.get('authors', []),
                        'domain': domain,
                        'source': exp['source']
                    }

                    features.append(feature)

            except Exception as e:
                logger.error(f"Error extracting features from experiment: {e}")

        return features

    def _apply_mvdp_criteria(self, experiments: List[Dict], domain: str) -> List[Dict]:
        """Apply Minimum Viable Data Point criteria"""
        mvdp_data = []

        for exp in experiments:
            try:
                if exp['source'] == 'PubMed' and 'paper' in exp:
                    paper = exp['paper']

                    # Check MVDP criteria with proper None handling
                    has_title = bool(paper.get('title'))
                    has_abstract = bool(paper.get('abstract'))
                    has_year = paper.get('year') is not None
                    has_authors = bool(paper.get('authors'))

                    # Domain-specific criteria with None handling
                    domain_criteria_met = False
                    title_text = paper.get('title', '') or ''
                    abstract_text = paper.get('abstract', '') or ''

                    if domain == 'crispr':
                        domain_criteria_met = 'crispr' in title_text.lower() or 'crispr' in abstract_text.lower()
                    elif domain == 'prime_editing':
                        domain_criteria_met = 'prime editing' in title_text.lower() or 'prime editing' in abstract_text.lower()
                    elif domain == 'base_editing':
                        domain_criteria_met = 'base editing' in title_text.lower() or 'base editing' in abstract_text.lower()

                    # Calculate completeness score
                    completeness_score = sum([has_title, has_abstract, has_year, has_authors, domain_criteria_met]) / 5.0

                    if completeness_score >= 0.8:  # 80% completeness threshold
                        mvdp_data.append({
                            'experiment_id': paper.get('pmid', 'unknown'),
                            'completeness_score': completeness_score,
                            'domain': domain,
                            'source': exp['source'],
                            'meets_mvdp': True
                        })

            except Exception as e:
                logger.error(f"Error applying MVDP criteria to experiment: {e}")

        return mvdp_data

    def _extract_crispr_efficiency(self, paper: Dict) -> Optional[float]:
        """Extract CRISPR efficiency from paper content"""
        # Simple keyword-based extraction
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        # Look for efficiency patterns
        import re
        efficiency_patterns = [
            r'(\d+(?:\.\d+)?)\s*%?\s*efficiency',
            r'efficiency\s*of\s*(\d+(?:\.\d+)?)\s*%?',
            r'(\d+(?:\.\d+)?)\s*%?\s*editing',
            r'editing\s*efficiency\s*(\d+(?:\.\d+)?)\s*%?'
        ]

        for pattern in efficiency_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    efficiency = float(match.group(1))
                    if 0 <= efficiency <= 100:
                        return efficiency / 100.0  # Normalize to 0-1
                except ValueError:
                    continue

        return None

    def _extract_prime_editing_efficiency(self, paper: Dict) -> Optional[float]:
        """Extract Prime editing efficiency from paper content"""
        # Similar to CRISPR but with Prime editing specific terms
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        import re
        efficiency_patterns = [
            r'(\d+(?:\.\d+)?)\s*%?\s*prime\s*editing\s*efficiency',
            r'prime\s*editing\s*efficiency\s*(\d+(?:\.\d+)?)\s*%?',
            r'(\d+(?:\.\d+)?)\s*%?\s*pegRNA\s*efficiency'
        ]

        for pattern in efficiency_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    efficiency = float(match.group(1))
                    if 0 <= efficiency <= 100:
                        return efficiency / 100.0
                except ValueError:
                    continue

        return None

    def _extract_base_editing_efficiency(self, paper: Dict) -> Optional[float]:
        """Extract Base editing efficiency from paper content"""
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        import re
        efficiency_patterns = [
            r'(\d+(?:\.\d+)?)\s*%?\s*base\s*editing\s*efficiency',
            r'base\s*editing\s*efficiency\s*(\d+(?:\.\d+)?)\s*%?',
            r'(\d+(?:\.\d+)?)\s*%?\s*conversion\s*efficiency'
        ]

        for pattern in efficiency_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    efficiency = float(match.group(1))
                    if 0 <= efficiency <= 100:
                        return efficiency / 100.0
                except ValueError:
                    continue

        return None

    def _calculate_domain_statistics(self, domain_data: Dict) -> Dict:
        """Calculate statistics for a domain"""
        stats = {
            'total_experiments': len(domain_data['experiments']),
            'total_outcomes': len(domain_data['outcomes']),
            'total_features': len(domain_data['features']),
            'total_mvdp_compliant': len(domain_data['mvdp_data']),
            'mvdp_compliance_rate': 0.0
        }

        if stats['total_experiments'] > 0:
            stats['mvdp_compliance_rate'] = stats['total_mvdp_compliant'] / stats['total_experiments']

        return stats

    async def mine_all_domains(self) -> Dict:
        """
        Mine data for all domains using async methods
        """
        start_time = time.time()
        logger.info("Starting comprehensive async mining for all domains")

        try:
            # Initialize results
            all_results = {
                'domains': {},
                'global_stats': {},
                'mining_session': {
                    'start_time': datetime.now().isoformat(),
                    'duration': 0,
                    'status': 'running'
                }
            }

            # Create async tasks for all domains
            domain_tasks = []
            for domain, config in self.domain_configs.items():
                logger.info(f"Creating mining task for domain: {domain}")
                task = self._mine_domain_data(domain, config)
                domain_tasks.append(task)

            # Execute all domain tasks concurrently
            domain_results = await asyncio.gather(*domain_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(domain_results):
                domain_name = list(self.domain_configs.keys())[i]

                if isinstance(result, Exception):
                    logger.error(f"Domain {domain_name} failed: {result}")
                    all_results['domains'][domain_name] = {'error': str(result)}
                else:
                    all_results['domains'][domain_name] = result

            # Calculate global statistics
            all_results['global_stats'] = self._calculate_global_statistics(all_results['domains'])

            # Update session info
            duration = time.time() - start_time
            all_results['mining_session']['duration'] = duration
            all_results['mining_session']['status'] = 'completed'

            logger.info(f"Completed comprehensive mining in {duration:.2f} seconds")

        except Exception as e:
            logger.error(f"Comprehensive mining failed: {e}")
            all_results = {
                'error': str(e),
                'mining_session': {
                    'start_time': datetime.now().isoformat(),
                    'duration': time.time() - start_time,
                    'status': 'failed'
                }
            }

        return all_results

    def _calculate_global_statistics(self, domains_data: Dict) -> Dict:
        """
        Calculate global statistics across all domains
        """
        global_stats = {
            'total_experiments': 0,
            'total_outcomes': 0,
            'total_features': 0,
            'total_mvdp_compliant': 0,
            'domains_processed': 0,
            'domains_failed': 0,
            'unique_sources': set(),
            'error_count': len(self.stats.get('errors', []))
        }

        for domain, data in domains_data.items():
            if isinstance(data, dict) and 'error' not in data:
                # Successful domain processing
                global_stats['domains_processed'] += 1
                global_stats['total_experiments'] += data.get('stats', {}).get('total_experiments', 0)
                global_stats['total_outcomes'] += data.get('stats', {}).get('total_outcomes', 0)
                global_stats['total_features'] += data.get('stats', {}).get('total_features', 0)
                global_stats['total_mvdp_compliant'] += data.get('stats', {}).get('total_mvdp_compliant', 0)

                # Track unique sources
                for exp in data.get('experiments', []):
                    if 'source' in exp:
                        global_stats['unique_sources'].add(exp['source'])
            else:
                # Failed domain processing
                global_stats['domains_failed'] += 1

        # Convert set to list for JSON serialization
        global_stats['unique_sources'] = list(global_stats['unique_sources'])

        return global_stats

    # Properties for compatibility with existing code
    @property
    def pubmed_client(self):
        return self.clients.get('pubmed')

    @property
    def semantic_scholar_client(self):
        return self.clients.get('semantic_scholar')

    @property
    def data_validator(self):
        # Return a simple validator object for compatibility
        class SimpleValidator:
            def validate_data_point(self, data_point):
                return True, []
        return SimpleValidator()

    def start_mining(self):
        """Alias for mine_all_domains for compatibility"""
        return asyncio.run(self.mine_all_domains())
