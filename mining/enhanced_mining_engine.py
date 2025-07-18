"""
Enhanced GeneX Mining Engine - Comprehensive Multi-API Data Mining
Handles all 11 projects with parallel API requests and data point extraction
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from ..api_clients.pubmed_client import PubMedClient
from ..api_clients.semantic_scholar_client import SemanticScholarClient
from ..api_clients.crossref_client import CrossRefClient
from ..api_clients.ncbi_client import NCBIClient
from ..api_clients.ensembl_client import EnsemblClient
from ..utils.config import Config
from ..utils.logger import setup_logger
from .data_validator import DataValidator
from .quality_controller import QualityController

logger = setup_logger(__name__)

class EnhancedMiningEngine:
    """
    Enhanced mining engine that processes all 11 GeneX projects
    with parallel API requests and comprehensive data extraction
    """

    def __init__(self, config: Config):
        self.config = config
        self.session_id = f"session_{int(time.time())}_{hash(datetime.now())}"
        self.start_time = time.time()

        # Initialize all API clients
        self.clients = {}
        self._initialize_clients()

        # Initialize processing components
        self.validator = DataValidator(config)
        self.quality_controller = QualityController(config)

        # Analytics tracking
        self.analytics = {
            'total_sessions': 0,
            'total_papers_processed': 0,
            'total_data_points_extracted': 0,
            'total_processing_time': 0,
            'api_usage': {},
            'project_stats': {},
            'error_frequency': {}
        }

        logger.info(f"Enhanced Mining Engine initialized with {len(self.clients)} API clients")

    def _initialize_clients(self):
        """Initialize all available API clients"""
        try:
            # Literature APIs
            self.clients['pubmed'] = PubMedClient(self.config)
            self.clients['semantic_scholar'] = SemanticScholarClient(self.config)
            self.clients['crossref'] = CrossRefClient(self.config)

            # Genomic APIs
            self.clients['ncbi'] = NCBIClient(self.config)
            self.clients['ensembl'] = EnsemblClient(self.config)

            logger.info(f"Initialized {len(self.clients)} API clients")

        except Exception as e:
            logger.error(f"Error initializing API clients: {e}")
            raise

    async def mine_all_projects(self) -> Dict[str, Any]:
        """
        Mine all 11 GeneX projects with parallel processing
        """
        logger.info("Starting comprehensive mining of all 11 GeneX projects")

        results = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'projects': {},
            'summary': {},
            'analytics': {}
        }

        # Get all projects from config
        projects = self.config.get('projects', {})

        # Process projects in parallel (but with rate limiting)
        tasks = []
        for project_id, project_config in projects.items():
            task = self._mine_project_async(project_id, project_config)
            tasks.append(task)

        # Execute all projects concurrently
        project_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, (project_id, project_config) in enumerate(projects.items()):
            if isinstance(project_results[i], Exception):
                logger.error(f"Project {project_id} failed: {project_results[i]}")
                results['projects'][project_id] = {
                    'status': 'failed',
                    'error': str(project_results[i])
                }
            else:
                results['projects'][project_id] = project_results[i]

        # Generate comprehensive summary
        results['summary'] = self._generate_comprehensive_summary(results['projects'])
        results['analytics'] = self.analytics
        results['end_time'] = datetime.now().isoformat()

        # Save comprehensive results
        self._save_comprehensive_results(results)

        logger.info(f"Comprehensive mining completed. Processed {len(projects)} projects")
        return results

    async def _mine_project_async(self, project_id: str, project_config: Dict) -> Dict[str, Any]:
        """
        Mine a single project with all available APIs in parallel
        """
        logger.info(f"Starting mining for project: {project_config.get('name', project_id)}")

        project_start = time.time()
        project_results = {
            'project_id': project_id,
            'name': project_config.get('name', project_id),
            'search_terms': project_config.get('search_terms', []),
            'genes': project_config.get('genes', []),
            'diseases': project_config.get('diseases', []),
            'papers': {},
            'data_points': {},
            'api_results': {},
            'processing_time': 0
        }

        # Extract search terms and data points
        search_terms = project_config.get('search_terms', [])
        genes = project_config.get('genes', [])
        diseases = project_config.get('diseases', [])

        # Create parallel tasks for different data types
        tasks = []

        # 1. Literature mining (papers) - parallel across APIs
        if search_terms:
            for term in search_terms:
                tasks.append(self._mine_literature_parallel(term))

        # 2. Genomic data mining (genes) - parallel across APIs
        if genes:
            tasks.append(self._mine_genomic_data_parallel(genes))

        # 3. Clinical data mining (diseases) - parallel across APIs
        if diseases:
            tasks.append(self._mine_clinical_data_parallel(diseases))

        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                    continue

                if 'literature' in result:
                    project_results['papers'].update(result['literature'])
                if 'genomic' in result:
                    project_results['data_points']['genomic'] = result['genomic']
                if 'clinical' in result:
                    project_results['data_points']['clinical'] = result['clinical']

        project_results['processing_time'] = time.time() - project_start

        # Update analytics
        self.analytics['total_sessions'] += 1
        self.analytics['total_papers_processed'] += len(project_results['papers'])
        self.analytics['total_data_points_extracted'] += len(project_results.get('data_points', {}))

        logger.info(f"Project {project_id} completed: {len(project_results['papers'])} papers, "
                   f"{len(project_results.get('data_points', {}))} data points")

        return project_results

    async def _mine_literature_parallel(self, search_term: str) -> Dict[str, Any]:
        """
        Mine literature data from multiple APIs in parallel
        """
        logger.info(f"Mining literature for: {search_term}")

        # Create parallel tasks for different literature APIs
        tasks = []

        # PubMed
        if 'pubmed' in self.clients:
            tasks.append(self._mine_pubmed(search_term))

        # Semantic Scholar
        if 'semantic_scholar' in self.clients:
            tasks.append(self._mine_semantic_scholar(search_term))

        # CrossRef
        if 'crossref' in self.clients:
            tasks.append(self._mine_crossref(search_term))

        # Execute literature mining in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            literature_results = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Literature API failed: {result}")
                    continue

                api_name = list(self.clients.keys())[i] if i < len(self.clients) else f"api_{i}"
                literature_results[api_name] = result

            return {'literature': literature_results}

        return {'literature': {}}

    async def _mine_genomic_data_parallel(self, genes: List[str]) -> Dict[str, Any]:
        """
        Mine genomic data from multiple APIs in parallel
        """
        logger.info(f"Mining genomic data for {len(genes)} genes")

        tasks = []

        # NCBI Gene data
        if 'ncbi' in self.clients:
            tasks.append(self._mine_ncbi_genes(genes))

        # Ensembl data
        if 'ensembl' in self.clients:
            tasks.append(self._mine_ensembl_genes(genes))

        # Execute genomic mining in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            genomic_results = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Genomic API failed: {result}")
                    continue

                api_name = ['ncbi', 'ensembl'][i] if i < 2 else f"genomic_api_{i}"
                genomic_results[api_name] = result

            return {'genomic': genomic_results}

        return {'genomic': {}}

    async def _mine_clinical_data_parallel(self, diseases: List[str]) -> Dict[str, Any]:
        """
        Mine clinical data from multiple APIs in parallel
        """
        logger.info(f"Mining clinical data for {len(diseases)} diseases")

        # For now, we'll use literature APIs to find clinical data
        # In the future, we can add specific clinical APIs
        tasks = []

        for disease in diseases:
            if 'pubmed' in self.clients:
                tasks.append(self._mine_pubmed(f"{disease} clinical trial"))
            if 'semantic_scholar' in self.clients:
                tasks.append(self._mine_semantic_scholar(f"{disease} clinical trial"))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            clinical_results = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Clinical API failed: {result}")
                    continue

                api_name = ['pubmed', 'semantic_scholar'][i % 2]
                disease = diseases[i // 2] if i // 2 < len(diseases) else f"disease_{i}"
                clinical_results[f"{api_name}_{disease}"] = result

            return {'clinical': clinical_results}

        return {'clinical': {}}

    async def _mine_pubmed(self, search_term: str) -> List[Dict]:
        """Mine PubMed data"""
        try:
            client = self.clients['pubmed']
            papers = await client.search_papers_async(search_term, max_results=10)
            return papers
        except Exception as e:
            logger.error(f"PubMed mining failed: {e}")
            return []

    async def _mine_semantic_scholar(self, search_term: str) -> List[Dict]:
        """Mine Semantic Scholar data"""
        try:
            client = self.clients['semantic_scholar']
            papers = await client.search_papers_async(search_term, max_results=10)
            return papers
        except Exception as e:
            logger.error(f"Semantic Scholar mining failed: {e}")
            return []

    async def _mine_crossref(self, search_term: str) -> List[Dict]:
        """Mine CrossRef data"""
        try:
            client = self.clients['crossref']
            papers = await client.search_papers_async(search_term, max_results=10)
            return papers
        except Exception as e:
            logger.error(f"CrossRef mining failed: {e}")
            return []

    async def _mine_ncbi_genes(self, genes: List[str]) -> Dict[str, Any]:
        """Mine NCBI gene data"""
        try:
            client = self.clients['ncbi']
            gene_data = {}
            for gene in genes[:5]:  # Limit to 5 genes for mini test
                data = await client.get_gene_info_async(gene)
                if data:
                    gene_data[gene] = data
            return gene_data
        except Exception as e:
            logger.error(f"NCBI gene mining failed: {e}")
            return {}

    async def _mine_ensembl_genes(self, genes: List[str]) -> Dict[str, Any]:
        """Mine Ensembl gene data"""
        try:
            client = self.clients['ensembl']
            gene_data = {}
            for gene in genes[:5]:  # Limit to 5 genes for mini test
                data = await client.get_gene_info_async(gene)
                if data:
                    gene_data[gene] = data
            return gene_data
        except Exception as e:
            logger.error(f"Ensembl gene mining failed: {e}")
            return {}

    def _generate_comprehensive_summary(self, project_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive summary of all projects"""
        total_papers = 0
        total_data_points = 0
        successful_projects = 0
        total_processing_time = 0

        for project_id, result in project_results.items():
            if result.get('status') != 'failed':
                successful_projects += 1
                total_papers += len(result.get('papers', {}))
                total_data_points += len(result.get('data_points', {}))
                total_processing_time += result.get('processing_time', 0)

        return {
            'total_projects': len(project_results),
            'successful_projects': successful_projects,
            'total_papers': total_papers,
            'total_data_points': total_data_points,
            'total_processing_time': total_processing_time,
            'average_papers_per_project': total_papers / max(successful_projects, 1),
            'average_data_points_per_project': total_data_points / max(successful_projects, 1)
        }

    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create data directory
        data_dir = Path("data/mining")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save comprehensive results
        results_file = data_dir / f"comprehensive_session_{self.session_id}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary
        summary_file = data_dir / f"comprehensive_summary_{self.session_id}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results['summary'], f, indent=2, default=str)

        logger.info(f"Comprehensive results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all clients"""
        metrics = {}
        for name, client in self.clients.items():
            try:
                metrics[name] = client.get_metrics()
            except Exception as e:
                logger.error(f"Error getting metrics for {name}: {e}")
                metrics[name] = {'error': str(e)}
        return metrics

    def clear_all_caches(self):
        """Clear caches for all clients"""
        for name, client in self.clients.items():
            try:
                client.clear_cache()
                logger.info(f"Cleared cache for {name}")
            except Exception as e:
                logger.error(f"Error clearing cache for {name}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_all_caches()
