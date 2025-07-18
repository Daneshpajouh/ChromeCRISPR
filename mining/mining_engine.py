"""
Enhanced Mining Engine for GeneX Phase 1
Comprehensive data mining system with advanced features, analytics, and quality control.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import statistics
import yaml

# Import API clients using relative imports
from ..api_clients.pubmed_client import PubMedClient
from ..api_clients.semantic_scholar_client import SemanticScholarClient
from .data_validator import DataValidator
from .quality_controller import QualityController

logger = logging.getLogger(__name__)

@dataclass
class MiningSession:
    """Comprehensive mining session metadata"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Search configuration
    search_terms: List[str] = field(default_factory=list)
    max_papers_per_term: int = 100
    enable_validation: bool = True
    enable_quality_control: bool = True

    # Results summary
    total_papers_mined: int = 0
    total_papers_validated: int = 0
    total_papers_quality_controlled: int = 0

    # Performance metrics
    total_processing_time: float = 0.0
    average_papers_per_second: float = 0.0

    # Quality metrics
    average_quality_score: float = 0.0
    validation_success_rate: float = 0.0

    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class MiningAnalytics:
    """Comprehensive analytics for mining operations"""
    total_sessions: int = 0
    total_papers_processed: int = 0
    total_processing_time: float = 0.0

    # Source distribution
    source_distribution: Dict[str, int] = field(default_factory=dict)

    # Quality metrics over time
    quality_scores: List[float] = field(default_factory=list)
    validation_rates: List[float] = field(default_factory=list)

    # Performance metrics
    average_session_time: float = 0.0
    papers_per_session: List[int] = field(default_factory=list)

    # Error analysis
    error_frequency: Dict[str, int] = field(default_factory=dict)

    # Search term analysis
    search_term_effectiveness: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class EnhancedMiningEngine:
    """
    Enhanced mining engine with comprehensive features:
    - Multi-source data collection
    - Advanced deduplication
    - Quality control and validation
    - Analytics and reporting
    - Async processing
    - Error handling and recovery
    """

    def __init__(self, api_configs: Any, output_dir: str = "data"):
        """Initialize the enhanced mining engine. Accepts config dict or path to YAML config file."""
        # If a string is passed, treat it as a path to a YAML config file
        if isinstance(api_configs, str):
            with open(api_configs, 'r') as f:
                config = yaml.safe_load(f)
            # Extract the 'apis' section for API client configs
            api_configs = config.get('apis', {})

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize API clients
        self.clients = {}
        self._initialize_clients(api_configs)

        # Initialize components
        self.validator = DataValidator()
        self.quality_controller = QualityController()

        # Session management
        self.current_session: Optional[MiningSession] = None
        self.analytics = MiningAnalytics()

        # Performance tracking
        self.start_time = datetime.now()

        logger.info("Enhanced Mining Engine initialized successfully")

    def _initialize_clients(self, api_configs: Dict[str, Any]):
        """Initialize API clients with proper configuration."""
        try:
            # Initialize PubMed client
            if 'pubmed' in api_configs:
                pubmed_config = api_configs['pubmed']
                self.clients['pubmed'] = PubMedClient(pubmed_config)
                logger.info("PubMed client initialized")

            # Initialize Semantic Scholar client
            if 'semantic_scholar' in api_configs:
                ss_config = api_configs['semantic_scholar']
                self.clients['semantic_scholar'] = SemanticScholarClient(ss_config)
                logger.info("Semantic Scholar client initialized")

            logger.info(f"Initialized {len(self.clients)} API clients")

        except Exception as e:
            logger.error(f"Error initializing API clients: {e}")
            raise

    def start_session(self, search_terms: List[str], **kwargs) -> str:
        """Start a new mining session."""
        session_id = f"session_{int(time.time())}_{hashlib.md5(str(search_terms).encode()).hexdigest()[:8]}"

        self.current_session = MiningSession(
            session_id=session_id,
            start_time=datetime.now(),
            search_terms=search_terms,
            max_papers_per_term=kwargs.get('max_papers_per_term', 100),
            enable_validation=kwargs.get('enable_validation', True),
            enable_quality_control=kwargs.get('enable_quality_control', True)
        )

        logger.info(f"Started mining session {session_id} with {len(search_terms)} search terms")
        return session_id

    def end_session(self):
        """End the current mining session and update analytics."""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.current_session.total_processing_time = (
                self.current_session.end_time - self.current_session.start_time
            ).total_seconds()

            # Update analytics
            self.analytics.total_sessions += 1
            self.analytics.total_papers_processed += self.current_session.total_papers_mined
            self.analytics.total_processing_time += self.current_session.total_processing_time

            if self.current_session.total_processing_time > 0:
                self.current_session.average_papers_per_second = (
                    self.current_session.total_papers_mined /
                    self.current_session.total_processing_time
                )

            logger.info(f"Ended session {self.current_session.session_id}")
            logger.info(f"Session summary: {self.current_session.total_papers_mined} papers in {self.current_session.total_processing_time:.2f}s")

    def mine_comprehensive(self,
                          search_terms: List[str],
                          max_papers_per_term: int = 100,
                          enable_validation: bool = True,
                          enable_quality_control: bool = True,
                          save_results: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Comprehensive mining operation with all features enabled.

        Args:
            search_terms: List of search terms to mine
            max_papers_per_term: Maximum papers per search term
            enable_validation: Enable data validation
            enable_quality_control: Enable quality control
            save_results: Save results to files

        Returns:
            Dictionary mapping search terms to lists of papers
        """
        try:
            # Start session
            session_id = self.start_session(
                search_terms=search_terms,
                max_papers_per_term=max_papers_per_term,
                enable_validation=enable_validation,
                enable_quality_control=enable_quality_control
            )

            logger.info(f"Starting comprehensive mining for {len(search_terms)} search terms")

            # Collect papers from all sources
            all_papers = {}

            for term in search_terms:
                logger.info(f"Mining papers for: {term}")
                papers = self._mine_single_term(term, max_papers_per_term)
                all_papers[term] = papers

                # Update session metrics
                self.current_session.total_papers_mined += len(papers)

            # Deduplicate across all terms
            if len(search_terms) > 1:
                logger.info("Performing cross-term deduplication")
                all_papers = self._deduplicate_across_terms(all_papers)

            # Validate papers if enabled
            if enable_validation:
                logger.info("Validating papers")
                all_papers = self._validate_papers(all_papers)

            # Quality control if enabled
            if enable_quality_control:
                logger.info("Performing quality control")
                all_papers = self._quality_control_papers(all_papers)

            # Save results if requested
            if save_results:
                self._save_mining_results(all_papers, session_id)

            # End session
            self.end_session()

            logger.info(f"Comprehensive mining completed. Total papers: {sum(len(papers) for papers in all_papers.values())}")

            # Alias for compatibility with test scripts
            start_mining = self.mine_comprehensive

            return all_papers

        except Exception as e:
            logger.error(f"Error in comprehensive mining: {e}")
            if self.current_session:
                self.current_session.errors.append(str(e))
            raise

    def _mine_single_term(self, term: str, max_papers: int) -> List[Dict[str, Any]]:
        """Mine papers for a single search term from all available sources."""
        papers = []

        # Mine from PubMed
        if 'pubmed' in self.clients:
            try:
                logger.info(f"Mining PubMed for: {term}")
                response = self.clients['pubmed'].search(term, max_results=max_papers)

                if response.success:
                    pubmed_papers = self.clients['pubmed'].parse_response(response)
                    papers.extend(pubmed_papers)
                    logger.info(f"Found {len(pubmed_papers)} papers from PubMed")
                else:
                    logger.warning(f"PubMed search failed for '{term}': {response.error}")

            except Exception as e:
                logger.error(f"Error mining PubMed for '{term}': {e}")

        # Mine from Semantic Scholar
        if 'semantic_scholar' in self.clients:
            try:
                logger.info(f"Mining Semantic Scholar for: {term}")
                response = self.clients['semantic_scholar'].search(term, max_results=max_papers)

                if response.success:
                    ss_papers = self.clients['semantic_scholar'].parse_response(response)
                    papers.extend(ss_papers)
                    logger.info(f"Found {len(ss_papers)} papers from Semantic Scholar")
                else:
                    logger.warning(f"Semantic Scholar search failed for '{term}': {response.error}")

            except Exception as e:
                logger.error(f"Error mining Semantic Scholar for '{term}': {e}")

        return papers

    def _deduplicate_across_terms(self, all_papers: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Remove duplicate papers across different search terms."""
        seen_papers = set()
        deduplicated = {}

        for term, papers in all_papers.items():
            deduplicated_papers = []

            for paper in papers:
                # Create unique identifier for paper
                paper_id = self._get_paper_identifier(paper)

                if paper_id not in seen_papers:
                    seen_papers.add(paper_id)
                    deduplicated_papers.append(paper)

            deduplicated[term] = deduplicated_papers
            logger.info(f"Deduplication for '{term}': {len(papers)} -> {len(deduplicated_papers)} papers")

        return deduplicated

    def _get_paper_identifier(self, paper: Dict[str, Any]) -> str:
        """Generate unique identifier for paper based on DOI, title, and authors."""
        # Try DOI first
        doi = paper.get('doi', '').strip()
        if doi:
            return f"doi:{doi.lower()}"

        # Fall back to title + first author
        title = paper.get('title', '').strip().lower()
        authors = paper.get('authors', [])
        first_author = authors[0] if authors else ''

        if title and first_author:
            return f"title:{title[:100]}:author:{first_author[:50]}"

        # Last resort: title only
        return f"title:{title[:100]}"

    def _validate_papers(self, all_papers: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Validate papers using the data validator."""
        validated = {}
        total_validated = 0

        for term, papers in all_papers.items():
            validated_papers = []

            for paper in papers:
                validation_result = self.validator.validate_paper(paper)
                if validation_result['is_valid']:
                    paper['validation_status'] = 'valid'
                    paper['validation_score'] = validation_result['score']
                    validated_papers.append(paper)
                else:
                    paper['validation_status'] = 'invalid'
                    paper['validation_errors'] = validation_result['errors']
                    # Still include invalid papers but mark them
                    validated_papers.append(paper)

            validated[term] = validated_papers
            total_validated += len(validated_papers)

            if self.current_session:
                self.current_session.total_papers_validated += len(validated_papers)

        logger.info(f"Validation completed: {total_validated} papers processed")
        return validated

    def _quality_control_papers(self, all_papers: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Apply quality control to papers."""
        quality_controlled = {}
        total_quality_controlled = 0

        for term, papers in all_papers.items():
            quality_papers = []

            for paper in papers:
                quality_result = self.quality_controller.assess_paper_quality(paper)
                paper['quality_score'] = quality_result['overall_score']
                paper['quality_assessment'] = quality_result
                quality_papers.append(paper)

            quality_controlled[term] = quality_papers
            total_quality_controlled += len(quality_papers)

            if self.current_session:
                self.current_session.total_papers_quality_controlled += len(quality_papers)

        logger.info(f"Quality control completed: {total_quality_controlled} papers processed")
        return quality_controlled

    def _save_mining_results(self, all_papers: Dict[str, List[Dict[str, Any]]], session_id: str):
        """Save mining results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save papers by search term
        for term, papers in all_papers.items():
            if papers:
                # Create safe filename
                safe_term = "".join(c for c in term if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_term = safe_term.replace(' ', '_')

                # Save as JSON
                json_file = self.output_dir / "mining" / f"session_{session_id}_{safe_term}_{timestamp}.json"
                json_file.parent.mkdir(parents=True, exist_ok=True)

                with open(json_file, 'w') as f:
                    json.dump(papers, f, indent=2, default=str)

                logger.info(f"Saved {len(papers)} papers for '{term}' to {json_file}")

        # Save session summary
        if self.current_session:
            session_file = self.output_dir / "mining" / f"session_{session_id}_summary_{timestamp}.json"
            session_file.parent.mkdir(parents=True, exist_ok=True)

            session_data = {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time.isoformat(),
                'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                'search_terms': self.current_session.search_terms,
                'total_papers_mined': self.current_session.total_papers_mined,
                'total_papers_validated': self.current_session.total_papers_validated,
                'total_papers_quality_controlled': self.current_session.total_papers_quality_controlled,
                'total_processing_time': self.current_session.total_processing_time,
                'average_papers_per_second': self.current_session.average_papers_per_second,
                'errors': self.current_session.errors,
                'warnings': self.current_session.warnings
            }

            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

            logger.info(f"Saved session summary to {session_file}")

    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about mining operations."""
        return {
            'total_sessions': self.analytics.total_sessions,
            'total_papers_processed': self.analytics.total_papers_processed,
            'total_processing_time': self.analytics.total_processing_time,
            'average_session_time': self.analytics.average_session_time,
            'source_distribution': self.analytics.source_distribution,
            'quality_scores': {
                'mean': statistics.mean(self.analytics.quality_scores) if self.analytics.quality_scores else 0,
                'median': statistics.median(self.analytics.quality_scores) if self.analytics.quality_scores else 0,
                'min': min(self.analytics.quality_scores) if self.analytics.quality_scores else 0,
                'max': max(self.analytics.quality_scores) if self.analytics.quality_scores else 0
            },
            'error_frequency': self.analytics.error_frequency
        }

    def clear_cache(self):
        """Clear cache for all API clients."""
        for client_name, client in self.clients.items():
            try:
                if hasattr(client, 'clear_cache'):
                    client.clear_cache()
                    logger.info(f"Cleared cache for {client_name}")
            except Exception as e:
                logger.warning(f"Error clearing cache for {client_name}: {e}")

    def get_cache_size(self) -> Dict[str, int]:
        """Get cache size for all API clients."""
        cache_sizes = {}
        for client_name, client in self.clients.items():
            try:
                if hasattr(client, 'get_cache_size'):
                    cache_sizes[client_name] = client.get_cache_size()
                else:
                    cache_sizes[client_name] = 0
            except Exception as e:
                logger.warning(f"Error getting cache size for {client_name}: {e}")
                cache_sizes[client_name] = 0
        return cache_sizes

    @property
    def pubmed_client(self):
        return self.clients.get('pubmed', None)

    @property
    def semantic_scholar_client(self):
        return self.clients.get('semantic_scholar', None)

    @property
    def data_validator(self):
        return self.validator

    # Alias for compatibility with test scripts
    start_mining = mine_comprehensive

# Alias for compatibility with test scripts and legacy code
MiningEngine = EnhancedMiningEngine
