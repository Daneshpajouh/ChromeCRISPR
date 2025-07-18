"""
Enhanced Semantic Scholar API Client for GeneX Phase 1
Provides comprehensive access to Semantic Scholar database via their API.
Enhanced with advanced features for bulk operations, async support, and analytics.
"""

import logging
import time
import hashlib
import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)

@dataclass
class SemanticScholarPaper:
    """Enhanced Semantic Scholar paper data structure with comprehensive metadata"""
    # Basic identification
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    venue: str
    year: Optional[int]
    doi: Optional[str]

    # Enhanced metadata
    publication_date: Optional[str]
    publication_types: List[str]
    fields_of_study: List[str]
    keywords: List[str]
    language: str

    # Citation and impact
    citation_count: int
    reference_count: int
    influential_citation_count: int

    # Source and processing info
    source: str = "Semantic Scholar"
    collection_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_pipeline: str = "GeneX_v2.0"

    # Quality metrics
    completeness_score: float = 0.0
    quality_score: float = 0.0
    checksum: str = ""

    # Gene editing specific features
    gene_editing_techniques: List[str] = field(default_factory=list)
    target_organisms: List[str] = field(default_factory=list)
    target_genes: List[str] = field(default_factory=list)
    experimental_methods: List[str] = field(default_factory=list)
    therapeutic_applications: List[str] = field(default_factory=list)

class SemanticScholarClient(BaseAPIClient):
    """
    Semantic Scholar API client for academic paper data.

    Provides access to paper metadata, citations, references, and embeddings.
    """

    def __init__(self, config: Dict[str, Any], session=None):
        """Initialize Semantic Scholar client"""
        super().__init__(config, session)

        # Semantic Scholar specific configuration
        self.default_fields = [
            'paperId', 'title', 'abstract', 'year', 'authors',
            'venue', 'url', 'doi', 'citationCount', 'openAccessPdf',
            'publicationTypes', 'publicationDate', 'journal',
            'references', 'citations', 'embedding', 'tldr'
        ]

    def _build_search_params(self, query: str, limit: int = 100, **kwargs) -> Dict[str, Any]:
        """Build parameters for paper search"""
        params = {
            'query': query,
            'limit': min(limit, 100),  # API limit
            'fields': ','.join(self.default_fields)
        }

        # Add optional parameters
        if 'year' in kwargs:
            params['year'] = kwargs['year']
        if 'venue' in kwargs:
            params['venue'] = kwargs['venue']
        if 'publicationTypes' in kwargs:
            params['publicationTypes'] = kwargs['publicationTypes']

        return params

    def _build_paper_params(self, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build parameters for paper details request"""
        if fields is None:
            fields = self.default_fields

        return {
            'fields': ','.join(fields)
        }

    async def search_papers(self, query: str, limit: int = 100, **kwargs) -> APIResponse:
        """
        Search for papers using Semantic Scholar API.

        Args:
            query: Search query string
            limit: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            APIResponse with search results
        """
        logger.info(f"Searching Semantic Scholar for: {query} (limit: {limit})")

        try:
            params = self._build_search_params(query, limit, **kwargs)
            response = await self.get('paper/search', params=params)

            if response and response.success and response.data:
                total = response.data.get('total', 0)
                logger.info(f"Semantic Scholar search returned {total} total results")
            elif response is None:
                logger.error("Semantic Scholar search returned None response")
                return APIResponse(
                    data=None,
                    status_code=500,
                    headers={},
                    url="",
                    success=False,
                    error_message="Search returned None response"
                )

            return response

        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return APIResponse(
                data=None,
                status_code=500,
                headers={},
                url="",
                success=False,
                error_message=str(e)
            )

    async def get_paper_details(self, paper_id: str, fields: Optional[List[str]] = None) -> APIResponse:
        """
        Get detailed information for a specific paper.

        Args:
            paper_id: Paper ID (can be Semantic Scholar ID, DOI, or other identifier)
            fields: List of fields to retrieve

        Returns:
            APIResponse with paper details
        """
        logger.debug(f"Fetching Semantic Scholar details for paper: {paper_id}")

        try:
            params = self._build_paper_params(fields)
            response = await self.get(f'paper/{paper_id}', params=params)
            return response
        except Exception as e:
            logger.error(f"Error fetching paper details for {paper_id}: {e}")
            return APIResponse(
                data=None,
                status_code=500,
                headers={},
                url="",
                success=False,
                error_message=str(e)
            )

    async def get_papers_batch(self, paper_ids: List[str], fields: Optional[List[str]] = None) -> APIResponse:
        """
        Get details for multiple papers in batch.

        Args:
            paper_ids: List of paper IDs
            fields: List of fields to retrieve

        Returns:
            APIResponse with batch paper details
        """
        if not paper_ids:
            return APIResponse(
                data=[],
                status_code=200,
                headers={},
                url="",
                success=True
            )

        logger.info(f"Fetching batch details for {len(paper_ids)} papers")

        try:
            # Semantic Scholar batch endpoint
            params = self._build_paper_params(fields)
            params['paperIds'] = ','.join(paper_ids)

            response = await self.post('paper/batch', data=params)
            return response
        except Exception as e:
            logger.error(f"Error fetching batch paper details: {e}")
            return APIResponse(
                data=None,
                status_code=500,
                headers={},
                url="",
                success=False,
                error_message=str(e)
            )

    def parse_search_results(self, response: APIResponse) -> List[Dict[str, Any]]:
        """
        Parse search results to extract paper data.

        Args:
            response: APIResponse from search_papers

        Returns:
            List of paper dictionaries
        """
        if not response.success or not response.data:
            return []

        try:
            papers = response.data.get('data', [])
            return papers
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return []

    def parse_paper_details(self, response: APIResponse) -> List[Dict[str, Any]]:
        """
        Parse paper details response.

        Args:
            response: APIResponse from get_paper_details or get_papers_batch

        Returns:
            List of paper dictionaries
        """
        if not response.success or not response.data:
            return []

        try:
            # Handle both single paper and batch responses
            if isinstance(response.data, list):
                papers = response.data
            else:
                papers = [response.data]

            # Add extraction timestamp to each paper
            for paper in papers:
                if isinstance(paper, dict):
                    paper['extracted_at'] = datetime.now().isoformat()

            return papers
        except Exception as e:
            logger.error(f"Error parsing paper details: {e}")
            return []

    async def search_papers_async(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """
        Complete async workflow: search for papers and return parsed results.

        Args:
            query: Search query string
            limit: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            List of paper dictionaries
        """
        logger.info(f"Starting complete Semantic Scholar workflow for query: {query}")

        # Search for papers
        search_response = await self.search_papers(query, limit, **kwargs)
        if not search_response.success:
            logger.error(f"Semantic Scholar search failed: {search_response.error_message}")
            return []

        # Parse results
        papers = self.parse_search_results(search_response)

        logger.info(f"Successfully extracted {len(papers)} papers")
        return papers

    async def get_paper_citations(self, paper_id: str, limit: int = 100) -> APIResponse:
        """
        Get papers that cite the specified paper.

        Args:
            paper_id: Paper ID
            limit: Maximum number of citations to return

        Returns:
            APIResponse with citation data
        """
        logger.debug(f"Fetching citations for paper: {paper_id}")

        params = {
            'limit': min(limit, 100),
            'fields': ','.join(['paperId', 'title', 'year', 'authors'])
        }

        response = await self.get(f'paper/{paper_id}/citations', params=params)
        return response

    async def get_paper_references(self, paper_id: str, limit: int = 100) -> APIResponse:
        """
        Get papers that are referenced by the specified paper.

        Args:
            paper_id: Paper ID
            limit: Maximum number of references to return

        Returns:
            APIResponse with reference data
        """
        logger.debug(f"Fetching references for paper: {paper_id}")

        params = {
            'limit': min(limit, 100),
            'fields': ','.join(['paperId', 'title', 'year', 'authors'])
        }

        response = await self.get(f'paper/{paper_id}/references', params=params)
        return response

    async def get_author_papers(self, author_id: str, limit: int = 100) -> APIResponse:
        """
        Get papers by a specific author.

        Args:
            author_id: Author ID
            limit: Maximum number of papers to return

        Returns:
            APIResponse with author's papers
        """
        logger.debug(f"Fetching papers for author: {author_id}")

        params = {
            'limit': min(limit, 100),
            'fields': ','.join(self.default_fields)
        }

        response = await self.get(f'author/{author_id}/papers', params=params)
        return response

    def extract_citation_network(self, paper_id: str, max_depth: int = 1) -> Dict[str, Any]:
        """
        Extract citation network for a paper (placeholder for future implementation).

        Args:
            paper_id: Paper ID
            max_depth: Maximum depth of citation network

        Returns:
            Dictionary with citation network data
        """
        # This would be implemented to build a citation network
        # For now, return a placeholder structure
        return {
            'paper_id': paper_id,
            'max_depth': max_depth,
            'citations': [],
            'references': [],
            'network_data': {}
        }
