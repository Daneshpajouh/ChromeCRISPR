"""
Enhanced PubMed API Client for GeneX Phase 1
Provides comprehensive access to NCBI PubMed database via E-utilities API.
Enhanced with advanced features for bulk operations, async support, and analytics.
"""

import logging
import re
import hashlib
import asyncio
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote_plus
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)

@dataclass
class PubMedPaper:
    """Enhanced PubMed paper data structure with comprehensive metadata"""
    # Basic identification
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: Optional[int]
    doi: Optional[str]

    # Enhanced metadata
    publication_date: Optional[str]
    publication_type: List[str]
    mesh_terms: List[str]
    keywords: List[str]
    language: str

    # Citation and impact
    citation_count: int
    reference_count: int

    # Source and processing info
    source: str = "PubMed"
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

class PubMedClient(BaseAPIClient):
    """
    PubMed E-utilities client for literature mining.

    Implements NCBI E-utilities API for searching and retrieving PubMed articles.
    Uses two-step process: ESearch for PMIDs, then EFetch for full records.
    """

    def __init__(self, config: Dict[str, Any], session=None):
        """Initialize PubMed client with E-utilities configuration"""
        super().__init__(config, session)

        # E-utilities specific configuration
        self.email = config.get('email')
        self.tool = config.get('tool', 'GeneX-Miner/1.0')

        # Ensure we have required parameters
        if not self.email:
            logger.warning("PubMed client initialized without email - may hit rate limits")

    def _build_search_params(self, query: str, max_results: int = 100, **kwargs) -> Dict[str, Any]:
        """Build parameters for ESearch request"""
        params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': min(max_results, 100000),  # NCBI limit
            'usehistory': 'y',  # Use history for large result sets
            'tool': self.tool
        }

        if self.email:
            params['email'] = self.email

        if self.api_key:
            params['api_key'] = self.api_key

        # Add any additional parameters
        params.update(kwargs)

        return params

    def _build_fetch_params(self, pmid_list: List[str], **kwargs) -> Dict[str, Any]:
        """Build parameters for EFetch request"""
        params = {
            'db': 'pubmed',
            'id': ','.join(pmid_list),
            'retmode': 'xml',
            'rettype': 'abstract',
            'tool': self.tool
        }

        if self.email:
            params['email'] = self.email

        if self.api_key:
            params['api_key'] = self.api_key

        # Add any additional parameters
        params.update(kwargs)

        return params

    async def search_papers(self, query: str, max_results: int = 100, **kwargs) -> APIResponse:
        """
        Search PubMed for papers using ESearch.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional ESearch parameters

        Returns:
            APIResponse with search results
        """
        logger.info(f"Searching PubMed for: {query} (max_results: {max_results})")

        params = self._build_search_params(query, max_results, **kwargs)
        response = await self.get('esearch.fcgi', params=params)

        if response.success and response.data:
            logger.info(f"PubMed search returned {len(response.data.get('esearchresult', {}).get('idlist', []))} results")

        return response

    async def get_paper_details(self, pmid: str) -> APIResponse:
        """
        Get detailed paper information using EFetch.

        Args:
            pmid: PubMed ID

        Returns:
            APIResponse with paper details
        """
        logger.debug(f"Fetching PubMed details for PMID: {pmid}")

        params = self._build_fetch_params([pmid])
        response = await self.get('efetch.fcgi', params=params)

        return response

    async def get_papers_batch(self, pmid_list: List[str]) -> APIResponse:
        """
        Get detailed information for multiple papers in batch.

        Args:
            pmid_list: List of PubMed IDs

        Returns:
            APIResponse with batch paper details
        """
        if not pmid_list:
            return APIResponse(
                data=[],
                status_code=200,
                headers={},
                url="",
                success=True
            )

        logger.info(f"Fetching batch details for {len(pmid_list)} PMIDs")

        # Split into chunks to avoid URL length limits
        chunk_size = 50  # Conservative chunk size
        all_results = []

        for i in range(0, len(pmid_list), chunk_size):
            chunk = pmid_list[i:i + chunk_size]
            params = self._build_fetch_params(chunk)
            response = await self.get('efetch.fcgi', params=params)

            if response.success and response.data:
                all_results.append(response.data)
            else:
                logger.error(f"Failed to fetch chunk {i//chunk_size + 1}: {response.error_message}")

        # Combine results
        combined_response = APIResponse(
            data=all_results,
            status_code=200,
            headers={},
            url="",
            success=len(all_results) > 0
        )

        return combined_response

    def parse_search_results(self, response: APIResponse) -> List[str]:
        """
        Parse ESearch results to extract PMID list.

        Args:
            response: APIResponse from search_papers

        Returns:
            List of PMIDs
        """
        if not response.success or not response.data:
            return []

        try:
            esearch_result = response.data.get('esearchresult', {})
            idlist = esearch_result.get('idlist', [])
            return idlist
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return []

    def parse_paper_details(self, response: APIResponse) -> List[Dict[str, Any]]:
        """
        Parse EFetch results to extract paper details.

        Args:
            response: APIResponse from get_paper_details or get_papers_batch

        Returns:
            List of paper dictionaries
        """
        if not response.success or not response.data:
            return []

        papers = []

        try:
            # Handle both single paper and batch responses
            if isinstance(response.data, list):
                xml_data_list = response.data
            else:
                xml_data_list = [response.data]

            for xml_data in xml_data_list:
                if isinstance(xml_data, str):
                    # Parse XML string
                    root = ET.fromstring(xml_data)
                else:
                    # Already parsed XML
                    root = xml_data

                # Extract papers from PubmedArticle elements
                for article in root.findall('.//PubmedArticle'):
                    paper = self._extract_paper_data(article)
                    if paper:
                        papers.append(paper)

        except Exception as e:
            logger.error(f"Error parsing paper details: {e}")

        return papers

    def _extract_paper_data(self, article_element) -> Optional[Dict[str, Any]]:
        """
        Extract paper data from a PubmedArticle XML element.

        Args:
            article_element: XML element representing a PubMed article

        Returns:
            Dictionary with paper data or None if extraction fails
        """
        try:
            paper = {}

            # Extract PMID
            pmid_elem = article_element.find('.//PMID')
            if pmid_elem is not None:
                paper['pmid'] = pmid_elem.text

            # Extract title
            title_elem = article_element.find('.//ArticleTitle')
            if title_elem is not None:
                paper['title'] = title_elem.text

            # Extract abstract
            abstract_elem = article_element.find('.//AbstractText')
            if abstract_elem is not None:
                paper['abstract'] = abstract_elem.text

            # Extract journal information
            journal_elem = article_element.find('.//Journal')
            if journal_elem is not None:
                journal_title_elem = journal_elem.find('.//Title')
                if journal_title_elem is not None:
                    paper['journal'] = journal_title_elem.text

                # Extract publication date
                pub_date_elem = journal_elem.find('.//PubDate')
                if pub_date_elem is not None:
                    year_elem = pub_date_elem.find('Year')
                    if year_elem is not None:
                        paper['year'] = int(year_elem.text)

            # Extract authors
            authors = []
            author_list = article_element.find('.//AuthorList')
            if author_list is not None:
                for author_elem in author_list.findall('Author'):
                    last_name_elem = author_elem.find('LastName')
                    first_name_elem = author_elem.find('ForeName')

                    if last_name_elem is not None:
                        author = {'last_name': last_name_elem.text}
                        if first_name_elem is not None:
                            author['first_name'] = first_name_elem.text
                        authors.append(author)

            if authors:
                paper['authors'] = authors

            # Extract DOI
            article_id_list = article_element.find('.//ArticleIdList')
            if article_id_list is not None:
                for article_id in article_id_list.findall('ArticleId'):
                    if article_id.get('IdType') == 'doi':
                        paper['doi'] = article_id.text
                        break

            # Extract MeSH terms
            mesh_terms = []
            mesh_heading_list = article_element.find('.//MeshHeadingList')
            if mesh_heading_list is not None:
                for mesh_heading in mesh_heading_list.findall('MeshHeading'):
                    descriptor_elem = mesh_heading.find('DescriptorName')
                    if descriptor_elem is not None:
                        mesh_terms.append(descriptor_elem.text)

            if mesh_terms:
                paper['mesh_terms'] = mesh_terms

            # Add extraction timestamp
            paper['extracted_at'] = datetime.now().isoformat()

            return paper if paper.get('pmid') else None

        except Exception as e:
            logger.error(f"Error extracting paper data: {e}")
            return None

    async def search_papers_async(self, query: str, max_results: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """
        Complete async workflow: search for papers and fetch their details.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of paper dictionaries with full details
        """
        logger.info(f"Starting complete PubMed workflow for query: {query}")

        # Step 1: Search for PMIDs
        search_response = await self.search_papers(query, max_results, **kwargs)
        if not search_response.success:
            logger.error(f"PubMed search failed: {search_response.error_message}")
            return []

        # Step 2: Extract PMIDs
        pmid_list = self.parse_search_results(search_response)
        if not pmid_list:
            logger.warning("No PMIDs found in search results")
            return []

        logger.info(f"Found {len(pmid_list)} PMIDs, fetching details...")

        # Step 3: Fetch paper details
        details_response = await self.get_papers_batch(pmid_list)
        if not details_response.success:
            logger.error(f"PubMed details fetch failed: {details_response.error_message}")
            return []

        # Step 4: Parse paper details
        papers = self.parse_paper_details(details_response)

        logger.info(f"Successfully extracted {len(papers)} papers")
        return papers

    def get_by_id(self, identifier: str, **kwargs) -> APIResponse:
        """Get paper by PMID"""
        return self.get_paper_details(identifier, **kwargs)

    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive search analytics"""
        if not self.enable_analytics:
            return {'analytics_disabled': True}

        return {
            'search_analytics': self.search_analytics,
            'client_metrics': self.get_metrics(),
            'rate_limit_info': self.get_rate_limit_info()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get enhanced client metrics"""
        base_metrics = super().get_metrics()
        base_metrics['pubmed_specific'] = {
            'max_batch_size': self.max_batch_size,
            'enable_analytics': self.enable_analytics,
            'extract_gene_editing_features': self.extract_gene_editing_features
        }
        return base_metrics

    def get_cache_size(self) -> int:
        """Return the number of items in the cache."""
        if hasattr(self, 'cache_manager'):
            return self.cache_manager.get_cache_size() if hasattr(self.cache_manager, 'get_cache_size') else 0
        return 0

    def clear_cache(self):
        """Clear the PubMed client cache directory."""
        import shutil
        shutil.rmtree(self.cache.cache_dir, ignore_errors=True)
        self.cache.cache_dir.mkdir(exist_ok=True)
