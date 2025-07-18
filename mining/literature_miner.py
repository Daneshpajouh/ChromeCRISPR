"""
Literature Miner
Coordinates searches across multiple literature API sources
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from ..api_clients.base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class LiteratureMiner:
    """Coordinates literature mining across multiple API sources"""

    def __init__(self,
                 api_clients: Dict[str, BaseAPIClient],
                 validator=None,
                 quality_controller=None):

        self.api_clients = api_clients
        self.validator = validator
        self.quality_controller = quality_controller

        # Track processed papers to avoid duplicates
        self.processed_papers: Set[str] = set()

        # Literature-specific clients
        self.literature_clients = {
            name: client for name, client in api_clients.items()
            if name in ['pubmed', 'semantic_scholar', 'crossref', 'biorxiv', 'medrxiv']
        }

        logger.info(f"Initialized literature miner with {len(self.literature_clients)} sources")

    async def search_all_sources(self,
                                query: str,
                                max_results: int = 100,
                                sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search across all literature sources

        Args:
            query: Search query
            max_results: Maximum results per source
            sources: Specific sources to search (uses all if None)

        Returns:
            List of papers from all sources
        """

        if sources is None:
            sources = list(self.literature_clients.keys())

        # Create search tasks for each source
        tasks = []
        for source_name in sources:
            if source_name in self.literature_clients:
                client = self.literature_clients[source_name]
                task = asyncio.create_task(
                    self._search_single_source(client, source_name, query, max_results)
                )
                tasks.append(task)

        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine and deduplicate results
        all_papers = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search failed: {str(result)}")
                continue

            if isinstance(result, list):
                all_papers.extend(result)

        # Deduplicate papers
        deduplicated_papers = self._deduplicate_papers(all_papers)

        logger.info(f"Found {len(all_papers)} papers, {len(deduplicated_papers)} after deduplication")
        return deduplicated_papers

    async def _search_single_source(self,
                                   client: BaseAPIClient,
                                   source_name: str,
                                   query: str,
                                   max_results: int) -> List[Dict[str, Any]]:
        """Search a single literature source"""

        try:
            logger.debug(f"Searching {source_name} for: {query}")

            # Search for papers
            papers = await client.search_papers(query, max_results=max_results)

            # Add source information
            for paper in papers:
                paper['source_api'] = source_name
                paper['search_query'] = query
                paper['mined_at'] = datetime.now().isoformat()

            logger.info(f"{source_name}: Found {len(papers)} papers for '{query}'")
            return papers

        except Exception as e:
            logger.error(f"Error searching {source_name} for '{query}': {str(e)}")
            return []

    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on various identifiers"""

        seen_ids = set()
        deduplicated = []

        for paper in papers:
            # Try to find a unique identifier
            paper_id = self._get_paper_identifier(paper)

            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                deduplicated.append(paper)
            elif not paper_id:
                # If no clear identifier, use title as fallback
                title = paper.get('title', '').lower().strip()
                if title and title not in seen_ids:
                    seen_ids.add(title)
                    deduplicated.append(paper)

        return deduplicated

    def _get_paper_identifier(self, paper: Dict[str, Any]) -> Optional[str]:
        """Extract the best available identifier for a paper"""

        # Try different identifier types in order of preference
        identifiers = [
            paper.get('doi'),
            paper.get('pmid'),
            paper.get('paper_id'),
            paper.get('id')
        ]

        for identifier in identifiers:
            if identifier and isinstance(identifier, str) and identifier.strip():
                return identifier.strip()

        return None

    async def get_paper_details(self,
                               paper_id: str,
                               source: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific paper"""

        if source not in self.literature_clients:
            logger.error(f"Unknown source: {source}")
            return None

        try:
            client = self.literature_clients[source]
            details = await client.get_paper_details(paper_id)

            if details:
                details['source_api'] = source
                details['mined_at'] = datetime.now().isoformat()

            return details

        except Exception as e:
            logger.error(f"Error getting paper details from {source}: {str(e)}")
            return None

    async def get_related_papers(self,
                                paper_id: str,
                                source: str,
                                max_results: int = 50) -> List[Dict[str, Any]]:
        """Get papers related to a specific paper"""

        if source not in self.literature_clients:
            logger.error(f"Unknown source: {source}")
            return []

        try:
            client = self.literature_clients[source]
            related = await client.get_related_papers(paper_id, max_results)

            # Add source information
            for paper in related:
                paper['source_api'] = source
                paper['related_to'] = paper_id
                paper['mined_at'] = datetime.now().isoformat()

            return related

        except Exception as e:
            logger.error(f"Error getting related papers from {source}: {str(e)}")
            return []

    async def search_by_date_range(self,
                                  query: str,
                                  start_date: str,
                                  end_date: str,
                                  max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for papers within a specific date range"""

        # This is primarily supported by PubMed
        if 'pubmed' in self.literature_clients:
            try:
                client = self.literature_clients['pubmed']
                papers = await client.search_papers(
                    query=query,
                    max_results=max_results,
                    date_from=start_date,
                    date_to=end_date
                )

                # Add source information
                for paper in papers:
                    paper['source_api'] = 'pubmed'
                    paper['search_query'] = query
                    paper['date_range'] = f"{start_date} to {end_date}"
                    paper['mined_at'] = datetime.now().isoformat()

                return papers

            except Exception as e:
                logger.error(f"Error searching PubMed by date range: {str(e)}")

        # Fallback to regular search
        return await self.search_all_sources(query, max_results)

    async def search_by_year(self,
                            query: str,
                            year: int,
                            max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for papers from a specific year"""

        # This is supported by Semantic Scholar
        if 'semantic_scholar' in self.literature_clients:
            try:
                client = self.literature_clients['semantic_scholar']
                papers = await client.search_papers(
                    query=query,
                    max_results=max_results,
                    year=year
                )

                # Add source information
                for paper in papers:
                    paper['source_api'] = 'semantic_scholar'
                    paper['search_query'] = query
                    paper['search_year'] = year
                    paper['mined_at'] = datetime.now().isoformat()

                return papers

            except Exception as e:
                logger.error(f"Error searching Semantic Scholar by year: {str(e)}")

        # Fallback to regular search
        return await self.search_all_sources(query, max_results)

    def get_available_sources(self) -> List[str]:
        """Get list of available literature sources"""
        return list(self.literature_clients.keys())

    def get_source_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for each literature source"""
        metrics = {}

        for source_name, client in self.literature_clients.items():
            try:
                client_metrics = client.get_metrics()
                metrics[source_name] = client_metrics
            except Exception as e:
                logger.error(f"Error getting metrics for {source_name}: {str(e)}")
                metrics[source_name] = {'error': str(e)}

        return metrics
