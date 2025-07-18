"""
CrossRef API Client
Real-time integration with CrossRef for literature mining
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


class CrossRefClient(BaseAPIClient):
    """Client for CrossRef API with real data integration."""

    def __init__(self, config: Dict[str, Any], session=None):
        """Initialize CrossRef client with configuration and optional session"""
        super().__init__(config, session)

        # CrossRef specific configuration
        self.email = config.get('email', '')
        self.user_agent = config.get('user_agent', 'GeneX-Miner/1.0')

        logger.info(f"Initialized CrossRefClient with base_url: {self.base_url}")

    async def search_papers_async(self, query: str, max_results: int = 100,
                                 year_from: Optional[int] = None,
                                 year_to: Optional[int] = None,
                                 type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Async version of search_papers"""
        try:
            params = {
                'query': query,
                'rows': min(max_results, 1000),  # CrossRef limit
                'offset': 0,
                'select': 'DOI,title,author,published-print,container-title,abstract,type,subject'
            }

            if year_from:
                params['from-pub-date'] = str(year_from)
            if year_to:
                params['until-pub-date'] = str(year_to)
            if type_filter:
                params['filter'] = f'type:{type_filter}'

            logger.info(f"Searching CrossRef for: {query}")
            response = await self.get_async('works', params=params)

            if not response.success or 'message' not in response.data or 'items' not in response.data['message']:
                logger.warning("No search results found")
                return []

            papers = response.data['message']['items']
            logger.info(f"Found {len(papers)} papers from CrossRef")

            # Process and standardize paper data
            processed_papers = []
            for paper in papers:
                try:
                    processed_paper = self._process_paper_data(paper)
                    if processed_paper:
                        processed_papers.append(processed_paper)
                except Exception as e:
                    logger.error(f"Error processing paper {paper.get('DOI', 'unknown')}: {str(e)}")
                    continue

            return processed_papers

        except Exception as e:
            logger.error(f"Error searching CrossRef: {str(e)}")
            return []

    def search_papers(self, query: str, max_results: int = 100,
                     year_from: Optional[int] = None,
                     year_to: Optional[int] = None,
                     type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search CrossRef for papers matching the query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            year_from: Start year filter
            year_to: End year filter
            type_filter: Filter by publication type (journal-article, conference-paper, etc.)

        Returns:
            List of paper metadata dictionaries
        """
        try:
            params = {
                'query': query,
                'rows': min(max_results, 1000),  # CrossRef limit
                'offset': 0,
                'select': 'DOI,title,author,published-print,container-title,abstract,type,subject'
            }

            if year_from:
                params['from-pub-date'] = str(year_from)
            if year_to:
                params['until-pub-date'] = str(year_to)
            if type_filter:
                params['filter'] = f'type:{type_filter}'

            self.logger.info(f"Searching CrossRef for: {query}")
            response = self.get('works', params=params)

            if 'message' not in response or 'items' not in response['message']:
                self.logger.warning("No search results found")
                return []

            papers = response['message']['items']
            self.logger.info(f"Found {len(papers)} papers from CrossRef")

            # Process and standardize paper data
            processed_papers = []
            for paper in papers:
                try:
                    processed_paper = self._process_paper_data(paper)
                    if processed_paper:
                        processed_papers.append(processed_paper)
                except Exception as e:
                    self.logger.error(f"Error processing paper {paper.get('DOI', 'unknown')}: {str(e)}")
                    continue

            return processed_papers

        except Exception as e:
            self.logger.error(f"Error searching CrossRef: {str(e)}")
            return []

    async def get_paper_details_async(self, doi: str) -> Optional[Dict[str, Any]]:
        """Async version of get_paper_details"""
        try:
            response = await self.get_async(f'works/{doi}')

            if not response.success or 'message' not in response.data:
                return None

            return self._process_paper_data(response.data['message'])

        except Exception as e:
            logger.error(f"Error fetching paper details for DOI {doi}: {str(e)}")
            return None

    def get_paper_details(self, doi: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific paper by DOI."""
        try:
            response = self.get(f'works/{doi}')

            if 'message' not in response:
                return None

            return self._process_paper_data(response['message'])

        except Exception as e:
            self.logger.error(f"Error fetching paper details for DOI {doi}: {str(e)}")
            return None

    async def search_by_doi_list_async(self, doi_list: List[str]) -> List[Dict[str, Any]]:
        """Async version of search_by_doi_list"""
        papers = []

        for doi in doi_list:
            try:
                paper = await self.get_paper_details_async(doi)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.error(f"Error fetching paper with DOI {doi}: {str(e)}")
                continue

        return papers

    def search_by_doi_list(self, doi_list: List[str]) -> List[Dict[str, Any]]:
        """Search for multiple papers by DOI list."""
        papers = []

        for doi in doi_list:
            try:
                paper = self.get_paper_details(doi)
                if paper:
                    papers.append(paper)
            except Exception as e:
                self.logger.error(f"Error fetching paper with DOI {doi}: {str(e)}")
                continue

        return papers

    async def search_by_journal_async(self, journal_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Async version of search_by_journal"""
        query = f'container-title:"{journal_name}"'
        return await self.search_papers_async(query, max_results, type_filter='journal-article')

    def search_by_journal(self, journal_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for papers from a specific journal."""
        query = f'container-title:"{journal_name}"'
        return self.search_papers(query, max_results, type_filter='journal-article')

    async def search_by_author_async(self, author_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Async version of search_by_author"""
        query = f'author:"{author_name}"'
        return await self.search_papers_async(query, max_results)

    def search_by_author(self, author_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for papers by a specific author."""
        query = f'author:"{author_name}"'
        return self.search_papers(query, max_results)

    def _process_paper_data(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and standardize CrossRef paper data."""
        try:
            # Extract DOI
            doi = paper.get('DOI', '')

            # Extract title
            title = ""
            if 'title' in paper and paper['title']:
                title_list = paper['title']
                if isinstance(title_list, list) and len(title_list) > 0:
                    title = title_list[0]
                elif isinstance(title_list, str):
                    title = title_list

            # Extract authors
            authors = []
            if 'author' in paper and paper['author']:
                for author in paper['author']:
                    if isinstance(author, dict):
                        given = author.get('given', '')
                        family = author.get('family', '')
                        if given and family:
                            authors.append(f"{given} {family}")
                        elif family:
                            authors.append(family)
                        elif given:
                            authors.append(given)

            # Extract journal/container title
            journal = ""
            if 'container-title' in paper and paper['container-title']:
                container_list = paper['container-title']
                if isinstance(container_list, list) and len(container_list) > 0:
                    journal = container_list[0]
                elif isinstance(container_list, str):
                    journal = container_list

            # Extract publication date
            publication_date = ""
            if 'published-print' in paper and paper['published-print']:
                date_parts = paper['published-print'].get('date-parts', [[]])
                if date_parts and len(date_parts[0]) > 0:
                    publication_date = '-'.join(str(part) for part in date_parts[0])

            # Extract abstract
            abstract = ""
            if 'abstract' in paper:
                abstract = paper['abstract']

            # Extract subjects/keywords
            subjects = []
            if 'subject' in paper and paper['subject']:
                subjects = paper['subject']

            # Extract publication type
            pub_type = ""
            if 'type' in paper:
                pub_type = paper['type']

            processed_paper = {
                'doi': doi,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'publication_date': publication_date,
                'subjects': subjects,
                'publication_type': pub_type,
                'source': 'CrossRef',
                'url': f"https://doi.org/{doi}" if doi else ""
            }

            return processed_paper

        except Exception as e:
            self.logger.error(f"Error processing CrossRef paper data: {str(e)}")
            return None

    async def search_by_gene_async(self, gene_symbol: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Async version of search_by_gene"""
        query = f'"{gene_symbol}"[gene]'
        return await self.search_papers_async(query, max_results)

    def search_by_gene(self, gene_symbol: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for papers related to a specific gene."""
        query = f'"{gene_symbol}"[gene]'
        return self.search_papers(query, max_results)

    async def search_by_disease_async(self, disease_name: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Async version of search_by_disease"""
        query = f'"{disease_name}"[disease]'
        return await self.search_papers_async(query, max_results)

    def search_by_disease(self, disease_name: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for papers related to a specific disease."""
        query = f'"{disease_name}"[disease]'
        return self.search_papers(query, max_results)

    async def get_journal_metadata_async(self, issn: str) -> Optional[Dict[str, Any]]:
        """Async version of get_journal_metadata"""
        try:
            response = await self.get_async(f'journals/{issn}')
            if response.success and 'message' in response.data:
                return response.data['message']
            return None
        except Exception as e:
            logger.error(f"Error fetching journal metadata for ISSN {issn}: {str(e)}")
            return None

    def get_journal_metadata(self, issn: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific journal by ISSN."""
        try:
            response = self.get(f'journals/{issn}')
            if 'message' in response:
                return response['message']
            return None
        except Exception as e:
            self.logger.error(f"Error fetching journal metadata for ISSN {issn}: {str(e)}")
            return None

    async def clear_cache_async(self):
        """Clear async cache"""
        await self._clear_cache_async()

    def clear_cache(self):
        """Clear sync cache"""
        self._clear_cache()
