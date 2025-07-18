#!/usr/bin/env python3
"""
Europe PMC API Client
Provides comprehensive access to Europe PMC database including PubMed content plus additional resources
Based on research findings: 43.3M citations, 9.3M full-text articles, patent records
"""

import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from urllib.parse import quote

from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class EuropePMCClient(BaseAPIClient):
    """
    Europe PMC API Client
    Provides access to comprehensive biomedical literature including patents and theses
    """

    def __init__(self, api_key: str = None, rate_limiter=None):
        # Create a config dictionary for the base client
        config = {
            'base_url': 'https://www.ebi.ac.uk/europepmc/webservices/rest/',
            'api_key': api_key,
            'rate_limit_per_sec': 10.0,  # Europe PMC allows 10 requests per second
            'max_retries': 3,
            'retry_delay': 1.0,
            'max_retry_delay': 60.0,
            'failure_threshold': 5,
            'recovery_timeout': 60
        }
        super().__init__(config)
        self.rate_limiter = rate_limiter

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_papers(self,
                          query: str,
                          limit: int = 100,
                          date_from: str = None,
                          date_to: str = None,
                          publication_type: str = None) -> List[Dict[str, Any]]:
        """
        Search papers in Europe PMC

        Args:
            query: Search query
            limit: Maximum number of results
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            publication_type: Type of publication (journal, preprint, patent, thesis)
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Build search parameters
        params = {
            'query': query,
            'resultType': 'core',
            'pageSize': min(limit, 1000),  # Max page size
            'format': 'json'
        }

        if date_from:
            params['from'] = date_from
        if date_to:
            params['to'] = date_to
        if publication_type:
            params['publicationType'] = publication_type

        try:
            # Rate limiting
            if self.rate_limiter:
                await self.rate_limiter.acquire_token('europe_pmc')

            url = f"{self.base_url}/search"
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                papers = []
                if 'resultList' in data and 'result' in data['resultList']:
                    for paper in data['resultList']['result']:
                        papers.append(self._parse_paper(paper))

                logger.info(f"Found {len(papers)} papers for query: {query}")
                return papers[:limit]

        except Exception as e:
            logger.error(f"Error searching Europe PMC: {e}")
            return []

    async def get_paper_details(self, pmid: str = None, pmcid: str = None, doi: str = None) -> Optional[Dict[str, Any]]:
        """Get detailed paper information"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Determine identifier type
        if pmid:
            identifier = f"EXT_ID:{pmid}"
        elif pmcid:
            identifier = f"PMCID:{pmcid}"
        elif doi:
            identifier = f"DOI:{doi}"
        else:
            logger.error("Must provide pmid, pmcid, or doi")
            return None

        try:
            if self.rate_limiter:
                await self.rate_limiter.acquire_token('europe_pmc')

            url = f"{self.base_url}/search"
            params = {
                'query': identifier,
                'resultType': 'core',
                'format': 'json'
            }

            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                if 'resultList' in data and 'result' in data['resultList'] and data['resultList']['result']:
                    return self._parse_paper(data['resultList']['result'][0])

                return None

        except Exception as e:
            logger.error(f"Error getting paper details: {e}")
            return None

    async def get_paper_text(self, pmid: str = None, pmcid: str = None) -> Optional[str]:
        """Get full text of paper if available"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Prefer PMCID for full text
        if pmcid:
            identifier = pmcid
        elif pmid:
            identifier = pmid
        else:
            logger.error("Must provide pmid or pmcid")
            return None

        try:
            if self.rate_limiter:
                await self.rate_limiter.acquire_token('europe_pmc')

            url = f"{self.base_url}/search"
            params = {
                'query': f"PMCID:{identifier}" if pmcid else f"EXT_ID:{identifier}",
                'resultType': 'core',
                'format': 'json'
            }

            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                if 'resultList' in data and 'result' in data['resultList'] and data['resultList']['result']:
                    paper = data['resultList']['result'][0]

                    # Try to get full text
                    if 'hasFullText' in paper and paper['hasFullText']:
                        return await self._get_full_text(identifier)

                    # Fallback to abstract
                    return paper.get('abstractText', '')

                return None

        except Exception as e:
            logger.error(f"Error getting paper text: {e}")
            return None

    async def _get_full_text(self, identifier: str) -> Optional[str]:
        """Get full text content"""
        try:
            url = f"{self.base_url}/search"
            params = {
                'query': f"PMCID:{identifier}",
                'resultType': 'fullText',
                'format': 'json'
            }

            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                if 'resultList' in data and 'result' in data['resultList'] and data['resultList']['result']:
                    paper = data['resultList']['result'][0]
                    return paper.get('fullText', '')

                return None

        except Exception as e:
            logger.error(f"Error getting full text: {e}")
            return None

    def _parse_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Parse paper data from Europe PMC response"""
        return {
            'id': paper.get('id'),
            'pmid': paper.get('pmid'),
            'pmcid': paper.get('pmcid'),
            'doi': paper.get('doi'),
            'title': paper.get('title'),
            'authors': [author.get('fullName', '') for author in paper.get('authorList', {}).get('author', [])],
            'journal': paper.get('journalInfo', {}).get('journal', {}).get('title'),
            'year': paper.get('journalInfo', {}).get('pubYear'),
            'abstract': paper.get('abstractText', ''),
            'keywords': paper.get('keywordList', {}).get('keyword', []),
            'publication_type': paper.get('publicationType'),
            'has_full_text': paper.get('hasFullText', False),
            'source': 'europe_pmc',
            'extracted_at': datetime.now().isoformat()
        }

    async def search_patents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search patent literature specifically"""
        return await self.search_papers(
            query=query,
            limit=limit,
            publication_type='patent'
        )

    async def search_preprints(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search preprint literature specifically"""
        return await self.search_papers(
            query=query,
            limit=limit,
            publication_type='preprint'
        )

    async def get_citation_count(self, pmid: str) -> Optional[int]:
        """Get citation count for a paper"""
        try:
            if self.rate_limiter:
                await self.rate_limiter.acquire_token('europe_pmc')

            url = f"{self.base_url}/citations/{pmid}"
            params = {'format': 'json'}

            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                return data.get('citationCount', 0)

        except Exception as e:
            logger.error(f"Error getting citation count: {e}")
            return None
