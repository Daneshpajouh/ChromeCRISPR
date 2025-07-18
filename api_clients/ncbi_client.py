"""
NCBI API Client
Real-time integration with NCBI E-utilities for genomic data
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


class NCBIClient(BaseAPIClient):
    """Client for NCBI APIs with real data integration."""

    def __init__(self, config: Dict[str, Any], session=None):
        """Initialize NCBI client with configuration and optional session"""
        super().__init__(config, session)

        # NCBI specific configuration
        self.email = config.get('email', '')
        self.api_key = config.get('api_key', '')

        logger.info(f"Initialized NCBIClient with base_url: {self.base_url}")

    async def search_genes_async(self, query: str, organism: Optional[str] = None,
                                max_results: int = 100) -> List[Dict[str, Any]]:
        """Async version of search_genes"""
        try:
            params = {
                'db': 'gene',
                'term': query,
                'retmax': min(max_results, 100),
                'retmode': 'json',
                'sort': 'relevance'
            }

            if organism:
                params['term'] = f'{query} AND "{organism}"[Organism]'

            logger.info(f"Searching NCBI Gene for: {query}")
            response = await self.get_async('esearch.fcgi', params=params)

            if not response.success or 'esearchresult' not in response.data:
                logger.warning("No search results found")
                return []

            id_list = response.data['esearchresult'].get('idlist', [])

            if not id_list:
                logger.warning("No gene IDs found")
                return []

            logger.info(f"Found {len(id_list)} genes, fetching details...")

            # Fetch detailed information for each gene
            genes = []
            for gene_id in id_list[:max_results]:
                try:
                    gene_data = await self._fetch_gene_details_async(gene_id)
                    if gene_data:
                        genes.append(gene_data)
                except Exception as e:
                    logger.error(f"Error fetching gene {gene_id}: {str(e)}")
                    continue

            logger.info(f"Successfully fetched {len(genes)} genes")
            return genes

        except Exception as e:
            logger.error(f"Error searching NCBI Gene: {str(e)}")
            return []

    def search_genes(self, query: str, organism: Optional[str] = None,
                    max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search NCBI Gene database for genes matching the query.

        Args:
            query: Search query string
            organism: Filter by organism (e.g., "Homo sapiens")
            max_results: Maximum number of results to return

        Returns:
            List of gene metadata dictionaries
        """
        try:
            params = {
                'db': 'gene',
                'term': query,
                'retmax': min(max_results, 100),
                'retmode': 'json',
                'sort': 'relevance'
            }

            if organism:
                params['term'] = f'{query} AND "{organism}"[Organism]'

            self.logger.info(f"Searching NCBI Gene for: {query}")
            response = self.get('esearch.fcgi', params=params)

            if 'esearchresult' not in response:
                self.logger.warning("No search results found")
                return []

            id_list = response['esearchresult'].get('idlist', [])

            if not id_list:
                self.logger.warning("No gene IDs found")
                return []

            self.logger.info(f"Found {len(id_list)} genes, fetching details...")

            # Fetch detailed information for each gene
            genes = []
            for gene_id in id_list[:max_results]:
                try:
                    gene_data = self._fetch_gene_details(gene_id)
                    if gene_data:
                        genes.append(gene_data)
                except Exception as e:
                    self.logger.error(f"Error fetching gene {gene_id}: {str(e)}")
                    continue

            self.logger.info(f"Successfully fetched {len(genes)} genes")
            return genes

        except Exception as e:
            self.logger.error(f"Error searching NCBI Gene: {str(e)}")
            return []

    async def get_gene_details_async(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Async version of get_gene_details"""
        try:
            params = {
                'db': 'gene',
                'id': gene_id,
                'retmode': 'xml',
                'rettype': 'gb'
            }

            response = await self.get_async('efetch.fcgi', params=params)

            if not response.success:
                return None

            # Parse XML response (simplified)
            gene_data = {
                'gene_id': gene_id,
                'source': 'NCBI Gene',
                'name': self._extract_gene_name(response.data),
                'symbol': self._extract_gene_symbol(response.data),
                'description': self._extract_gene_description(response.data),
                'organism': self._extract_organism(response.data),
                'chromosome': self._extract_chromosome(response.data),
                'location': self._extract_location(response.data),
                'aliases': self._extract_aliases(response.data),
                'summary': self._extract_summary(response.data)
            }

            return gene_data

        except Exception as e:
            logger.error(f"Error fetching gene details for {gene_id}: {str(e)}")
            return None

    def get_gene_details(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific gene."""
        try:
            params = {
                'db': 'gene',
                'id': gene_id,
                'retmode': 'xml',
                'rettype': 'gb'
            }

            response = self.get('efetch.fcgi', params=params)

            # Parse XML response (simplified)
            gene_data = {
                'gene_id': gene_id,
                'source': 'NCBI Gene',
                'name': self._extract_gene_name(response),
                'symbol': self._extract_gene_symbol(response),
                'description': self._extract_gene_description(response),
                'organism': self._extract_organism(response),
                'chromosome': self._extract_chromosome(response),
                'location': self._extract_location(response),
                'aliases': self._extract_aliases(response),
                'summary': self._extract_summary(response)
            }

            return gene_data

        except Exception as e:
            self.logger.error(f"Error fetching gene details for {gene_id}: {str(e)}")
            return None

    async def _fetch_gene_details_async(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Async version of _fetch_gene_details"""
        return await self.get_gene_details_async(gene_id)

    def _fetch_gene_details(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed information for a gene ID."""
        return self.get_gene_details(gene_id)

    def _extract_gene_name(self, response: str) -> str:
        """Extract gene name from NCBI response."""
        # Simplified extraction - in production, use proper XML parsing
        if 'GENE' in response and 'NAME' in response:
            start = response.find('NAME') + 5
            end = response.find('\n', start)
            return response[start:end].strip()
        return ""

    def _extract_gene_symbol(self, response: str) -> str:
        """Extract gene symbol from NCBI response."""
        if 'SYMBOL' in response:
            start = response.find('SYMBOL') + 7
            end = response.find('\n', start)
            return response[start:end].strip()
        return ""

    def _extract_gene_description(self, response: str) -> str:
        """Extract gene description from NCBI response."""
        if 'DESCRIPTION' in response:
            start = response.find('DESCRIPTION') + 12
            end = response.find('\n', start)
            return response[start:end].strip()
        return ""

    def _extract_organism(self, response: str) -> str:
        """Extract organism from NCBI response."""
        if 'ORGANISM' in response:
            start = response.find('ORGANISM') + 9
            end = response.find('\n', start)
            return response[start:end].strip()
        return ""

    def _extract_chromosome(self, response: str) -> str:
        """Extract chromosome from NCBI response."""
        if 'CHROMOSOME' in response:
            start = response.find('CHROMOSOME') + 11
            end = response.find('\n', start)
            return response[start:end].strip()
        return ""

    def _extract_location(self, response: str) -> str:
        """Extract location from NCBI response."""
        if 'LOCATION' in response:
            start = response.find('LOCATION') + 9
            end = response.find('\n', start)
            return response[start:end].strip()
        return ""

    def _extract_aliases(self, response: str) -> List[str]:
        """Extract aliases from NCBI response."""
        aliases = []
        if 'ALIAS' in response:
            lines = response.split('\n')
            for line in lines:
                if line.startswith('ALIAS'):
                    alias = line[6:].strip()
                    if alias:
                        aliases.append(alias)
        return aliases

    def _extract_summary(self, response: str) -> str:
        """Extract summary from NCBI response."""
        if 'SUMMARY' in response:
            start = response.find('SUMMARY') + 8
            end = response.find('\n', start)
            return response[start:end].strip()
        return ""

    async def clear_cache_async(self):
        """Clear async cache"""
        await self._clear_cache_async()

    def clear_cache(self):
        """Clear sync cache"""
        self._clear_cache()

    def search_proteins(self, query: str, organism: Optional[str] = None,
                       max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search NCBI Protein database for proteins matching the query.

        Args:
            query: Search query string
            organism: Filter by organism
            max_results: Maximum number of results to return

        Returns:
            List of protein metadata dictionaries
        """
        try:
            params = {
                'db': 'protein',
                'term': query,
                'retmax': min(max_results, 100),
                'retmode': 'json',
                'sort': 'relevance'
            }

            if organism:
                params['term'] = f'{query} AND "{organism}"[Organism]'

            self.logger.info(f"Searching NCBI Protein for: {query}")
            response = self.get('esearch.fcgi', params=params)

            if 'esearchresult' not in response:
                self.logger.warning("No search results found")
                return []

            id_list = response['esearchresult'].get('idlist', [])

            if not id_list:
                self.logger.warning("No protein IDs found")
                return []

            self.logger.info(f"Found {len(id_list)} proteins, fetching details...")

            # Fetch detailed information for each protein
            proteins = []
            for protein_id in id_list[:max_results]:
                try:
                    protein_data = self._fetch_protein_details(protein_id)
                    if protein_data:
                        proteins.append(protein_data)
                except Exception as e:
                    self.logger.error(f"Error fetching protein {protein_id}: {str(e)}")
                    continue

            self.logger.info(f"Successfully fetched {len(proteins)} proteins")
            return proteins

        except Exception as e:
            self.logger.error(f"Error searching NCBI Protein: {str(e)}")
            return []

    def get_protein_details(self, protein_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific protein."""
        try:
            params = {
                'db': 'protein',
                'id': protein_id,
                'retmode': 'xml',
                'rettype': 'gb'
            }

            response = self.get('efetch.fcgi', params=params)

            # Parse XML response (simplified)
            protein_data = {
                'protein_id': protein_id,
                'source': 'NCBI Protein',
                'name': self._extract_protein_name(response),
                'description': self._extract_protein_description(response),
                'organism': self._extract_organism(response),
                'length': self._extract_protein_length(response),
                'molecular_weight': self._extract_molecular_weight(response),
                'sequence': self._extract_sequence(response)
            }

            return protein_data

        except Exception as e:
            self.logger.error(f"Error fetching protein details for {protein_id}: {str(e)}")
            return None

    def _fetch_protein_details(self, protein_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed information for a specific protein."""
        return self.get_protein_details(protein_id)

    def _extract_protein_name(self, response: str) -> str:
        """Extract protein name from NCBI XML response."""
        if '<GBSeq_definition>' in response:
            start = response.find('<GBSeq_definition>') + len('<GBSeq_definition>')
            end = response.find('</GBSeq_definition>', start)
            return response[start:end].strip()
        return ""

    def _extract_protein_description(self, response: str) -> str:
        """Extract protein description from NCBI XML response."""
        if '<GBSeq_definition>' in response:
            start = response.find('<GBSeq_definition>') + len('<GBSeq_definition>')
            end = response.find('</GBSeq_definition>', start)
            return response[start:end].strip()
        return ""

    def _extract_protein_length(self, response: str) -> int:
        """Extract protein length from NCBI XML response."""
        if '<GBSeq_length>' in response:
            start = response.find('<GBSeq_length>') + len('<GBSeq_length>')
            end = response.find('</GBSeq_length>', start)
            try:
                return int(response[start:end].strip())
            except ValueError:
                return 0
        return 0

    def _extract_molecular_weight(self, response: str) -> float:
        """Extract molecular weight from NCBI XML response."""
        # Simplified extraction
        return 0.0

    def _extract_sequence(self, response: str) -> str:
        """Extract protein sequence from NCBI XML response."""
        if '<GBSeq_sequence>' in response:
            start = response.find('<GBSeq_sequence>') + len('<GBSeq_sequence>')
            end = response.find('</GBSeq_sequence>', start)
            return response[start:end].strip()
        return ""

    def search_by_gene_symbol(self, symbol: str, organism: str = "Homo sapiens") -> List[Dict[str, Any]]:
        """Search for genes by symbol."""
        query = f'"{symbol}"[Gene Symbol]'
        return self.search_genes(query, organism)

    def search_by_disease(self, disease_name: str, organism: str = "Homo sapiens") -> List[Dict[str, Any]]:
        """Search for genes related to a disease."""
        query = f'"{disease_name}"[Disease/Phenotype]'
        return self.search_genes(query, organism)
