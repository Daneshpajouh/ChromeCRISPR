"""
Ensembl API Client
Real-time integration with Ensembl REST API for genomic data
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_client import BaseAPIClient, APIResponse

logger = logging.getLogger(__name__)


class EnsemblClient(BaseAPIClient):
    """Client for Ensembl API with real data integration."""

    def __init__(self, config: Dict[str, Any], session=None):
        """Initialize Ensembl client with configuration and optional session"""
        super().__init__(config, session)

        # Ensembl specific configuration
        self.default_species = config.get('default_species', 'homo_sapiens')

        logger.info(f"Initialized EnsemblClient with base_url: {self.base_url}")

    async def search_genes_async(self, query: str, species: str = "homo_sapiens",
                                max_results: int = 100) -> List[Dict[str, Any]]:
        """Async version of search_genes"""
        try:
            params = {
                'q': query,
                'species': species,
                'expand': 1
            }

            logger.info(f"Searching Ensembl for: {query}")
            response = await self.get_async('lookup/symbol', params=params)

            if not response.success or not response.data:
                logger.warning("No search results found")
                return []

            # Handle single result vs multiple results
            data = response.data
            if isinstance(data, dict):
                genes = [data]
            elif isinstance(data, list):
                genes = data[:max_results]
            else:
                genes = []

            logger.info(f"Found {len(genes)} genes from Ensembl")

            # Process and standardize gene data
            processed_genes = []
            for gene in genes:
                try:
                    processed_gene = self._process_gene_data(gene)
                    if processed_gene:
                        processed_genes.append(processed_gene)
                except Exception as e:
                    logger.error(f"Error processing gene {gene.get('id', 'unknown')}: {str(e)}")
                    continue

            return processed_genes

        except Exception as e:
            logger.error(f"Error searching Ensembl: {str(e)}")
            return []

    def search_genes(self, query: str, species: str = "homo_sapiens",
                    max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search Ensembl for genes matching the query.

        Args:
            query: Search query string
            species: Species name (e.g., "homo_sapiens")
            max_results: Maximum number of results to return

        Returns:
            List of gene metadata dictionaries
        """
        try:
            params = {
                'q': query,
                'species': species,
                'expand': 1
            }

            self.logger.info(f"Searching Ensembl for: {query}")
            response = self.get('lookup/symbol', params=params)

            if not response:
                self.logger.warning("No search results found")
                return []

            # Handle single result vs multiple results
            if isinstance(response, dict):
                genes = [response]
            elif isinstance(response, list):
                genes = response[:max_results]
            else:
                genes = []

            self.logger.info(f"Found {len(genes)} genes from Ensembl")

            # Process and standardize gene data
            processed_genes = []
            for gene in genes:
                try:
                    processed_gene = self._process_gene_data(gene)
                    if processed_gene:
                        processed_genes.append(processed_gene)
                except Exception as e:
                    self.logger.error(f"Error processing gene {gene.get('id', 'unknown')}: {str(e)}")
                    continue

            return processed_genes

        except Exception as e:
            self.logger.error(f"Error searching Ensembl: {str(e)}")
            return []

    async def get_gene_details_async(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Async version of get_gene_details"""
        try:
            response = await self.get_async(f'lookup/id/{gene_id}', params={'expand': 1})

            if not response.success or not response.data:
                return None

            return self._process_gene_data(response.data)

        except Exception as e:
            logger.error(f"Error fetching gene details for {gene_id}: {str(e)}")
            return None

    def get_gene_details(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific gene by Ensembl ID."""
        try:
            response = self.get(f'lookup/id/{gene_id}', params={'expand': 1})

            if not response:
                return None

            return self._process_gene_data(response)

        except Exception as e:
            self.logger.error(f"Error fetching gene details for {gene_id}: {str(e)}")
            return None

    async def get_gene_by_symbol_async(self, symbol: str, species: str = "homo_sapiens") -> Optional[Dict[str, Any]]:
        """Async version of get_gene_by_symbol"""
        try:
            params = {
                'species': species,
                'expand': 1
            }

            response = await self.get_async(f'lookup/symbol/{symbol}', params=params)

            if not response.success or not response.data:
                return None

            return self._process_gene_data(response.data)

        except Exception as e:
            logger.error(f"Error fetching gene by symbol {symbol}: {str(e)}")
            return None

    def get_gene_by_symbol(self, symbol: str, species: str = "homo_sapiens") -> Optional[Dict[str, Any]]:
        """Get gene information by symbol."""
        try:
            params = {
                'species': species,
                'expand': 1
            }

            response = self.get(f'lookup/symbol/{symbol}', params=params)

            if not response:
                return None

            return self._process_gene_data(response)

        except Exception as e:
            self.logger.error(f"Error fetching gene by symbol {symbol}: {str(e)}")
            return None

    async def get_gene_sequence_async(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Async version of get_gene_sequence"""
        try:
            response = await self.get_async(f'sequence/id/{gene_id}')

            if not response.success or not response.data:
                return None

            data = response.data
            return {
                'gene_id': gene_id,
                'sequence': data.get('seq', ''),
                'molecule': data.get('molecule', ''),
                'version': data.get('version', ''),
                'source': 'Ensembl'
            }

        except Exception as e:
            logger.error(f"Error fetching gene sequence for {gene_id}: {str(e)}")
            return None

    def get_gene_sequence(self, gene_id: str) -> Optional[Dict[str, Any]]:
        """Get DNA sequence for a specific gene."""
        try:
            response = self.get(f'sequence/id/{gene_id}')

            if not response:
                return None

            return {
                'gene_id': gene_id,
                'sequence': response.get('seq', ''),
                'molecule': response.get('molecule', ''),
                'version': response.get('version', ''),
                'source': 'Ensembl'
            }

        except Exception as e:
            self.logger.error(f"Error fetching gene sequence for {gene_id}: {str(e)}")
            return None

    async def clear_cache_async(self):
        """Clear async cache"""
        await self._clear_cache_async()

    def clear_cache(self):
        """Clear sync cache"""
        self._clear_cache()

    def get_gene_transcripts(self, gene_id: str) -> List[Dict[str, Any]]:
        """Get transcripts for a specific gene."""
        try:
            response = self.get(f'overlap/id/{gene_id}', params={'feature': 'transcript'})

            if not response:
                return []

            transcripts = []
            for transcript in response:
                try:
                    processed_transcript = self._process_transcript_data(transcript)
                    if processed_transcript:
                        transcripts.append(processed_transcript)
                except Exception as e:
                    self.logger.error(f"Error processing transcript: {str(e)}")
                    continue

            return transcripts

        except Exception as e:
            self.logger.error(f"Error fetching transcripts for {gene_id}: {str(e)}")
            return []

    def get_gene_variants(self, gene_id: str) -> List[Dict[str, Any]]:
        """Get variants for a specific gene."""
        try:
            response = self.get(f'variation/id/{gene_id}')

            if not response:
                return []

            variants = []
            for variant in response:
                try:
                    processed_variant = self._process_variant_data(variant)
                    if processed_variant:
                        variants.append(processed_variant)
                except Exception as e:
                    self.logger.error(f"Error processing variant: {str(e)}")
                    continue

            return variants

        except Exception as e:
            self.logger.error(f"Error fetching variants for {gene_id}: {str(e)}")
            return []

    def search_by_region(self, region: str, species: str = "homo_sapiens") -> List[Dict[str, Any]]:
        """Search for genes in a specific genomic region."""
        try:
            params = {
                'species': species,
                'feature': 'gene'
            }

            response = self.get(f'overlap/region/{species}/{region}', params=params)

            if not response:
                return []

            genes = []
            for gene in response:
                try:
                    processed_gene = self._process_gene_data(gene)
                    if processed_gene:
                        genes.append(processed_gene)
                except Exception as e:
                    self.logger.error(f"Error processing region gene: {str(e)}")
                    continue

            return genes

        except Exception as e:
            self.logger.error(f"Error searching by region {region}: {str(e)}")
            return []

    def _process_gene_data(self, gene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and standardize Ensembl gene data."""
        try:
            # Extract basic information
            gene_id = gene.get('id', '')
            name = gene.get('display_name', '')
            description = gene.get('description', '')

            # Extract location information
            location = {}
            if 'seq_region_name' in gene:
                location['chromosome'] = gene['seq_region_name']
            if 'start' in gene:
                location['start'] = gene['start']
            if 'end' in gene:
                location['end'] = gene['end']
            if 'strand' in gene:
                location['strand'] = gene['strand']

            # Extract biotype
            biotype = gene.get('biotype', '')

            # Extract version
            version = gene.get('version', '')

            # Extract species
            species = gene.get('species', '')

            processed_gene = {
                'gene_id': gene_id,
                'name': name,
                'description': description,
                'location': location,
                'biotype': biotype,
                'version': version,
                'species': species,
                'source': 'Ensembl',
                'url': f"http://www.ensembl.org/id/{gene_id}" if gene_id else ""
            }

            return processed_gene

        except Exception as e:
            self.logger.error(f"Error processing Ensembl gene data: {str(e)}")
            return None

    def _process_transcript_data(self, transcript: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and standardize transcript data."""
        try:
            transcript_id = transcript.get('id', '')
            name = transcript.get('display_name', '')
            biotype = transcript.get('biotype', '')

            # Extract location
            location = {}
            if 'seq_region_name' in transcript:
                location['chromosome'] = transcript['seq_region_name']
            if 'start' in transcript:
                location['start'] = transcript['start']
            if 'end' in transcript:
                location['end'] = transcript['end']
            if 'strand' in transcript:
                location['strand'] = transcript['strand']

            processed_transcript = {
                'transcript_id': transcript_id,
                'name': name,
                'biotype': biotype,
                'location': location,
                'source': 'Ensembl'
            }

            return processed_transcript

        except Exception as e:
            self.logger.error(f"Error processing transcript data: {str(e)}")
            return None

    def _process_variant_data(self, variant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and standardize variant data."""
        try:
            variant_id = variant.get('id', '')
            name = variant.get('name', '')
            consequence_type = variant.get('consequence_type', '')

            # Extract location
            location = {}
            if 'seq_region_name' in variant:
                location['chromosome'] = variant['seq_region_name']
            if 'start' in variant:
                location['start'] = variant['start']
            if 'end' in variant:
                location['end'] = variant['end']
            if 'strand' in variant:
                location['strand'] = variant['strand']

            processed_variant = {
                'variant_id': variant_id,
                'name': name,
                'consequence_type': consequence_type,
                'location': location,
                'source': 'Ensembl'
            }

            return processed_variant

        except Exception as e:
            self.logger.error(f"Error processing variant data: {str(e)}")
            return None

    def search_by_gene_symbol(self, symbol: str, species: str = "homo_sapiens") -> List[Dict[str, Any]]:
        """Search for genes by symbol."""
        gene = self.get_gene_by_symbol(symbol, species)
        return [gene] if gene else []

    def get_species_list(self) -> List[Dict[str, Any]]:
        """Get list of available species."""
        try:
            response = self.get('info/species')

            if not response or 'species' not in response:
                return []

            species_list = []
            for species in response['species']:
                try:
                    species_data = {
                        'name': species.get('name', ''),
                        'display_name': species.get('display_name', ''),
                        'assembly': species.get('assembly', ''),
                        'accession': species.get('accession', ''),
                        'source': 'Ensembl'
                    }
                    species_list.append(species_data)
                except Exception as e:
                    self.logger.error(f"Error processing species: {str(e)}")
                    continue

            return species_list

        except Exception as e:
            self.logger.error(f"Error fetching species list: {str(e)}")
            return []
