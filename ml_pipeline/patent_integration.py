"""
Patent Integration Module
Based on GeneX Phase 1 Research Report 3/3

This module implements specialized extraction for patent databases to integrate
commercial and translational data into the GeneX knowledge base.
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import requests
from urllib.parse import urlencode
import time

logger = logging.getLogger(__name__)


class PatentSource(Enum):
    """Patent data sources as recommended in the research report."""
    USPTO = "uspto"
    GOOGLE_PATENTS = "google_patents"
    LENS_ORG = "lens_org"
    EPO = "epo"
    WIPO = "wipo"


@dataclass
class PatentMetadata:
    """Metadata for patent documents."""
    patent_id: str
    title: str
    abstract: str
    inventors: List[str]
    assignees: List[str]
    filing_date: str
    publication_date: str
    priority_date: str
    patent_family: List[str]
    legal_status: str
    jurisdiction: str
    classification_codes: List[str]
    citation_count: int
    forward_citations: List[str]
    backward_citations: List[str]


@dataclass
class PatentClaim:
    """Individual patent claim with structured information."""
    claim_number: int
    claim_text: str
    claim_type: str  # independent, dependent
    key_entities: List[str]
    key_relations: List[str]
    novelty_elements: List[str]


@dataclass
class PatentDocument:
    """Complete patent document with structured extraction."""
    metadata: PatentMetadata
    claims: List[PatentClaim]
    description: str
    drawings: List[str]
    examples: List[str]
    extracted_entities: Dict[str, List[str]]
    extracted_relations: List[Dict[str, str]]
    gene_editing_technologies: List[str]
    therapeutic_applications: List[str]
    delivery_methods: List[str]
    efficiency_claims: List[str]
    safety_claims: List[str]


class PatentAPIClient:
    """Base class for patent API clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GeneX-Patent-Integration/1.0'
        })

    def search_patents(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for patents using the specific API."""
        raise NotImplementedError

    def get_patent_details(self, patent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific patent."""
        raise NotImplementedError

    def extract_gene_editing_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract gene editing specific entities from patent text."""
        raise NotImplementedError


class GooglePatentsClient(PatentAPIClient):
    """Client for Google Patents API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = "https://patents.google.com/api/query"
        self.api_key = config.get('google_patents', {}).get('api_key', '')

    def search_patents(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search Google Patents for gene editing related patents."""
        search_terms = [
            "CRISPR",
            "gene editing",
            "genome editing",
            "prime editing",
            "base editing",
            "TALEN",
            "ZFN",
            "nuclease",
            "guide RNA"
        ]

        results = []
        for term in search_terms:
            try:
                params = {
                    'q': f'"{term}" AND "gene editing"',
                    'num': min(max_results // len(search_terms), 20),
                    'language': 'ENGLISH',
                    'type': 'PATENT'
                }

                if self.api_key:
                    params['key'] = self.api_key

                response = self.session.get(self.base_url, params=params)
                response.raise_for_status()

                data = response.json()
                if 'results' in data:
                    results.extend(data['results'])

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error searching Google Patents for '{term}': {e}")

        return results[:max_results]

    def get_patent_details(self, patent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed patent information from Google Patents."""
        try:
            url = f"https://patents.google.com/patent/{patent_id}/en"
            response = self.session.get(url)
            response.raise_for_status()

            # Parse the HTML response to extract structured data
            # This is a simplified version - in practice, you'd use proper HTML parsing
            return {
                'patent_id': patent_id,
                'url': url,
                'raw_html': response.text
            }

        except Exception as e:
            logger.error(f"Error getting patent details for {patent_id}: {e}")
            return None


class USPTOClient(PatentAPIClient):
    """Client for USPTO Patent API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = "https://developer.uspto.gov/ds-api"
        self.api_key = config.get('uspto', {}).get('api_key', '')

    def search_patents(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search USPTO patents for gene editing related inventions."""
        try:
            params = {
                'searchText': query,
                'maxRec': max_results,
                'startRec': 0
            }

            if self.api_key:
                params['api_key'] = self.api_key

            response = self.session.get(f"{self.base_url}/patents", params=params)
            response.raise_for_status()

            data = response.json()
            return data.get('results', [])

        except Exception as e:
            logger.error(f"Error searching USPTO patents: {e}")
            return []


class PatentExtractor:
    """
    Specialized extractor for patent language and structure.

    Based on the research report recommendation to develop dedicated NLP models
    fine-tuned on patent language, which is highly formulaic and distinct from
    academic writing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patent_specific_entities = {
            'inventor': [],
            'assignee': [],
            'patent_family': [],
            'priority_date': [],
            'legal_status': [],
            'classification': [],
            'claim_element': [],
            'novelty_statement': [],
            'advantage_statement': []
        }

    def extract_patent_entities(self, patent_text: str) -> Dict[str, List[str]]:
        """
        Extract patent-specific entities using specialized patterns.

        Args:
            patent_text: Text content of the patent

        Returns:
            Dictionary of extracted entities by category
        """
        entities = {}

        # Extract inventors
        entities['inventors'] = self._extract_inventors(patent_text)

        # Extract assignees
        entities['assignees'] = self._extract_assignees(patent_text)

        # Extract gene editing technologies
        entities['gene_editing_technologies'] = self._extract_gene_editing_tech(patent_text)

        # Extract therapeutic applications
        entities['therapeutic_applications'] = self._extract_therapeutic_apps(patent_text)

        # Extract delivery methods
        entities['delivery_methods'] = self._extract_delivery_methods(patent_text)

        # Extract efficiency claims
        entities['efficiency_claims'] = self._extract_efficiency_claims(patent_text)

        # Extract safety claims
        entities['safety_claims'] = self._extract_safety_claims(patent_text)

        return entities

    def _extract_inventors(self, text: str) -> List[str]:
        """Extract inventor names from patent text."""
        import re

        # Common patterns for inventor extraction
        patterns = [
            r'inventors?[:\s]+([^\.]+)',
            r'invented by[:\s]+([^\.]+)',
            r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]

        inventors = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            inventors.extend(matches)

        return list(set(inventors))

    def _extract_assignees(self, text: str) -> List[str]:
        """Extract assignee organizations from patent text."""
        import re

        # Common patterns for assignee extraction
        patterns = [
            r'assignee[:\s]+([^\.]+)',
            r'assigned to[:\s]+([^\.]+)',
            r'owned by[:\s]+([^\.]+)',
        ]

        assignees = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            assignees.extend(matches)

        return list(set(assignees))

    def _extract_gene_editing_tech(self, text: str) -> List[str]:
        """Extract gene editing technologies mentioned in patent."""
        import re

        technologies = []

        # CRISPR-related
        crispr_patterns = [
            r'CRISPR[-\s]?Cas\d*',
            r'clustered regularly interspaced short palindromic repeats',
            r'guide RNA',
            r'sgRNA',
            r'pegRNA'
        ]

        for pattern in crispr_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technologies.extend(matches)

        # Base editing
        base_editing_patterns = [
            r'base editor',
            r'cytosine base editor',
            r'adenine base editor',
            r'BE\d*',
            r'CBE',
            r'ABE'
        ]

        for pattern in base_editing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technologies.extend(matches)

        # Prime editing
        prime_editing_patterns = [
            r'prime editor',
            r'prime editing',
            r'PE\d*'
        ]

        for pattern in prime_editing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technologies.extend(matches)

        # Other nucleases
        other_patterns = [
            r'TALEN',
            r'ZFN',
            r'zinc finger nuclease',
            r'transcription activator-like effector nuclease'
        ]

        for pattern in other_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technologies.extend(matches)

        return list(set(technologies))

    def _extract_therapeutic_apps(self, text: str) -> List[str]:
        """Extract therapeutic applications from patent text."""
        import re

        applications = []

        # Disease patterns
        disease_patterns = [
            r'treat(?:ing|ment of)\s+([^\.]+)',
            r'therapeutic.*?for\s+([^\.]+)',
            r'medical.*?application.*?([^\.]+)',
        ]

        for pattern in disease_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            applications.extend(matches)

        return list(set(applications))

    def _extract_delivery_methods(self, text: str) -> List[str]:
        """Extract delivery methods from patent text."""
        import re

        delivery_methods = []

        # Viral vectors
        viral_patterns = [
            r'AAV',
            r'adeno-associated virus',
            r'lentivirus',
            r'retrovirus',
            r'viral vector'
        ]

        for pattern in viral_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            delivery_methods.extend(matches)

        # Non-viral methods
        nonviral_patterns = [
            r'lipid nanoparticle',
            r'LNP',
            r'electroporation',
            r'microinjection',
            r'nanoparticle'
        ]

        for pattern in nonviral_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            delivery_methods.extend(matches)

        return list(set(delivery_methods))

    def _extract_efficiency_claims(self, text: str) -> List[str]:
        """Extract efficiency claims from patent text."""
        import re

        efficiency_claims = []

        # Efficiency patterns
        patterns = [
            r'efficiency.*?(\d+[\.\d]*\s*%)',
            r'(\d+[\.\d]*\s*%).*?efficiency',
            r'editing.*?(\d+[\.\d]*\s*%)',
            r'(\d+[\.\d]*\s*%).*?editing'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            efficiency_claims.extend(matches)

        return list(set(efficiency_claims))

    def _extract_safety_claims(self, text: str) -> List[str]:
        """Extract safety claims from patent text."""
        import re

        safety_claims = []

        # Safety patterns
        patterns = [
            r'safe.*?(\d+[\.\d]*\s*%)',
            r'(\d+[\.\d]*\s*%).*?safe',
            r'off-target.*?(\d+[\.\d]*\s*%)',
            r'(\d+[\.\d]*\s*%).*?off-target',
            r'no.*?off-target',
            r'reduced.*?off-target'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            safety_claims.extend(matches)

        return list(set(safety_claims))

    def extract_patent_relations(self, patent_text: str) -> List[Dict[str, str]]:
        """
        Extract relations from patent text using specialized patterns.

        Args:
            patent_text: Text content of the patent

        Returns:
            List of extracted relations
        """
        relations = []

        # Extract technology-application relations
        tech_app_relations = self._extract_tech_application_relations(patent_text)
        relations.extend(tech_app_relations)

        # Extract technology-delivery relations
        tech_delivery_relations = self._extract_tech_delivery_relations(patent_text)
        relations.extend(tech_delivery_relations)

        # Extract efficiency-claims relations
        efficiency_relations = self._extract_efficiency_relations(patent_text)
        relations.extend(efficiency_relations)

        return relations

    def _extract_tech_application_relations(self, text: str) -> List[Dict[str, str]]:
        """Extract technology-application relations."""
        import re

        relations = []

        # Pattern: [Technology] for [Application]
        pattern = r'([A-Z][A-Za-z\s-]+(?:CRISPR|Cas\d*|TALEN|ZFN|editor|editing))[^\.]*?(?:for|in|to)\s+([^\.]+)'

        matches = re.findall(pattern, text, re.IGNORECASE)
        for tech, app in matches:
            relations.append({
                'subject': tech.strip(),
                'predicate': 'TREATS',
                'object': app.strip()
            })

        return relations

    def _extract_tech_delivery_relations(self, text: str) -> List[Dict[str, str]]:
        """Extract technology-delivery method relations."""
        import re

        relations = []

        # Pattern: [Technology] delivered by [Method]
        pattern = r'([A-Z][A-Za-z\s-]+(?:CRISPR|Cas\d*|TALEN|ZFN|editor|editing))[^\.]*?(?:delivered|administered|introduced)[^\.]*?(?:by|via|using)\s+([^\.]+)'

        matches = re.findall(pattern, text, re.IGNORECASE)
        for tech, method in matches:
            relations.append({
                'subject': tech.strip(),
                'predicate': 'DELIVERED_BY',
                'object': method.strip()
            })

        return relations

    def _extract_efficiency_relations(self, text: str) -> List[Dict[str, str]]:
        """Extract efficiency-claim relations."""
        import re

        relations = []

        # Pattern: [Technology] achieves [Efficiency]
        pattern = r'([A-Z][A-Za-z\s-]+(?:CRISPR|Cas\d*|TALEN|ZFN|editor|editing))[^\.]*?(?:achieves|achieve|achieveing)\s+(\d+[\.\d]*\s*%)'

        matches = re.findall(pattern, text, re.IGNORECASE)
        for tech, efficiency in matches:
            relations.append({
                'subject': tech.strip(),
                'predicate': 'ACHIEVES_EFFICIENCY',
                'object': efficiency.strip()
            })

        return relations


class PatentIntegrationPipeline:
    """
    Main pipeline for integrating patent data into the GeneX knowledge base.

    This implements the research report recommendation to expand the knowledge
    base with heterogeneous sources including patent databases.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clients = {
            PatentSource.GOOGLE_PATENTS: GooglePatentsClient(config),
            PatentSource.USPTO: USPTOClient(config)
        }
        self.extractor = PatentExtractor(config)

    def search_gene_editing_patents(self, max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Search for gene editing related patents across multiple sources.

        Args:
            max_results: Maximum number of patents to retrieve

        Returns:
            List of patent metadata
        """
        all_patents = []

        for source, client in self.clients.items():
            try:
                logger.info(f"Searching {source.value} for gene editing patents...")
                patents = client.search_patents("gene editing", max_results // len(self.clients))
                all_patents.extend(patents)

            except Exception as e:
                logger.error(f"Error searching {source.value}: {e}")

        return all_patents[:max_results]

    def process_patent_document(self, patent_data: Dict[str, Any]) -> Optional[PatentDocument]:
        """
        Process a patent document and extract structured information.

        Args:
            patent_data: Raw patent data from API

        Returns:
            Structured PatentDocument or None if processing fails
        """
        try:
            # Extract metadata
            metadata = self._extract_metadata(patent_data)

            # Extract claims
            claims = self._extract_claims(patent_data)

            # Extract description and other sections
            description = patent_data.get('description', '')
            drawings = patent_data.get('drawings', [])
            examples = patent_data.get('examples', [])

            # Extract entities and relations
            entities = self.extractor.extract_patent_entities(description)
            relations = self.extractor.extract_patent_relations(description)

            # Create patent document
            patent_doc = PatentDocument(
                metadata=metadata,
                claims=claims,
                description=description,
                drawings=drawings,
                examples=examples,
                extracted_entities=entities,
                extracted_relations=relations,
                gene_editing_technologies=entities.get('gene_editing_technologies', []),
                therapeutic_applications=entities.get('therapeutic_applications', []),
                delivery_methods=entities.get('delivery_methods', []),
                efficiency_claims=entities.get('efficiency_claims', []),
                safety_claims=entities.get('safety_claims', [])
            )

            return patent_doc

        except Exception as e:
            logger.error(f"Error processing patent document: {e}")
            return None

    def _extract_metadata(self, patent_data: Dict[str, Any]) -> PatentMetadata:
        """Extract metadata from patent data."""
        return PatentMetadata(
            patent_id=patent_data.get('patent_id', ''),
            title=patent_data.get('title', ''),
            abstract=patent_data.get('abstract', ''),
            inventors=patent_data.get('inventors', []),
            assignees=patent_data.get('assignees', []),
            filing_date=patent_data.get('filing_date', ''),
            publication_date=patent_data.get('publication_date', ''),
            priority_date=patent_data.get('priority_date', ''),
            patent_family=patent_data.get('patent_family', []),
            legal_status=patent_data.get('legal_status', ''),
            jurisdiction=patent_data.get('jurisdiction', ''),
            classification_codes=patent_data.get('classification_codes', []),
            citation_count=patent_data.get('citation_count', 0),
            forward_citations=patent_data.get('forward_citations', []),
            backward_citations=patent_data.get('backward_citations', [])
        )

    def _extract_claims(self, patent_data: Dict[str, Any]) -> List[PatentClaim]:
        """Extract patent claims."""
        claims = []
        raw_claims = patent_data.get('claims', [])

        for i, claim_text in enumerate(raw_claims):
            claim = PatentClaim(
                claim_number=i + 1,
                claim_text=claim_text,
                claim_type='independent' if i == 0 else 'dependent',
                key_entities=[],
                key_relations=[],
                novelty_elements=[]
            )
            claims.append(claim)

        return claims

    def integrate_with_academic_literature(self, patent_docs: List[PatentDocument],
                                         academic_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Integrate patent data with academic literature facts.

        Args:
            patent_docs: List of processed patent documents
            academic_facts: List of facts from academic literature

        Returns:
            Integrated facts with patent-academic links
        """
        integrated_facts = []

        for patent_doc in patent_docs:
            # Create patent-specific facts
            patent_facts = self._create_patent_facts(patent_doc)
            integrated_facts.extend(patent_facts)

            # Link to academic literature
            academic_links = self._find_academic_links(patent_doc, academic_facts)
            integrated_facts.extend(academic_links)

        return integrated_facts

    def _create_patent_facts(self, patent_doc: PatentDocument) -> List[Dict[str, Any]]:
        """Create facts from patent document."""
        facts = []

        # Technology facts
        for tech in patent_doc.gene_editing_technologies:
            facts.append({
                'subject': tech,
                'predicate': 'PATENTED_IN',
                'object': patent_doc.metadata.patent_id,
                'source_type': 'patent',
                'source_id': patent_doc.metadata.patent_id,
                'publication_date': patent_doc.metadata.publication_date
            })

        # Application facts
        for app in patent_doc.therapeutic_applications:
            facts.append({
                'subject': patent_doc.gene_editing_technologies[0] if patent_doc.gene_editing_technologies else 'Gene Editing Technology',
                'predicate': 'TREATS',
                'object': app,
                'source_type': 'patent',
                'source_id': patent_doc.metadata.patent_id,
                'publication_date': patent_doc.metadata.publication_date
            })

        # Delivery facts
        for method in patent_doc.delivery_methods:
            facts.append({
                'subject': patent_doc.gene_editing_technologies[0] if patent_doc.gene_editing_technologies else 'Gene Editing Technology',
                'predicate': 'DELIVERED_BY',
                'object': method,
                'source_type': 'patent',
                'source_id': patent_doc.metadata.patent_id,
                'publication_date': patent_doc.metadata.publication_date
            })

        return facts

    def _find_academic_links(self, patent_doc: PatentDocument,
                           academic_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find links between patent and academic literature."""
        links = []

        # Simple string matching for now - could be enhanced with semantic similarity
        for fact in academic_facts:
            for tech in patent_doc.gene_editing_technologies:
                if tech.lower() in fact.get('subject', '').lower():
                    links.append({
                        'subject': patent_doc.metadata.patent_id,
                        'predicate': 'RELATES_TO_ACADEMIC',
                        'object': fact.get('source_id', ''),
                        'source_type': 'integration',
                        'source_id': f"link_{patent_doc.metadata.patent_id}_{fact.get('source_id', '')}",
                        'publication_date': patent_doc.metadata.publication_date
                    })

        return links
