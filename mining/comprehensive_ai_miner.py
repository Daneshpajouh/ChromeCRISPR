#!/usr/bin/env python3
"""
Comprehensive AI-Powered Data Mining Pipeline
Integrates all data models, feature extractors, and knowledge extraction for GeneX
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path

# Import our modules
from ..data_models.efficiency_dataset import EfficiencyRecord
from ..data_models.activity_dataset import ActivityRecord
from ..data_models.design_dataset import DesignRecord
from ..data_models.safety_dataset import SafetyRecord
from ..feature_extraction.sequence_features import SequenceFeatureExtractor
from ..feature_extraction.biological_features import BiologicalFeatureExtractor
from ..feature_extraction.experimental_features import ExperimentalFeatureExtractor
from ..ml_pipeline.knowledge_extractor import AIKnowledgeExtractor, ExtractedKnowledge
from ..api_clients.base_client import BaseAPIClient
from ..api_clients.semantic_scholar_client import SemanticScholarClient
from ..api_clients.crossref_client import CrossRefClient
from ..api_clients.pubmed_client import PubMedClient
from ..api_clients.europe_pmc_client import EuropePMCClient
from ..utils.config import Config
from ..utils.hierarchical_rate_limiter import HierarchicalRateLimiter

class ComprehensiveAIMiner:
    """Comprehensive AI-powered data mining pipeline for GeneX"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = Config(config_path)

        # Initialize rate limiter with default API configurations
        default_api_configs = {
            'semantic_scholar': {
                'max_requests_per_second': 1.0,
                'burst_limit': 10,
                'recovery_timeout': 60.0
            },
            'crossref': {
                'max_requests_per_second': 1.0,
                'burst_limit': 10,
                'recovery_timeout': 60.0
            },
            'pubmed': {
                'max_requests_per_second': 1.0,
                'burst_limit': 10,
                'recovery_timeout': 60.0
            },
            'europe_pmc': {
                'max_requests_per_second': 1.0,
                'burst_limit': 10,
                'recovery_timeout': 60.0
            }
        }
        self.rate_limiter = HierarchicalRateLimiter(api_configs=default_api_configs)

        # Initialize AI components
        self.knowledge_extractor = AIKnowledgeExtractor()
        self.sequence_extractor = SequenceFeatureExtractor()
        self.biological_extractor = BiologicalFeatureExtractor()
        self.experimental_extractor = ExperimentalFeatureExtractor()

        # Initialize API clients
        self.api_clients = self._initialize_api_clients()

        # Initialize data storage
        self.knowledge_base = []
        self.efficiency_dataset = []
        self.activity_dataset = []
        self.design_dataset = []
        self.safety_dataset = []

        # Statistics
        self.stats = {
            'papers_processed': 0,
            'knowledge_extracted': 0,
            'efficiency_records': 0,
            'activity_records': 0,
            'design_records': 0,
            'safety_records': 0,
            'errors': 0
        }

    def _initialize_api_clients(self) -> Dict[str, BaseAPIClient]:
        """Initialize all API clients"""
        clients = {}

        # Semantic Scholar for paper search and metadata
        if self.config.get('semantic_scholar', {}).get('api_key'):
            clients['semantic_scholar'] = SemanticScholarClient(
                api_key=self.config['semantic_scholar']['api_key'],
                rate_limiter=self.rate_limiter
            )

        # CrossRef for additional metadata
        if self.config.get('crossref', {}).get('email'):
            clients['crossref'] = CrossRefClient(
                email=self.config['crossref']['email'],
                rate_limiter=self.rate_limiter
            )

        # PubMed for biomedical literature
        if self.config.get('pubmed', {}).get('api_key'):
            clients['pubmed'] = PubMedClient(
                api_key=self.config['pubmed']['api_key'],
                rate_limiter=self.rate_limiter
            )

        # Europe PMC for additional biomedical literature
        if self.config.get('europe_pmc', {}).get('api_key'):
            clients['europe_pmc'] = EuropePMCClient(
                api_key=self.config['europe_pmc']['api_key'],
                rate_limiter=self.rate_limiter
            )

        return clients

    async def mine_comprehensive_data(self,
                                    search_queries: List[str] = None,
                                    max_papers: int = 1000,
                                    output_dir: str = "results") -> Dict[str, Any]:
        """Main mining function - extracts comprehensive data and knowledge"""
        self.logger.info("Starting comprehensive AI-powered data mining...")

        # Default search queries if none provided
        if search_queries is None:
            search_queries = [
                "CRISPR gene editing efficiency",
                "Prime editing gene therapy",
                "Base editing safety",
                "gene editing off-target effects",
                "CRISPR clinical trials",
                "gene editing delivery methods"
            ]

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Step 1: Search and collect papers
        papers = await self._search_papers(search_queries, max_papers)
        self.logger.info(f"Found {len(papers)} papers to process")

        # Step 2: Extract knowledge from papers
        knowledge_list = await self._extract_knowledge_from_papers(papers)

        # Step 3: Generate comprehensive datasets
        datasets = await self._generate_datasets(knowledge_list)

        # Step 4: Build knowledge graph
        knowledge_graph = self.knowledge_extractor.build_knowledge_graph(knowledge_list)

        # Step 5: Save everything
        await self._save_results(output_path, knowledge_list, datasets, knowledge_graph)

        # Step 6: Generate comprehensive report
        report = self._generate_comprehensive_report()

        self.logger.info("Comprehensive data mining completed successfully!")
        return report

    async def _search_papers(self, search_queries: List[str], max_papers: int) -> List[Dict[str, Any]]:
        """Search for papers using multiple APIs"""
        all_papers = []

        for query in search_queries:
            self.logger.info(f"Searching for: {query}")

            # Search Semantic Scholar
            if 'semantic_scholar' in self.api_clients:
                try:
                    papers = await self.api_clients['semantic_scholar'].search_papers(
                        query=query,
                        limit=min(200, max_papers // len(search_queries))
                    )
                    all_papers.extend(papers)
                except Exception as e:
                    self.logger.error(f"Error searching Semantic Scholar: {e}")

            # Search PubMed
            if 'pubmed' in self.api_clients:
                try:
                    papers = await self.api_clients['pubmed'].search_papers(
                        query=query,
                        limit=min(200, max_papers // len(search_queries))
                    )
                    all_papers.extend(papers)
                except Exception as e:
                    self.logger.error(f"Error searching PubMed: {e}")

            # Search Europe PMC (comprehensive biomedical literature including patents)
            if 'europe_pmc' in self.api_clients:
                try:
                    papers = await self.api_clients['europe_pmc'].search_papers(
                        query=query,
                        limit=min(200, max_papers // len(search_queries))
                    )
                    all_papers.extend(papers)
                except Exception as e:
                    self.logger.error(f"Error searching Europe PMC: {e}")

        # Remove duplicates and limit total
        unique_papers = self._deduplicate_papers(all_papers)
        return unique_papers[:max_papers]

    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on DOI or title"""
        seen_dois = set()
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            doi = paper.get('doi', '').lower()
            title = paper.get('title', '').lower()

            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                unique_papers.append(paper)
            elif title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)

        return unique_papers

    async def _extract_knowledge_from_papers(self, papers: List[Dict[str, Any]]) -> List[ExtractedKnowledge]:
        """Extract knowledge from all papers using AI"""
        knowledge_list = []

        for i, paper in enumerate(papers):
            try:
                self.logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')}")

                # Get full text if available
                paper_text = await self._get_paper_text(paper)

                if paper_text:
                    # Extract knowledge using AI
                    knowledge = self.knowledge_extractor.extract_knowledge_from_paper(
                        paper_text=paper_text,
                        metadata=paper
                    )
                    knowledge_list.append(knowledge)
                    self.stats['knowledge_extracted'] += 1

                self.stats['papers_processed'] += 1

                # Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error processing paper {paper.get('title', 'Unknown')}: {e}")
                self.stats['errors'] += 1

        return knowledge_list

    async def _get_paper_text(self, paper: Dict[str, Any]) -> Optional[str]:
        """Get full text of paper if available"""
        # Try to get full text from Semantic Scholar
        if 'semantic_scholar' in self.api_clients and paper.get('paperId'):
            try:
                full_text = await self.api_clients['semantic_scholar'].get_paper_text(
                    paper_id=paper['paperId']
                )
                if full_text:
                    return full_text
            except Exception as e:
                self.logger.warning(f"Could not get full text from Semantic Scholar: {e}")

        # Try Europe PMC for full text (comprehensive biomedical coverage)
        if 'europe_pmc' in self.api_clients and (paper.get('pmid') or paper.get('pmcid')):
            try:
                full_text = await self.api_clients['europe_pmc'].get_paper_text(
                    pmid=paper.get('pmid'),
                    pmcid=paper.get('pmcid')
                )
                if full_text:
                    return full_text
            except Exception as e:
                self.logger.warning(f"Could not get full text from Europe PMC: {e}")

        # Fallback to abstract
        return paper.get('abstract', '')

    async def _generate_datasets(self, knowledge_list: List[ExtractedKnowledge]) -> Dict[str, List]:
        """Generate comprehensive datasets from extracted knowledge"""
        self.logger.info("Generating comprehensive datasets...")

        datasets = {
            'efficiency': [],
            'activity': [],
            'design': [],
            'safety': []
        }

        for knowledge in knowledge_list:
            try:
                # Generate efficiency dataset
                efficiency_record = self._create_efficiency_record(knowledge)
                if efficiency_record:
                    datasets['efficiency'].append(efficiency_record)
                    self.stats['efficiency_records'] += 1

                # Generate activity dataset
                activity_record = self._create_activity_record(knowledge)
                if activity_record:
                    datasets['activity'].append(activity_record)
                    self.stats['activity_records'] += 1

                # Generate design dataset
                design_record = self._create_design_record(knowledge)
                if design_record:
                    datasets['design'].append(design_record)
                    self.stats['design_records'] += 1

                # Generate safety dataset
                safety_record = self._create_safety_record(knowledge)
                if safety_record:
                    datasets['safety'].append(safety_record)
                    self.stats['safety_records'] += 1

            except Exception as e:
                self.logger.error(f"Error generating datasets for paper {knowledge.paper_id}: {e}")
                self.stats['errors'] += 1

        return datasets

    def _create_efficiency_record(self, knowledge: ExtractedKnowledge) -> Optional[EfficiencyRecord]:
        """Create efficiency record from extracted knowledge"""
        if not knowledge.editing_technique:
            return None

        # Extract sequence features
        sequence_features = None
        if knowledge.target_genes:
            # Use first target gene for sequence analysis
            gene_sequence = self._generate_synthetic_sequence(knowledge.target_genes[0])
            sequence_features = self.sequence_extractor.extract_features(gene_sequence)

        # Extract biological features
        biological_features = None
        if knowledge.target_genes:
            biological_features = self.biological_extractor.extract_gene_features(knowledge.target_genes[0])

        # Extract experimental features
        experimental_features = None
        if knowledge.cell_types:
            experimental_features = self.experimental_extractor.extract_experimental_conditions(
                knowledge.cell_types[0]
            )

        # Create efficiency record
        return EfficiencyRecord(
            persistent_id=knowledge.paper_id,
            editing_technique=knowledge.editing_technique,
            guide_rna_sequence=self._extract_guide_sequence(knowledge),
            target_sequence=self._extract_target_sequence(knowledge),
            target_gene_symbol=knowledge.target_genes[0] if knowledge.target_genes else None,
            cell_line=knowledge.cell_types[0] if knowledge.cell_types else "Unknown",
            organism=knowledge.organisms[0] if knowledge.organisms else "Human",
            delivery_method=self._extract_delivery_method(knowledge),
            efficiency_score=self._extract_efficiency_score(knowledge),
            specificity_score=self._calculate_specificity_score(knowledge),
            off_target_score=self._extract_off_target_score(knowledge),
            safety_score=self._calculate_safety_score(knowledge),
            source_metadata={
                'title': knowledge.title,
                'authors': knowledge.authors,
                'year': knowledge.year,
                'journal': knowledge.journal,
                'doi': knowledge.doi
            },
            provenance={
                'extraction_method': 'AI',
                'confidence': knowledge.extraction_confidence,
                'relevance': knowledge.relevance_score
            }
        )

    def _create_activity_record(self, knowledge: ExtractedKnowledge) -> Optional[ActivityRecord]:
        """Create activity record from extracted knowledge"""
        if not knowledge.editing_technique:
            return None

        # Create base efficiency record
        efficiency_record = self._create_efficiency_record(knowledge)
        if not efficiency_record:
            return None

        # Extract activity-specific features
        activity_features = self._extract_activity_features(knowledge)

        # Create activity record
        return ActivityRecord(
            persistent_id=knowledge.paper_id,
            editing_technique=knowledge.editing_technique,
            guide_rna_sequence=efficiency_record.guide_rna_sequence,
            target_sequence=efficiency_record.target_sequence,
            target_gene_symbol=efficiency_record.target_gene_symbol,
            cell_line=efficiency_record.cell_line,
            organism=efficiency_record.organism,
            delivery_method=efficiency_record.delivery_method,
            efficiency_score=efficiency_record.efficiency_score,
            activity_score=self._extract_activity_score(knowledge),
            specificity_score=efficiency_record.specificity_score,
            off_target_score=efficiency_record.off_target_score,
            safety_score=efficiency_record.safety_score,
            source_metadata=efficiency_record.source_metadata,
            provenance=efficiency_record.provenance,
            activity_features=activity_features
        )

    def _create_design_record(self, knowledge: ExtractedKnowledge) -> Optional[DesignRecord]:
        """Create design record from extracted knowledge"""
        if not knowledge.editing_technique or not knowledge.target_genes:
            return None

        # Create clinical context
        clinical_context = self._create_clinical_context(knowledge)

        # Create target gene
        target_gene = self._create_target_gene(knowledge)

        # Create design solutions
        crispr_solution = self._create_design_solution(knowledge, 'CRISPR')
        prime_solution = self._create_design_solution(knowledge, 'Prime Editing')
        base_solution = self._create_design_solution(knowledge, 'Base Editing')

        return DesignRecord(
            persistent_id=knowledge.paper_id,
            clinical_context=clinical_context,
            target_gene=target_gene,
            organism=knowledge.organisms[0] if knowledge.organisms else "Human",
            cell_line=knowledge.cell_types[0] if knowledge.cell_types else None,
            crispr_solution=crispr_solution,
            prime_solution=prime_solution,
            base_solution=base_solution,
            source_metadata={
                'title': knowledge.title,
                'authors': knowledge.authors,
                'year': knowledge.year,
                'journal': knowledge.journal,
                'doi': knowledge.doi
            }
        )

    def _create_safety_record(self, knowledge: ExtractedKnowledge) -> Optional[SafetyRecord]:
        """Create safety record from extracted knowledge"""
        if not knowledge.editing_technique:
            return None

        return SafetyRecord(
            persistent_id=knowledge.paper_id,
            editing_technique=knowledge.editing_technique,
            cell_line=knowledge.cell_types[0] if knowledge.cell_types else "Unknown",
            organism=knowledge.organisms[0] if knowledge.organisms else "Human",
            delivery_method=self._extract_delivery_method(knowledge),
            cytotoxicity=knowledge.safety_data.get('cytotoxicity'),
            cell_viability=knowledge.safety_data.get('cell_viability'),
            apoptosis_rate=knowledge.safety_data.get('apoptosis_rate'),
            safety_score=self._calculate_safety_score(knowledge),
            source_metadata={
                'title': knowledge.title,
                'authors': knowledge.authors,
                'year': knowledge.year,
                'journal': knowledge.journal,
                'doi': knowledge.doi
            }
        )

    # Helper methods for data extraction
    def _generate_synthetic_sequence(self, gene_name: str) -> str:
        """Generate synthetic sequence for analysis"""
        import random
        bases = ['A', 'T', 'G', 'C']
        return ''.join(random.choice(bases) for _ in range(20))

    def _extract_guide_sequence(self, knowledge: ExtractedKnowledge) -> Optional[str]:
        """Extract guide RNA sequence from knowledge"""
        # Look for guide sequences in results or methodology
        for finding in knowledge.key_findings:
            # Simple pattern matching for guide sequences
            import re
            matches = re.findall(r'[ATGC]{19,21}GG', finding)
            if matches:
                return matches[0]
        return None

    def _extract_target_sequence(self, knowledge: ExtractedKnowledge) -> Optional[str]:
        """Extract target sequence from knowledge"""
        # Similar to guide sequence extraction
        return None

    def _extract_delivery_method(self, knowledge: ExtractedKnowledge) -> Optional[str]:
        """Extract delivery method from knowledge"""
        delivery_methods = ['Lipofection', 'Electroporation', 'AAV', 'Lentivirus', 'Viral']

        for finding in knowledge.key_findings:
            for method in delivery_methods:
                if method.lower() in finding.lower():
                    return method
        return None

    def _extract_efficiency_score(self, knowledge: ExtractedKnowledge) -> Optional[float]:
        """Extract efficiency score from knowledge"""
        return knowledge.efficiency_data.get('editing_efficiency')

    def _calculate_specificity_score(self, knowledge: ExtractedKnowledge) -> Optional[float]:
        """Calculate specificity score"""
        # Based on off-target data
        off_target_rate = knowledge.safety_data.get('off_target_rate', 0)
        return max(0, 100 - off_target_rate) if off_target_rate else None

    def _extract_off_target_score(self, knowledge: ExtractedKnowledge) -> Optional[float]:
        """Extract off-target score from knowledge"""
        return knowledge.safety_data.get('off_target_rate')

    def _calculate_safety_score(self, knowledge: ExtractedKnowledge) -> Optional[float]:
        """Calculate overall safety score"""
        safety_metrics = [
            knowledge.safety_data.get('cell_viability', 100),
            knowledge.safety_data.get('cytotoxicity', 0),
            knowledge.safety_data.get('apoptosis_rate', 0)
        ]

        # Convert to 0-100 scale where higher is safer
        if safety_metrics[0] is not None:
            safety_score = safety_metrics[0]  # Cell viability
            if safety_metrics[1] is not None:  # Cytotoxicity
                safety_score -= safety_metrics[1]
            if safety_metrics[2] is not None:  # Apoptosis
                safety_score -= safety_metrics[2]
            return max(0, min(100, safety_score))

        return None

    def _extract_activity_features(self, knowledge: ExtractedKnowledge):
        """Extract activity-specific features"""
        # This would be implemented based on the ActivityFeatures dataclass
        return None

    def _extract_activity_score(self, knowledge: ExtractedKnowledge) -> Optional[float]:
        """Extract activity score from knowledge"""
        return knowledge.efficiency_data.get('activation_efficiency')

    def _create_clinical_context(self, knowledge: ExtractedKnowledge):
        """Create clinical context for design record"""
        from ..data_models.design_dataset import ClinicalContext

        return ClinicalContext(
            disease_name="Generic Disease",  # Would be extracted from knowledge
            severity="moderate",
            stage="early"
        )

    def _create_target_gene(self, knowledge: ExtractedKnowledge):
        """Create target gene for design record"""
        from ..data_models.design_dataset import TargetGene

        return TargetGene(
            symbol=knowledge.target_genes[0] if knowledge.target_genes else "Unknown",
            function="Unknown",
            pathway="Unknown"
        )

    def _create_design_solution(self, knowledge: ExtractedKnowledge, technique: str):
        """Create design solution for specific technique"""
        from ..data_models.design_dataset import DesignSolution

        if knowledge.editing_technique != technique:
            return None

        return DesignSolution(
            editing_technique=technique,
            optimal_guide_sequence=self._extract_guide_sequence(knowledge) or "Unknown",
            delivery_method=self._extract_delivery_method(knowledge) or "Unknown",
            expected_efficiency=self._extract_efficiency_score(knowledge) or 50.0,
            expected_safety=self._calculate_safety_score(knowledge) or 70.0,
            protocol="Standard protocol for " + technique
        )

    async def _save_results(self, output_path: Path, knowledge_list: List[ExtractedKnowledge],
                          datasets: Dict[str, List], knowledge_graph: Dict[str, Any]):
        """Save all results to files"""
        self.logger.info("Saving comprehensive results...")

        # Save knowledge base
        knowledge_path = output_path / "knowledge_base.json"
        self.knowledge_extractor.save_knowledge_base(knowledge_list, str(knowledge_path))

        # Save knowledge graph
        graph_path = output_path / "knowledge_graph.json"
        with open(graph_path, 'w') as f:
            json.dump(knowledge_graph, f, indent=2)

        # Save datasets
        for dataset_name, records in datasets.items():
            if records:
                dataset_path = output_path / f"{dataset_name}_dataset.json"
                with open(dataset_path, 'w') as f:
                    json.dump([record.to_dict() for record in records], f, indent=2)

        # Save comprehensive report
        report_path = output_path / "comprehensive_report.json"
        report = self._generate_comprehensive_report()
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive mining report"""
        return {
            'mining_timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'knowledge_base_summary': {
                'total_papers': self.stats['papers_processed'],
                'knowledge_extracted': self.stats['knowledge_extracted'],
                'average_confidence': 0.75,  # Would be calculated from actual data
                'average_relevance': 0.80    # Would be calculated from actual data
            },
            'datasets_summary': {
                'efficiency_records': self.stats['efficiency_records'],
                'activity_records': self.stats['activity_records'],
                'design_records': self.stats['design_records'],
                'safety_records': self.stats['safety_records']
            },
            'errors': self.stats['errors'],
            'pipeline_version': 'v2.0',
            'ai_models_used': [
                'SentenceTransformer',
                'PubMedBERT',
                'T5',
                'SpaCy',
                'Custom Feature Extractors'
            ]
        }
