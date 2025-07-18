"""
Comprehensive AI Pipeline for GeneX Phase 1
Enhanced with Research Report 3/3 Recommendations

This pipeline integrates all components including the new confidence scoring,
patent integration, clinical trial integration, and continuous improvement
mechanisms recommended in the third research report.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio
import aiohttp
from dataclasses import asdict

# Import existing modules
from ..api_clients.pubmed_client import PubMedClient
from ..api_clients.semantic_scholar_client import SemanticScholarClient
from ..api_clients.europe_pmc_client import EuropePMCClient
from ..data_models.efficiency_dataset import EfficiencyRecord
from ..data_models.activity_dataset import ActivityRecord
from ..data_models.design_dataset import DesignRecord
from ..data_models.safety_dataset import SafetyRecord
from ..feature_extraction.sequence_features import SequenceFeatureExtractor
from ..feature_extraction.biological_features import BiologicalFeatureExtractor
from ..feature_extraction.experimental_features import ExperimentalFeatureExtractor
from ..mining.comprehensive_ai_miner import ComprehensiveAIMiner
from ..utils.config import load_config
from ..utils.hierarchical_rate_limiter import HierarchicalRateLimiter

# Import new modules from Research Report 3/3
from .confidence_scoring import ConfidenceScoringFramework, ExtractedFact, SourceType
from .patent_integration import PatentIntegrationPipeline, PatentSource
from .clinical_trial_integration import ClinicalTrialIntegrationPipeline
from .continuous_improvement import ContinuousImprovementPipeline, UserFeedback, FeedbackType

logger = logging.getLogger(__name__)


class EnhancedComprehensiveAIPipeline:
    """
    Enhanced comprehensive AI pipeline incorporating all Research Report 3/3 recommendations.

    This pipeline implements:
    1. Confidence scoring and evidence tracking framework
    2. Patent data integration for commercial/translational data
    3. Clinical trial integration for human safety data
    4. Continuous improvement mechanisms with human-in-the-loop feedback
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.setup_components()
        self.setup_enhanced_components()

    def setup_components(self):
        """Setup existing pipeline components."""
        # API Clients - pass specific config sections
        self.pubmed_client = PubMedClient(self.config['apis']['pubmed'])
        self.semantic_scholar_client = SemanticScholarClient(self.config['apis']['semantic_scholar'])

        # Europe PMC client needs different initialization pattern
        europe_pmc_config = self.config['apis']['europe_pmc']
        self.europe_pmc_client = EuropePMCClient(
            api_key=europe_pmc_config.get('api_key'),
            rate_limiter=None  # Will be set up separately
        )

        # Feature Extractors
        self.sequence_extractor = SequenceFeatureExtractor()
        self.biological_extractor = BiologicalFeatureExtractor()
        self.experimental_extractor = ExperimentalFeatureExtractor()

        # AI Miner
        self.ai_miner = ComprehensiveAIMiner(config_path="config/config.yaml")

        # Rate Limiter
        api_configs = {
            'pubmed': self.config['apis']['pubmed'],
            'semantic_scholar': self.config['apis']['semantic_scholar'],
            'europe_pmc': self.config['apis']['europe_pmc']
        }
        self.rate_limiter = HierarchicalRateLimiter(api_configs)

        # Data storage
        self.results_dir = Path(self.config['output']['base_dir'])
        self.results_dir.mkdir(exist_ok=True)

    def setup_enhanced_components(self):
        """Setup new components from Research Report 3/3."""
        # Confidence scoring framework
        self.confidence_framework = ConfidenceScoringFramework(self.config)

        # Patent integration pipeline
        self.patent_pipeline = PatentIntegrationPipeline(self.config)

        # Clinical trial integration pipeline
        self.clinical_trial_pipeline = ClinicalTrialIntegrationPipeline(self.config)

        # Continuous improvement pipeline
        self.continuous_improvement = ContinuousImprovementPipeline(self.config)

    async def run_comprehensive_mining(self, search_terms: List[str],
                                     max_results: int = 1000) -> Dict[str, Any]:
        """
        Run comprehensive mining with all enhanced features.

        Args:
            search_terms: List of search terms for literature mining
            max_results: Maximum number of results to process

        Returns:
            Comprehensive results including academic, patent, and clinical trial data
        """
        logger.info("Starting enhanced comprehensive mining pipeline...")

        results = {
            'academic_literature': {},
            'patent_data': {},
            'clinical_trial_data': {},
            'integrated_knowledge': {},
            'confidence_metrics': {},
            'continuous_improvement_status': {},
            'training_datasets': {},  # NEW: Training datasets for 3 domains
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Create aiohttp session for API clients
            async with aiohttp.ClientSession() as session:
                # Set session for API clients
                self.pubmed_client.session = session
                self.semantic_scholar_client.session = session
                self.europe_pmc_client.session = session

                # Step 1: Academic Literature Mining with Confidence Scoring
                logger.info("Step 1: Mining academic literature with confidence scoring...")
                academic_results = await self._mine_academic_literature(search_terms, max_results)
                results['academic_literature'] = academic_results

                # Step 2: Patent Data Integration
                logger.info("Step 2: Integrating patent data...")
                patent_results = await self._integrate_patent_data()
                results['patent_data'] = patent_results

                # Step 3: Clinical Trial Data Integration
                logger.info("Step 3: Integrating clinical trial data...")
                clinical_results = await self._integrate_clinical_trial_data()
                results['clinical_trial_data'] = clinical_results

                # Step 4: Knowledge Integration and Conflict Resolution
                logger.info("Step 4: Integrating knowledge and resolving conflicts...")
                integrated_results = await self._integrate_knowledge_sources(
                    academic_results, patent_results, clinical_results
                )
                results['integrated_knowledge'] = integrated_results

                # Step 5: Generate Training Datasets for 3 Domains
                logger.info("Step 5: Generating training datasets for CRISPR, Prime, and Base editing...")
                training_datasets = await self._generate_training_datasets(academic_results['papers'])
                results['training_datasets'] = training_datasets

                # Step 6: Confidence Analysis
                logger.info("Step 6: Analyzing confidence metrics...")
                confidence_metrics = self._analyze_confidence_metrics(integrated_results)
                results['confidence_metrics'] = confidence_metrics

                # Step 7: Continuous Improvement Monitoring
                logger.info("Step 7: Monitoring continuous improvement...")
                improvement_status = self.continuous_improvement.monitor_and_update()
                results['continuous_improvement_status'] = improvement_status

                # Save comprehensive results
                self._save_comprehensive_results(results)

                logger.info("Enhanced comprehensive mining completed successfully!")
                return results

        except Exception as e:
            logger.error(f"Error in comprehensive mining: {e}")
            results['error'] = str(e)
            return results

    async def _mine_academic_literature(self, search_terms: List[str],
                                      max_results: int) -> Dict[str, Any]:
        """Mine academic literature with enhanced confidence scoring."""
        academic_results = {
            'papers': [],
            'extracted_facts': [],
            'confidence_scores': {},
            'conflicts_detected': []
        }

        # Search across multiple sources
        all_papers = []

        # PubMed search
        for term in search_terms:
            try:
                response = await self.pubmed_client.search_papers(term, max_results // len(search_terms))
                if response.success and response.data:
                    # Extract PMIDs from response
                    pmid_list = self.pubmed_client.parse_search_results(response)
                    if pmid_list:
                        # Get paper details for first few PMIDs
                        details_response = await self.pubmed_client.get_papers_batch(pmid_list[:5])
                        if details_response.success and details_response.data:
                            papers = self.pubmed_client.parse_paper_details(details_response)
                            all_papers.extend(papers)
            except Exception as e:
                logger.warning(f"PubMed search failed for '{term}': {e}")

        # Semantic Scholar search
        for term in search_terms:
            try:
                papers = await self.semantic_scholar_client.search_papers(term, max_results // len(search_terms))
                if isinstance(papers, list):
                    all_papers.extend(papers)
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed for '{term}': {e}")

        # Europe PMC search
        for term in search_terms:
            try:
                papers = await self.europe_pmc_client.search_papers(term, max_results // len(search_terms))
                if isinstance(papers, list):
                    all_papers.extend(papers)
            except Exception as e:
                logger.warning(f"Europe PMC search failed for '{term}': {e}")

        academic_results['papers'] = all_papers[:max_results]

        # Extract facts with confidence scoring
        for paper in academic_results['papers']:
            try:
                # Create extracted fact with confidence scoring
                fact = ExtractedFact(
                    subject=paper.get('title', 'Unknown'),
                    predicate='MENTIONS_GENE_EDITING',
                    object=paper.get('abstract', '')[:200] + '...',
                    source=SourceType.ACADEMIC_LITERATURE,
                    confidence=0.8,  # Default confidence
                    evidence=[paper.get('doi', ''), paper.get('pmid', '')],
                    timestamp=datetime.now().isoformat()
                )

                academic_results['extracted_facts'].append(fact)

                # Store confidence score
                academic_results['confidence_scores'][fact.fact_id] = fact.confidence

            except Exception as e:
                logger.warning(f"Failed to extract facts from paper: {e}")

        return academic_results

    async def _integrate_patent_data(self) -> Dict[str, Any]:
        """Integrate patent data using the patent integration pipeline."""
        patent_results = {
            'patents_found': 0,
            'patent_documents': [],
            'patent_facts': [],
            'academic_links': []
        }

        try:
            # Search for gene editing patents
            patents = self.patent_pipeline.search_gene_editing_patents(max_results=500)
            patent_results['patents_found'] = len(patents)

            # Process patent documents
            for patent_data in patents[:100]:  # Process subset for efficiency
                patent_doc = self.patent_pipeline.process_patent_document(patent_data)
                if patent_doc:
                    patent_results['patent_documents'].append(patent_doc)

                    # Create patent facts
                    patent_facts = self.patent_pipeline._create_patent_facts(patent_doc)
                    patent_results['patent_facts'].extend(patent_facts)

        except Exception as e:
            logger.error(f"Error integrating patent data: {e}")

        return patent_results

    async def _integrate_clinical_trial_data(self) -> Dict[str, Any]:
        """Integrate clinical trial data using the clinical trial integration pipeline."""
        clinical_results = {
            'trials_found': 0,
            'trial_documents': [],
            'trial_facts': [],
            'academic_links': []
        }

        try:
            # Search for gene editing clinical trials
            trials = self.clinical_trial_pipeline.search_gene_editing_trials(max_results=200)
            clinical_results['trials_found'] = len(trials)

            # Process trial documents
            for trial_data in trials[:50]:  # Process subset for efficiency
                trial_doc = self.clinical_trial_pipeline.process_trial_document(trial_data)
                if trial_doc:
                    clinical_results['trial_documents'].append(trial_doc)

                    # Create trial facts
                    trial_facts = self.clinical_trial_pipeline._create_trial_facts(trial_doc)
                    clinical_results['trial_facts'].extend(trial_facts)

        except Exception as e:
            logger.error(f"Error integrating clinical trial data: {e}")

        return clinical_results

    async def _integrate_knowledge_sources(self, academic_results: Dict[str, Any],
                                         patent_results: Dict[str, Any],
                                         clinical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate knowledge from all sources and resolve conflicts."""
        integrated_results = {
            'all_facts': [],
            'resolved_conflicts': [],
            'knowledge_graph_ready': False,
            'integration_metrics': {}
        }

        # Collect all facts
        all_facts = []

        # Academic facts
        for fact in academic_results.get('extracted_facts', []):
            all_facts.append({
                'fact': self.confidence_framework.to_dict(fact),
                'source': 'academic',
                'confidence': fact.confidence_score.overall_confidence
            })

        # Patent facts
        for fact in patent_results.get('patent_facts', []):
            all_facts.append({
                'fact': fact,
                'source': 'patent',
                'confidence': 0.7  # Default confidence for patent facts
            })

        # Clinical trial facts
        for fact in clinical_results.get('trial_facts', []):
            all_facts.append({
                'fact': fact,
                'source': 'clinical_trial',
                'confidence': 0.8  # Default confidence for clinical trial facts
            })

        integrated_results['all_facts'] = all_facts

        # Resolve conflicts
        resolved_conflicts = self._resolve_all_conflicts(all_facts)
        integrated_results['resolved_conflicts'] = resolved_conflicts

        # Calculate integration metrics
        integration_metrics = {
            'total_facts': len(all_facts),
            'academic_facts': len(academic_results.get('extracted_facts', [])),
            'patent_facts': len(patent_results.get('patent_facts', [])),
            'clinical_facts': len(clinical_results.get('trial_facts', [])),
            'conflicts_resolved': len(resolved_conflicts),
            'average_confidence': sum(f['confidence'] for f in all_facts) / len(all_facts) if all_facts else 0
        }
        integrated_results['integration_metrics'] = integration_metrics

        # Mark as ready for knowledge graph
        integrated_results['knowledge_graph_ready'] = True

        return integrated_results

    def _resolve_all_conflicts(self, all_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts across all knowledge sources."""
        resolved_conflicts = []

        # Group facts by subject-predicate pairs
        fact_groups = {}
        for fact_data in all_facts:
            fact = fact_data['fact']
            key = f"{fact.get('subject', '')}|{fact.get('predicate', '')}"

            if key not in fact_groups:
                fact_groups[key] = []
            fact_groups[key].append(fact_data)

        # Resolve conflicts within each group
        for key, facts in fact_groups.items():
            if len(facts) > 1:
                # Check for conflicts
                conflicting_facts = []
                for i, fact1 in enumerate(facts):
                    for j, fact2 in enumerate(facts[i+1:], i+1):
                        if self._facts_conflict(fact1['fact'], fact2['fact']):
                            conflicting_facts.extend([fact1, fact2])

                if conflicting_facts:
                    resolution = self._resolve_fact_conflicts(conflicting_facts)
                    resolved_conflicts.append({
                        'conflict_key': key,
                        'conflicting_facts': conflicting_facts,
                        'resolution': resolution
                    })

        return resolved_conflicts

    def _facts_conflict(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> bool:
        """Check if two facts conflict."""
        # Simple conflict detection - could be enhanced
        if (fact1.get('subject') == fact2.get('subject') and
            fact1.get('predicate') == fact2.get('predicate')):

            obj1, obj2 = fact1.get('object', ''), fact2.get('object', '')

            # Check for numeric conflicts
            try:
                import re
                num1 = re.findall(r'(\d+(?:\.\d+)?)', obj1)
                num2 = re.findall(r'(\d+(?:\.\d+)?)', obj2)

                if num1 and num2:
                    val1, val2 = float(num1[0]), float(num2[0])
                    return abs(val1 - val2) > 20  # 20% difference threshold

            except (ValueError, IndexError):
                pass

            # Check for opposite qualitative terms
            positive_terms = ['high', 'efficient', 'successful', 'good']
            negative_terms = ['low', 'inefficient', 'unsuccessful', 'poor']

            obj1_lower, obj2_lower = obj1.lower(), obj2.lower()

            obj1_positive = any(term in obj1_lower for term in positive_terms)
            obj1_negative = any(term in obj1_lower for term in negative_terms)
            obj2_positive = any(term in obj2_lower for term in positive_terms)
            obj2_negative = any(term in obj2_lower for term in negative_terms)

            return (obj1_positive and obj2_negative) or (obj1_negative and obj2_positive)

        return False

    def _resolve_fact_conflicts(self, conflicting_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts between facts."""
        # Sort by confidence and source reliability
        def sort_key(fact_data):
            confidence = fact_data['confidence']
            source = fact_data['source']

            # Source reliability weights
            source_weights = {
                'academic': 1.0,
                'clinical_trial': 0.9,
                'patent': 0.7
            }

            return confidence * source_weights.get(source, 0.5)

        sorted_facts = sorted(conflicting_facts, key=sort_key, reverse=True)

        return {
            'recommended_fact': sorted_facts[0],
            'all_facts': sorted_facts,
            'resolution_strategy': 'confidence_based',
            'confidence_difference': sorted_facts[0]['confidence'] - sorted_facts[1]['confidence'] if len(sorted_facts) > 1 else 0
        }

    def _analyze_confidence_metrics(self, integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence metrics across all integrated knowledge."""
        confidence_metrics = {
            'total_facts': 0,
            'average_confidence': 0.0,
            'confidence_distribution': {},
            'high_confidence_facts': 0,
            'medium_confidence_facts': 0,
            'low_confidence_facts': 0,
            'source_confidence': {},
            'fact_type_confidence': {}
        }

        try:
            # Extract facts from integrated results
            all_facts = []

            # Academic literature facts
            academic_facts = integrated_results.get('academic_facts', [])
            all_facts.extend(academic_facts)

            # Patent facts
            patent_facts = integrated_results.get('patent_facts', [])
            all_facts.extend(patent_facts)

            # Clinical trial facts
            clinical_facts = integrated_results.get('clinical_facts', [])
            all_facts.extend(clinical_facts)

            confidence_metrics['total_facts'] = len(all_facts)

            if all_facts:
                # Calculate average confidence
                total_confidence = sum(fact.get('confidence', 0.0) for fact in all_facts)
                confidence_metrics['average_confidence'] = total_confidence / len(all_facts)

                # Analyze confidence distribution
                for fact in all_facts:
                    confidence = fact.get('confidence', 0.0)
                    if confidence >= 0.8:
                        confidence_metrics['high_confidence_facts'] += 1
                    elif confidence >= 0.6:
                        confidence_metrics['medium_confidence_facts'] += 1
                    else:
                        confidence_metrics['low_confidence_facts'] += 1

                    # Source confidence analysis
                    source = fact.get('source', 'unknown')
                    if source not in confidence_metrics['source_confidence']:
                        confidence_metrics['source_confidence'][source] = []
                    confidence_metrics['source_confidence'][source].append(confidence)

                    # Fact type confidence analysis
                    fact_type = fact.get('fact_type', 'general')
                    if fact_type not in confidence_metrics['fact_type_confidence']:
                        confidence_metrics['fact_type_confidence'][fact_type] = []
                    confidence_metrics['fact_type_confidence'][fact_type].append(confidence)

                # Calculate average confidence by source
                for source, confidences in confidence_metrics['source_confidence'].items():
                    confidence_metrics['source_confidence'][source] = sum(confidences) / len(confidences)

                # Calculate average confidence by fact type
                for fact_type, confidences in confidence_metrics['fact_type_confidence'].items():
                    confidence_metrics['fact_type_confidence'][fact_type] = sum(confidences) / len(confidences)

        except Exception as e:
            logger.error(f"Error analyzing confidence metrics: {e}")
            # Ensure total_facts is always present
            confidence_metrics['total_facts'] = 0

        return confidence_metrics

    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main results
        results_file = self.results_dir / f"comprehensive_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save individual components
        components_dir = self.results_dir / "components" / timestamp
        components_dir.mkdir(parents=True, exist_ok=True)

        # Academic literature
        academic_file = components_dir / "academic_literature.json"
        with open(academic_file, 'w') as f:
            json.dump(results['academic_literature'], f, indent=2, default=str)

        # Patent data
        patent_file = components_dir / "patent_data.json"
        with open(patent_file, 'w') as f:
            json.dump(results['patent_data'], f, indent=2, default=str)

        # Clinical trial data
        clinical_file = components_dir / "clinical_trial_data.json"
        with open(clinical_file, 'w') as f:
            json.dump(results['clinical_trial_data'], f, indent=2, default=str)

        # Integrated knowledge
        integrated_file = components_dir / "integrated_knowledge.json"
        with open(integrated_file, 'w') as f:
            json.dump(results['integrated_knowledge'], f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")

    def process_user_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user feedback for continuous improvement.

        Args:
            feedback_data: User feedback data

        Returns:
            Processing result
        """
        try:
            # Create UserFeedback object
            feedback = UserFeedback(
                feedback_id=feedback_data.get('feedback_id', f"fb_{datetime.now().timestamp()}"),
                user_id=feedback_data.get('user_id', 'anonymous'),
                feedback_type=FeedbackType(feedback_data.get('feedback_type', 'entity_correction')),
                fact_id=feedback_data.get('fact_id', ''),
                original_fact=feedback_data.get('original_fact', {}),
                suggested_correction=feedback_data.get('suggested_correction', {}),
                feedback_text=feedback_data.get('feedback_text', ''),
                confidence_score=feedback_data.get('confidence_score', 0.8),
                timestamp=datetime.now().isoformat(),
                status=FeedbackStatus.PENDING
            )

            # Process feedback
            result = self.continuous_improvement.process_user_feedback(feedback)

            return result

        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
            return {'error': str(e)}

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        return self.continuous_improvement.get_system_health_metrics()

    async def _generate_training_datasets(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive training datasets for the 3 domains (CRISPR, Prime, Base editing)
        using real data extracted from papers.

        Args:
            papers: List of papers with real data from API calls

        Returns:
            Dictionary containing training datasets for each domain
        """
        logger.info(f"Generating training datasets from {len(papers)} papers...")

        training_datasets = {
            'crispr': {
                'efficiency_records': [],
                'activity_records': [],
                'design_records': [],
                'safety_records': [],
                'statistics': {}
            },
            'prime_editing': {
                'efficiency_records': [],
                'activity_records': [],
                'design_records': [],
                'safety_records': [],
                'statistics': {}
            },
            'base_editing': {
                'efficiency_records': [],
                'activity_records': [],
                'design_records': [],
                'safety_records': [],
                'statistics': {}
            },
            'overall_statistics': {
                'total_papers_processed': len(papers),
                'papers_with_gene_editing': 0,
                'papers_with_crispr': 0,
                'papers_with_prime_editing': 0,
                'papers_with_base_editing': 0,
                'total_training_records': 0
            }
        }

        for paper in papers:
            try:
                # Extract real data from paper
                paper_data = self._extract_real_paper_data(paper)

                if not paper_data['is_gene_editing_paper']:
                    continue

                training_datasets['overall_statistics']['papers_with_gene_editing'] += 1

                # Process for each domain based on detected techniques
                if paper_data['techniques']['crispr']:
                    training_datasets['overall_statistics']['papers_with_crispr'] += 1
                    crispr_records = self._create_domain_training_records(paper_data, 'CRISPR')
                    self._add_records_to_domain(training_datasets['crispr'], crispr_records)

                if paper_data['techniques']['prime_editing']:
                    training_datasets['overall_statistics']['papers_with_prime_editing'] += 1
                    prime_records = self._create_domain_training_records(paper_data, 'Prime Editing')
                    self._add_records_to_domain(training_datasets['prime_editing'], prime_records)

                if paper_data['techniques']['base_editing']:
                    training_datasets['overall_statistics']['papers_with_base_editing'] += 1
                    base_records = self._create_domain_training_records(paper_data, 'Base Editing')
                    self._add_records_to_domain(training_datasets['base_editing'], base_records)

            except Exception as e:
                logger.warning(f"Error processing paper {paper.get('pmid', 'unknown')}: {e}")
                continue

        # Calculate statistics for each domain
        for domain in ['crispr', 'prime_editing', 'base_editing']:
            domain_data = training_datasets[domain]
            domain_data['statistics'] = {
                'efficiency_records': len(domain_data['efficiency_records']),
                'activity_records': len(domain_data['activity_records']),
                'design_records': len(domain_data['design_records']),
                'safety_records': len(domain_data['safety_records']),
                'total_records': (len(domain_data['efficiency_records']) +
                                len(domain_data['activity_records']) +
                                len(domain_data['design_records']) +
                                len(domain_data['safety_records']))
            }
            training_datasets['overall_statistics']['total_training_records'] += domain_data['statistics']['total_records']

        logger.info(f"Generated {training_datasets['overall_statistics']['total_training_records']} training records")
        return training_datasets

    def _extract_real_paper_data(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract real data from a paper for training dataset generation.
        Uses NLP and pattern matching to extract structured information.
        """
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        full_text = f"{title} {abstract}"

        # Detect gene editing techniques using real patterns
        techniques = {
            'crispr': any(term in full_text for term in [
                'crispr', 'cas9', 'cas12', 'cas13', 'guide rna', 'grna', 'crrna',
                'tracrrna', 'pam sequence', 'double-strand break', 'dsb'
            ]),
            'prime_editing': any(term in full_text for term in [
                'prime editing', 'prime editor', 'peg', 'peg-rna', 'reverse transcriptase',
                'nickase', 'single-strand break', 'ssb'
            ]),
            'base_editing': any(term in full_text for term in [
                'base editing', 'base editor', 'cytosine base editor', 'cbe',
                'adenine base editor', 'abe', 'deaminase', 'nickase'
            ])
        }

        # Extract target genes using real patterns
        target_genes = self._extract_target_genes(full_text)

        # Extract efficiency data using real patterns
        efficiency_data = self._extract_efficiency_data(full_text)

        # Extract experimental conditions
        experimental_conditions = self._extract_experimental_conditions(full_text)

        # Extract safety data
        safety_data = self._extract_safety_data(full_text)

        return {
            'paper_id': paper.get('pmid', paper.get('doi', 'unknown')),
            'title': paper.get('title', ''),
            'abstract': paper.get('abstract', ''),
            'authors': paper.get('authors', []),
            'journal': paper.get('journal', ''),
            'year': paper.get('year', 2025),
            'doi': paper.get('doi', ''),
            'is_gene_editing_paper': any(techniques.values()),
            'techniques': techniques,
            'target_genes': target_genes,
            'efficiency_data': efficiency_data,
            'experimental_conditions': experimental_conditions,
            'safety_data': safety_data
        }

    def _extract_target_genes(self, text: str) -> List[str]:
        """Extract target gene names from text using real patterns."""
        import re

        # Common gene name patterns
        gene_patterns = [
            r'\b[A-Z][A-Z0-9]{2,}\b',  # All caps gene symbols like BRCA1, TP53
            r'\b[A-Z][a-z]+[0-9]*\b',  # Mixed case like Cas9, Cas12
            r'\b[A-Z][a-z]+[A-Z][a-z]+\b',  # Mixed case like CRISPR, Cas9
        ]

        genes = set()
        for pattern in gene_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Filter out common non-gene words
                if len(match) >= 3 and match not in ['the', 'and', 'for', 'with', 'this', 'that']:
                    genes.add(match)

        return list(genes)[:10]  # Limit to top 10

    def _extract_efficiency_data(self, text: str) -> Dict[str, Any]:
        """Extract efficiency data from text using real patterns."""
        import re

        efficiency_data = {
            'efficiency_score': None,
            'specificity_score': None,
            'off_target_score': None,
            'activity_score': None
        }

        # Efficiency patterns (percentage, decimal, ratio)
        efficiency_patterns = [
            r'(\d+(?:\.\d+)?)\s*%?\s*(?:efficiency|editing\s+efficiency|targeting\s+efficiency)',
            r'efficiency\s*(?:of|:)\s*(\d+(?:\.\d+)?)\s*%?',
            r'(\d+(?:\.\d+)?)\s*%?\s*(?:success|rate|frequency)',
        ]

        for pattern in efficiency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    efficiency_data['efficiency_score'] = float(matches[0]) / 100.0
                    break
                except ValueError:
                    continue

        # Specificity patterns
        specificity_patterns = [
            r'(\d+(?:\.\d+)?)\s*%?\s*specificity',
            r'specificity\s*(?:of|:)\s*(\d+(?:\.\d+)?)\s*%?',
        ]

        for pattern in specificity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    efficiency_data['specificity_score'] = float(matches[0]) / 100.0
                    break
                except ValueError:
                    continue

        return efficiency_data

    def _extract_experimental_conditions(self, text: str) -> Dict[str, Any]:
        """Extract experimental conditions from text."""
        conditions = {
            'cell_line': None,
            'organism': None,
            'delivery_method': None,
            'cell_type': None
        }

        # Cell line patterns
        cell_patterns = [
            r'\b(HEK293|HeLa|Jurkat|K562|A549|MCF7|HCT116|U2OS|HUVEC|iPSC)\b',
            r'\b(human|mouse|rat|zebrafish|drosophila|yeast|bacteria)\b',
        ]

        for pattern in cell_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if not conditions['cell_line']:
                    conditions['cell_line'] = matches[0]
                if not conditions['organism'] and matches[0].lower() in ['human', 'mouse', 'rat', 'zebrafish', 'drosophila', 'yeast', 'bacteria']:
                    conditions['organism'] = matches[0]

        # Delivery method patterns
        delivery_patterns = [
            r'\b(lipofection|electroporation|viral|adenovirus|lentivirus|retrovirus|AAV|plasmid|mRNA|protein)\b',
        ]

        for pattern in delivery_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                conditions['delivery_method'] = matches[0]
                break

        return conditions

    def _extract_safety_data(self, text: str) -> Dict[str, Any]:
        """Extract safety data from text."""
        safety_data = {
            'cytotoxicity': None,
            'cell_viability': None,
            'apoptosis_rate': None,
            'safety_score': None
        }

        # Cell viability patterns
        viability_patterns = [
            r'(\d+(?:\.\d+)?)\s*%?\s*(?:viability|survival)',
            r'viability\s*(?:of|:)\s*(\d+(?:\.\d+)?)\s*%?',
        ]

        for pattern in viability_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    safety_data['cell_viability'] = float(matches[0]) / 100.0
                    break
                except ValueError:
                    continue

        return safety_data

    def _create_domain_training_records(self, paper_data: Dict[str, Any], technique: str) -> Dict[str, List]:
        """Create training records for a specific domain."""
        records = {
            'efficiency_records': [],
            'activity_records': [],
            'design_records': [],
            'safety_records': []
        }

        # Create Efficiency Record
        if paper_data['efficiency_data']['efficiency_score'] is not None:
            efficiency_record = EfficiencyRecord(
                persistent_id=paper_data['paper_id'],
                editing_technique=technique,
                guide_rna_sequence=self._extract_guide_sequence(paper_data),
                target_sequence=self._extract_target_sequence(paper_data),
                target_gene_symbol=paper_data['target_genes'][0] if paper_data['target_genes'] else None,
                cell_line=paper_data['experimental_conditions']['cell_line'] or "Unknown",
                organism=paper_data['experimental_conditions']['organism'] or "Human",
                delivery_method=paper_data['experimental_conditions']['delivery_method'] or "Unknown",
                efficiency_score=paper_data['efficiency_data']['efficiency_score'],
                specificity_score=paper_data['efficiency_data']['specificity_score'],
                off_target_score=paper_data['efficiency_data']['off_target_score'],
                safety_score=paper_data['safety_data']['safety_score'],
                source_metadata={
                    'title': paper_data['title'],
                    'authors': [f"{a.get('first_name', '')} {a.get('last_name', '')}" for a in paper_data['authors']],
                    'year': paper_data['year'],
                    'journal': paper_data['journal'],
                    'doi': paper_data['doi']
                },
                provenance={
                    'extraction_method': 'Real Data Extraction',
                    'confidence': 0.8,
                    'relevance': 0.9
                }
            )
            records['efficiency_records'].append(efficiency_record)

        # Create Activity Record (extends efficiency)
        if records['efficiency_records']:
            activity_record = ActivityRecord(
                persistent_id=paper_data['paper_id'],
                editing_technique=technique,
                guide_rna_sequence=records['efficiency_records'][0].guide_rna_sequence,
                target_sequence=records['efficiency_records'][0].target_sequence,
                target_gene_symbol=records['efficiency_records'][0].target_gene_symbol,
                cell_line=records['efficiency_records'][0].cell_line,
                organism=records['efficiency_records'][0].organism,
                delivery_method=records['efficiency_records'][0].delivery_method,
                efficiency_score=records['efficiency_records'][0].efficiency_score,
                activity_score=paper_data['efficiency_data']['activity_score'],
                specificity_score=records['efficiency_records'][0].specificity_score,
                off_target_score=records['efficiency_records'][0].off_target_score,
                safety_score=records['efficiency_records'][0].safety_score,
                source_metadata=records['efficiency_records'][0].source_metadata,
                provenance=records['efficiency_records'][0].provenance,
                activity_features=self._create_activity_features(paper_data)
            )
            records['activity_records'].append(activity_record)

        # Create Design Record
        if paper_data['target_genes']:
            design_record = DesignRecord(
                persistent_id=paper_data['paper_id'],
                clinical_context=self._create_clinical_context(paper_data),
                target_gene=self._create_target_gene(paper_data),
                organism=paper_data['experimental_conditions']['organism'] or "Human",
                cell_line=paper_data['experimental_conditions']['cell_line'],
                crispr_solution=self._create_design_solution(paper_data, 'CRISPR'),
                prime_solution=self._create_design_solution(paper_data, 'Prime Editing'),
                base_solution=self._create_design_solution(paper_data, 'Base Editing'),
                source_metadata={
                    'title': paper_data['title'],
                    'authors': [f"{a.get('first_name', '')} {a.get('last_name', '')}" for a in paper_data['authors']],
                    'year': paper_data['year'],
                    'journal': paper_data['journal'],
                    'doi': paper_data['doi']
                }
            )
            records['design_records'].append(design_record)

        # Create Safety Record
        if paper_data['safety_data']['cell_viability'] is not None:
            safety_record = SafetyRecord(
                persistent_id=paper_data['paper_id'],
                editing_technique=technique,
                cell_line=paper_data['experimental_conditions']['cell_line'] or "Unknown",
                organism=paper_data['experimental_conditions']['organism'] or "Human",
                delivery_method=paper_data['experimental_conditions']['delivery_method'] or "Unknown",
                cytotoxicity=paper_data['safety_data']['cytotoxicity'],
                cell_viability=paper_data['safety_data']['cell_viability'],
                apoptosis_rate=paper_data['safety_data']['apoptosis_rate'],
                safety_score=paper_data['safety_data']['safety_score'],
                source_metadata={
                    'title': paper_data['title'],
                    'authors': [f"{a.get('first_name', '')} {a.get('last_name', '')}" for a in paper_data['authors']],
                    'year': paper_data['year'],
                    'journal': paper_data['journal'],
                    'doi': paper_data['doi']
                }
            )
            records['safety_records'].append(safety_record)

        return records

    def _add_records_to_domain(self, domain_data: Dict[str, List], records: Dict[str, List]):
        """Add training records to a specific domain."""
        for record_type, record_list in records.items():
            domain_data[record_type].extend(record_list)

    def _extract_guide_sequence(self, paper_data: Dict[str, Any]) -> Optional[str]:
        """Extract guide RNA sequence from paper data."""
        # Look for guide sequences in text
        text = f"{paper_data['title']} {paper_data['abstract']}"
        import re

        # Guide sequence patterns (20-21 bp sequences)
        guide_patterns = [
            r'\b([ATGC]{20,21})\b',  # DNA sequences
            r'guide\s*(?:sequence|rna)?\s*:?\s*([ATGC]{20,21})',
            r'grna\s*:?\s*([ATGC]{20,21})',
        ]

        for pattern in guide_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]

        return None

    def _extract_target_sequence(self, paper_data: Dict[str, Any]) -> Optional[str]:
        """Extract target sequence from paper data."""
        # Similar to guide sequence extraction
        return self._extract_guide_sequence(paper_data)

    def _create_activity_features(self, paper_data: Dict[str, Any]):
        """Create activity features for activity record."""
        from ..data_models.activity_dataset import ActivityFeatures

        return ActivityFeatures(
            expression_level=None,
            functional_assay=None,
            phenotypic_effect=None,
            pathway_activation=None,
            protein_activity=None
        )

    def _create_clinical_context(self, paper_data: Dict[str, Any]):
        """Create clinical context for design record."""
        from ..data_models.design_dataset import ClinicalContext

        return ClinicalContext(
            disease_name="Unknown",
            disease_description="Extracted from literature",
            target_population="General",
            clinical_urgency="Medium",
            regulatory_considerations=[]
        )

    def _create_target_gene(self, paper_data: Dict[str, Any]):
        """Create target gene for design record."""
        from ..data_models.design_dataset import TargetGene

        return TargetGene(
            gene_symbol=paper_data['target_genes'][0] if paper_data['target_genes'] else "Unknown",
            gene_name="Extracted from literature",
            chromosome_location="Unknown",
            protein_function="Unknown",
            disease_association="Unknown"
        )

    def _create_design_solution(self, paper_data: Dict[str, Any], technique: str):
        """Create design solution for design record."""
        from ..data_models.design_dataset import DesignSolution

        return DesignSolution(
            technique_name=technique,
            guide_design="Standard design",
            delivery_strategy="Standard delivery",
            optimization_approach="Standard optimization",
            expected_efficiency=0.7,
            risk_assessment="Medium"
        )


async def main():
    """Main function to run the enhanced comprehensive AI pipeline."""
    # Initialize pipeline
    pipeline = EnhancedComprehensiveAIPipeline()

    # Define search terms based on Research Report 3/3
    search_terms = [
        "CRISPR-Cas9 efficiency",
        "prime editing safety",
        "base editing off-target",
        "gene editing delivery",
        "CRISPR clinical trial",
        "genome editing therapeutic"
    ]

    # Run comprehensive mining
    results = await pipeline.run_comprehensive_mining(search_terms, max_results=500)

    # Print summary
    print("\n=== Enhanced Comprehensive AI Pipeline Results ===")
    print(f"Academic Papers Processed: {len(results['academic_literature'].get('papers', []))}")
    print(f"Patents Found: {results['patent_data'].get('patents_found', 0)}")
    print(f"Clinical Trials Found: {results['clinical_trial_data'].get('trials_found', 0)}")
    print(f"Total Facts Extracted: {results['integrated_knowledge'].get('integration_metrics', {}).get('total_facts', 0)}")
    print(f"Conflicts Resolved: {results['integrated_knowledge'].get('integration_metrics', {}).get('conflicts_resolved', 0)}")
    print(f"Average Confidence: {results['confidence_metrics'].get('average_confidence', 0):.3f}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
