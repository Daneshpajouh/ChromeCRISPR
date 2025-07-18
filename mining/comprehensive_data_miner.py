"""
Comprehensive Gene Editing Data Miner - Enhanced Version
Implements multi-metric outcome framework and comprehensive feature engineering
Based on the technical blueprint for large-scale gene editing database construction
"""

import asyncio
import json
import logging
import time
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
import requests
import yaml

from ..api_clients.pubmed_client import PubMedClient
from ..api_clients.semantic_scholar_client import SemanticScholarClient
from ..api_clients.crossref_client import CrossRefClient
from ..api_clients.ncbi_client import NCBIClient
from ..api_clients.ensembl_client import EnsemblClient
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class MultiMetricOutcomes:
    """Multi-metric outcome framework for gene editing results"""
    # CRISPR-Cas9 (NHEJ-mediated Knockout) metrics
    indel_frequency: Optional[float] = None  # Primary metric
    on_target_off_target_ratio: Optional[float] = None  # Secondary metric
    repair_profile_spectrum: Optional[Dict[str, float]] = None  # Tertiary metric
    cell_survival_rate: Optional[float] = None  # Tertiary metric

    # Prime Editing metrics
    precise_editing_efficiency: Optional[float] = None  # Primary metric
    indel_frequency_pe: Optional[float] = None  # Secondary metric
    product_purity: Optional[float] = None  # Tertiary metric
    peg_rna_component_efficiencies: Optional[Dict[str, float]] = None  # Tertiary metric

    # Base Editing metrics
    target_base_conversion_efficiency: Optional[float] = None  # Primary metric
    product_purity_be: Optional[float] = None  # Secondary metric
    bystander_edits: Optional[float] = None  # Secondary metric
    editing_window_profile: Optional[Dict[str, float]] = None  # Tertiary metric
    indel_frequency_be: Optional[float] = None  # Tertiary metric
    incorrect_transversion_edits: Optional[float] = None  # Tertiary metric

@dataclass
class ComprehensiveFeatures:
    """Comprehensive feature set for predictive modeling"""
    # Core sequence features
    guide_sequence: str = ""
    target_sequence_40bp: str = ""
    pam_sequence: str = ""
    gc_content_guide: float = 0.0
    gc_content_seed: float = 0.0

    # Thermodynamic features
    tm_guide_dna_duplex: Optional[float] = None
    tm_peg_pbs: Optional[float] = None

    # RNA structure features (from ViennaRNA)
    grna_mfe: Optional[float] = None
    grna_ensemble_diversity: Optional[float] = None
    grna_seed_accessibility: Optional[float] = None

    # Chromatin accessibility features (from ATAC-seq)
    accessibility_score: Optional[float] = None
    accessibility_peak_dist: Optional[int] = None
    accessibility_imputed_flag: bool = False

    # Histone modification features (from ChIP-seq)
    h3k27ac_score: Optional[float] = None
    h3k4me3_score: Optional[float] = None
    h3k27me3_score: Optional[float] = None

    # DNA methylation features (from WGBS)
    cpg_methylation_200bp: Optional[float] = None

    # 3D chromatin features (from Hi-C)
    compartment_ab: Optional[str] = None
    tad_boundary_dist: Optional[int] = None

    # Biological context features
    genomic_feature: str = ""
    gene_expression_tpm: Optional[float] = None
    conservation_phastcons: Optional[float] = None

@dataclass
class EnhancedDataPoint:
    """Enhanced data point with multi-metric outcomes and comprehensive features"""
    # Core identification
    event_id: str
    editor_name: str
    organism_id: int
    cell_type_id: str
    target_gene_id: str

    # Multi-metric outcomes
    outcomes: MultiMetricOutcomes

    # Comprehensive features
    features: ComprehensiveFeatures

    # Experimental metadata
    experimental_conditions: Dict[str, Any] = field(default_factory=dict)
    measurement_method: str = ""
    sequencing_depth: Optional[int] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    # Data provenance
    source_experiment: str = ""
    data_processing_pipeline: str = ""
    imputation_flags: Dict[str, bool] = field(default_factory=dict)
    validation_status: str = "pending"

    # Timestamps
    extraction_date: str = ""
    processing_date: str = ""

class ComprehensiveDataMiner:
    """
    Enhanced comprehensive data miner implementing the technical blueprint
    - Multi-metric outcome framework
    - Comprehensive feature engineering
    - Large-scale data acquisition
    - Advanced processing pipeline
    - Missing data imputation
    """

    def __init__(self, config: Config):
        self.config = config
        self.session_id = f"enhanced_comprehensive_{int(time.time())}"

        # Initialize API clients
        self.clients = {}
        self._initialize_clients()

        # Initialize data acquisition clients
        self.data_acquisition_clients = {}
        self._initialize_data_acquisition_clients()

        # Feature extraction tools
        self.feature_extractors = {}
        self._initialize_feature_extractors()

        # Data storage
        self.data_points = {
            'crispr': [],
            'prime_editing': [],
            'base_editing': []
        }

        # Statistics and analytics
        self.stats = {
            'total_data_points': 0,
            'domain_stats': {
                'crispr': {'count': 0, 'experiments': set(), 'mvdp_count': 0},
                'prime_editing': {'count': 0, 'experiments': set(), 'mvdp_count': 0},
                'base_editing': {'count': 0, 'experiments': set(), 'mvdp_count': 0}
            },
            'processing_time': 0,
            'errors': [],
            'quality_metrics': {}
        }

        # MVDP criteria
        self.mvdp_criteria = {
            'required_fields': [
                'event_id', 'editor_name', 'organism_id', 'cell_type_id',
                'outcomes.indel_frequency', 'outcomes.precise_editing_efficiency',
                'outcomes.target_base_conversion_efficiency',
                'features.guide_sequence', 'features.target_sequence_40bp'
            ],
            'min_sequencing_depth': 1000,
            'min_quality_score': 0.7
        }

        logger.info("Enhanced Comprehensive Data Miner initialized")

    def _initialize_clients(self):
        """Initialize literature and genomic API clients"""
        try:
            self.clients['pubmed'] = PubMedClient(self.config)
            self.clients['semantic_scholar'] = SemanticScholarClient(self.config)
            self.clients['crossref'] = CrossRefClient(self.config)
            self.clients['ncbi'] = NCBIClient(self.config)
            self.clients['ensembl'] = EnsemblClient(self.config)

            logger.info(f"Initialized {len(self.clients)} API clients")

        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            raise

    def _initialize_data_acquisition_clients(self):
        """Initialize clients for large-scale data acquisition"""
        try:
            # ENCODE API client
            self.data_acquisition_clients['encode'] = {
                'base_url': 'https://www.encodeproject.org/search/',
                'rate_limit': 0.1,  # 10 req/sec
                'headers': {'accept': 'application/json'}
            }

            # UCSC Genome Browser API client
            self.data_acquisition_clients['ucsc'] = {
                'base_url': 'https://api.genome.ucsc.edu/',
                'rate_limit': 1.0,  # 1 req/sec
            }

            # SRA Toolkit configuration
            self.data_acquisition_clients['sra'] = {
                'prefetch_path': 'prefetch',
                'fasterq_dump_path': 'fasterq-dump',
                'max_concurrent_downloads': 5
            }

            logger.info(f"Initialized {len(self.data_acquisition_clients)} data acquisition clients")

        except Exception as e:
            logger.error(f"Error initializing data acquisition clients: {e}")
            raise

    def _initialize_feature_extractors(self):
        """Initialize feature extraction tools"""
        try:
            # RNA structure prediction (ViennaRNA)
            self.feature_extractors['rna_structure'] = {
                'tool': 'RNAfold',
                'available': self._check_rnafold_availability(),
                'parameters': {
                    'temperature': 37.0,
                    'no_lonely_pairs': True,
                    'no_gu': False
                }
            }

            # Sequence analysis tools
            self.feature_extractors['sequence_analysis'] = {
                'gc_content': self._calculate_gc_content,
                'melting_temperature': self._calculate_melting_temperature,
                'cpg_content': self._calculate_cpg_content
            }

            # Epigenetic feature extraction (placeholders for now)
            self.feature_extractors['epigenetic'] = {
                'chromatin_accessibility': self._extract_chromatin_accessibility,
                'histone_modifications': self._extract_histone_modifications,
                'dna_methylation': self._extract_dna_methylation,
                'chromatin_3d': self._extract_chromatin_3d
            }

            logger.info(f"Initialized {len(self.feature_extractors)} feature extractors")

        except Exception as e:
            logger.error(f"Error initializing feature extractors: {e}")
            raise

    def _check_rnafold_availability(self) -> bool:
        """Check if RNAfold is available in the system"""
        try:
            import subprocess
            result = subprocess.run(['RNAfold', '--version'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("RNAfold not available - RNA structure features will be skipped")
            return False

    async def mine_all_domains(self) -> Dict[str, Any]:
        """
        Mine data for all three gene editing domains with enhanced features
        """
        logger.info("Starting enhanced comprehensive mining for all gene editing domains")

        start_time = time.time()

        # Define domain-specific configurations
        domain_configs = {
            'crispr': {
                'search_terms': [
                    'CRISPR-Cas9 efficiency indel frequency',
                    'CRISPR-Cas9 on-target off-target ratio',
                    'CRISPR-Cas9 repair profile spectrum',
                    'CRISPR-Cas9 cell survival rate',
                    'CRISPR-Cas12 efficiency',
                    'CRISPR-Cas13 efficiency'
                ],
                'genes': ['BRCA1', 'BRCA2', 'TP53', 'CFTR', 'HBB', 'PCSK9'],
                'efficiency_metrics': ['indel_frequency', 'on_target_off_target_ratio', 'cell_survival_rate'],
                'target_data_points': 500000
            },
            'prime_editing': {
                'search_terms': [
                    'prime editing precise efficiency',
                    'PE2 precise editing efficiency',
                    'PE3 precise editing efficiency',
                    'pegRNA component efficiency',
                    'prime editing product purity',
                    'prime editing indel frequency'
                ],
                'genes': ['PCSK9', 'LDLR', 'APOB', 'BRCA1', 'TP53'],
                'efficiency_metrics': ['precise_editing_efficiency', 'product_purity', 'indel_frequency_pe'],
                'target_data_points': 500000
            },
            'base_editing': {
                'search_terms': [
                    'base editing target conversion efficiency',
                    'CBE bystander edits',
                    'ABE bystander edits',
                    'base editing product purity',
                    'base editing window profile',
                    'base editing indel frequency'
                ],
                'genes': ['APOBEC', 'TAD', 'TREX2', 'PCSK9', 'BRCA1'],
                'efficiency_metrics': ['target_base_conversion_efficiency', 'product_purity_be', 'bystander_edits'],
                'target_data_points': 500000
            }
        }

        # Process each domain
        for domain, config in domain_configs.items():
            logger.info(f"Processing domain: {domain}")
            await self._mine_domain_enhanced(domain, config)

        self.stats['processing_time'] = time.time() - start_time

        # Generate comprehensive report
        report = self._generate_enhanced_report()

        # Save results
        self._save_enhanced_results()

        logger.info(f"Enhanced comprehensive mining completed. Total data points: {self.stats['total_data_points']}")
        return report

    async def _mine_domain_enhanced(self, domain: str, config: Dict):
        """
        Enhanced domain mining with multi-metric outcomes and comprehensive features
        """
        logger.info(f"Starting enhanced mining for domain: {domain}")

        # 1. Large-scale data acquisition from multiple sources
        experiments = await self._acquire_large_scale_data(domain, config)

        # 2. Extract multi-metric outcomes from experiments
        data_points = await self._extract_multi_metric_outcomes(domain, experiments)

        # 3. Extract comprehensive features
        enhanced_data_points = await self._extract_comprehensive_features_enhanced(domain, data_points)

        # 4. Apply MVDP criteria and quality control
        final_data_points = await self._apply_mvdp_criteria(domain, enhanced_data_points)

        # Store results
        self.data_points[domain] = final_data_points
        self.stats['domain_stats'][domain]['count'] = len(final_data_points)
        self.stats['domain_stats'][domain]['mvdp_count'] = len([dp for dp in final_data_points if self._meets_mvdp_criteria(dp)])
        self.stats['total_data_points'] += len(final_data_points)

        logger.info(f"Domain {domain}: {len(final_data_points)} data points extracted ({self.stats['domain_stats'][domain]['mvdp_count']} meet MVDP criteria)")

    async def _acquire_large_scale_data(self, domain: str, config: Dict) -> List[Dict]:
        """
        Large-scale data acquisition from ENCODE, GEO, SRA, and other sources
        """
        logger.info(f"Starting large-scale data acquisition for domain: {domain}")

        experiments = []

        # 1. ENCODE data acquisition
        encode_experiments = await self._acquire_encode_data(domain, config)
        experiments.extend(encode_experiments)

        # 2. GEO data acquisition
        geo_experiments = await self._acquire_geo_data(domain, config)
        experiments.extend(geo_experiments)

        # 3. Literature mining for experimental metadata
        literature_experiments = await self._mine_literature_for_experiments(domain, config)
        experiments.extend(literature_experiments)

        logger.info(f"Acquired {len(experiments)} experiments for domain: {domain}")
        return experiments

    async def _acquire_encode_data(self, domain: str, config: Dict) -> List[Dict]:
        """
        Acquire data from ENCODE portal
        """
        experiments = []

        try:
            # Search for gene editing experiments
            search_terms = config['search_terms']

            for term in search_terms:
                # Query ENCODE API
                params = {
                    'type': 'Experiment',
                    'assay_title': 'CRISPR+RNA-seq',  # Adjust based on domain
                    'status': 'released',
                    'limit': 'all',
                    'format': 'json'
                }

                # Add domain-specific filters
                if domain == 'crispr':
                    params['assay_title'] = 'CRISPR+RNA-seq'
                elif domain == 'prime_editing':
                    params['assay_title'] = 'Prime+editing'
                elif domain == 'base_editing':
                    params['assay_title'] = 'Base+editing'

                # Make API request with rate limiting
                await asyncio.sleep(self.data_acquisition_clients['encode']['rate_limit'])

                response = requests.get(
                    self.data_acquisition_clients['encode']['base_url'],
                    params=params,
                    headers=self.data_acquisition_clients['encode']['headers']
                )

                if response.status_code == 200:
                    data = response.json()
                    experiments.extend(data.get('@graph', []))
                    logger.info(f"Retrieved {len(data.get('@graph', []))} ENCODE experiments for term: {term}")

        except Exception as e:
            logger.error(f"Error acquiring ENCODE data: {e}")
            self.stats['errors'].append(f"ENCODE acquisition error: {e}")

        return experiments

    async def _acquire_geo_data(self, domain: str, config: Dict) -> List[Dict]:
        """
        Acquire data from GEO database
        """
        experiments = []

        try:
            # Use NCBI E-utilities to search GEO
            search_terms = config['search_terms']

            for term in search_terms:
                # Search for GEO series
                search_params = {
                    'db': 'gds',
                    'term': f"{term}[Title/Abstract]",
                    'retmax': 100,
                    'retmode': 'json'
                }

                # Make API request with rate limiting
                await asyncio.sleep(0.34)  # NCBI rate limit

                # This would use the NCBI client to search GEO
                # For now, return placeholder
                experiments.append({
                    'source': 'GEO',
                    'search_term': term,
                    'experiment_type': domain,
                    'status': 'placeholder'
                })

        except Exception as e:
            logger.error(f"Error acquiring GEO data: {e}")
            self.stats['errors'].append(f"GEO acquisition error: {e}")

        return experiments

    async def _mine_literature_for_experiments(self, domain: str, config: Dict) -> List[Dict]:
        """
        Mine literature for experimental data and metadata
        """
        experiments = []

        try:
            search_terms = config['search_terms']

            for term in search_terms:
                # Use PubMed client
                if 'pubmed' in self.clients:
                    response = self.clients['pubmed'].search(term, max_results=100)
                    if response.success:
                        papers = self.clients['pubmed'].parse_response(response)
                        for paper in papers:
                            experiments.append({
                                'source': 'PubMed',
                                'paper': paper,
                                'search_term': term,
                                'experiment_type': domain
                            })

                # Use Semantic Scholar client
                if 'semantic_scholar' in self.clients:
                    response = self.clients['semantic_scholar'].search(term, max_results=100)
                    if response.success:
                        papers = self.clients['semantic_scholar'].parse_response(response)
                        for paper in papers:
                            experiments.append({
                                'source': 'Semantic Scholar',
                                'paper': paper,
                                'search_term': term,
                                'experiment_type': domain
                            })

        except Exception as e:
            logger.error(f"Error mining literature: {e}")
            self.stats['errors'].append(f"Literature mining error: {e}")

        return experiments

    async def _extract_multi_metric_outcomes(self, domain: str, experiments: List[Dict]) -> List[EnhancedDataPoint]:
        """
        Extract multi-metric outcomes from experiments
        """
        logger.info(f"Extracting multi-metric outcomes for {len(experiments)} experiments")

        data_points = []

        for experiment in experiments:
            try:
                # Extract outcomes based on domain
                if domain == 'crispr':
                    outcomes = self._extract_crispr_outcomes(experiment)
                elif domain == 'prime_editing':
                    outcomes = self._extract_prime_editing_outcomes(experiment)
                elif domain == 'base_editing':
                    outcomes = self._extract_base_editing_outcomes(experiment)
                else:
                    continue

                # Create data point if outcomes are valid
                if self._validate_outcomes(outcomes, domain):
                    data_point = self._create_enhanced_data_point(experiment, outcomes, domain)
                    data_points.append(data_point)

            except Exception as e:
                logger.error(f"Error extracting outcomes from experiment: {e}")
                self.stats['errors'].append(f"Outcome extraction error: {e}")

        logger.info(f"Extracted {len(data_points)} data points with multi-metric outcomes")
        return data_points

    def _extract_crispr_outcomes(self, experiment: Dict) -> MultiMetricOutcomes:
        """Extract CRISPR-Cas9 outcomes from experiment data"""
        outcomes = MultiMetricOutcomes()

        # Extract from experiment data (placeholder implementation)
        # In real implementation, this would parse NGS data, CRISPResso2 output, etc.

        # Primary metric: Indel frequency
        outcomes.indel_frequency = self._extract_numeric_value(experiment, 'indel_frequency')

        # Secondary metric: On-target vs off-target ratio
        outcomes.on_target_off_target_ratio = self._extract_numeric_value(experiment, 'on_target_off_target_ratio')

        # Tertiary metrics
        outcomes.repair_profile_spectrum = self._extract_dict_value(experiment, 'repair_profile_spectrum')
        outcomes.cell_survival_rate = self._extract_numeric_value(experiment, 'cell_survival_rate')

        return outcomes

    def _extract_prime_editing_outcomes(self, experiment: Dict) -> MultiMetricOutcomes:
        """Extract Prime Editing outcomes from experiment data"""
        outcomes = MultiMetricOutcomes()

        # Primary metric: Precise editing efficiency
        outcomes.precise_editing_efficiency = self._extract_numeric_value(experiment, 'precise_editing_efficiency')

        # Secondary metric: Indel frequency
        outcomes.indel_frequency_pe = self._extract_numeric_value(experiment, 'indel_frequency_pe')

        # Tertiary metrics
        outcomes.product_purity = self._extract_numeric_value(experiment, 'product_purity')
        outcomes.peg_rna_component_efficiencies = self._extract_dict_value(experiment, 'peg_rna_component_efficiencies')

        return outcomes

    def _extract_base_editing_outcomes(self, experiment: Dict) -> MultiMetricOutcomes:
        """Extract Base Editing outcomes from experiment data"""
        outcomes = MultiMetricOutcomes()

        # Primary metric: Target base conversion efficiency
        outcomes.target_base_conversion_efficiency = self._extract_numeric_value(experiment, 'target_base_conversion_efficiency')

        # Secondary metrics
        outcomes.product_purity_be = self._extract_numeric_value(experiment, 'product_purity_be')
        outcomes.bystander_edits = self._extract_numeric_value(experiment, 'bystander_edits')

        # Tertiary metrics
        outcomes.editing_window_profile = self._extract_dict_value(experiment, 'editing_window_profile')
        outcomes.indel_frequency_be = self._extract_numeric_value(experiment, 'indel_frequency_be')
        outcomes.incorrect_transversion_edits = self._extract_numeric_value(experiment, 'incorrect_transversion_edits')

        return outcomes

    def _extract_numeric_value(self, data: Dict, key: str) -> Optional[float]:
        """Extract numeric value from data"""
        try:
            value = data.get(key)
            if value is not None:
                return float(value)
        except (ValueError, TypeError):
            pass
        return None

    def _extract_dict_value(self, data: Dict, key: str) -> Optional[Dict[str, float]]:
        """Extract dictionary value from data"""
        try:
            value = data.get(key)
            if isinstance(value, dict):
                return {k: float(v) for k, v in value.items() if isinstance(v, (int, float))}
        except (ValueError, TypeError):
            pass
        return None

    def _validate_outcomes(self, outcomes: MultiMetricOutcomes, domain: str) -> bool:
        """Validate that outcomes meet minimum criteria"""
        if domain == 'crispr':
            return outcomes.indel_frequency is not None
        elif domain == 'prime_editing':
            return outcomes.precise_editing_efficiency is not None
        elif domain == 'base_editing':
            return outcomes.target_base_conversion_efficiency is not None
        return False

    def _create_enhanced_data_point(self, experiment: Dict, outcomes: MultiMetricOutcomes, domain: str) -> EnhancedDataPoint:
        """Create enhanced data point with outcomes and placeholder features"""

        # Generate unique event ID
        event_id = f"{domain}_{int(time.time() * 1000)}_{hash(str(experiment))}"

        # Create comprehensive features (placeholder for now)
        features = ComprehensiveFeatures()

        # Extract basic information from experiment
        editor_name = experiment.get('editor_name', f'{domain}_editor')
        organism_id = experiment.get('organism_id', 9606)  # Default to human
        cell_type_id = experiment.get('cell_type_id', 'CL:0000057')  # Default cell type
        target_gene_id = experiment.get('target_gene_id', 'ENSG00000139618')  # Default gene

        return EnhancedDataPoint(
            event_id=event_id,
            editor_name=editor_name,
            organism_id=organism_id,
            cell_type_id=cell_type_id,
            target_gene_id=target_gene_id,
            outcomes=outcomes,
            features=features,
            experimental_conditions=experiment.get('experimental_conditions', {}),
            measurement_method=experiment.get('measurement_method', 'NGS'),
            sequencing_depth=experiment.get('sequencing_depth'),
            quality_metrics=experiment.get('quality_metrics', {}),
            source_experiment=experiment.get('source', 'unknown'),
            data_processing_pipeline='enhanced_comprehensive_v1.0',
            extraction_date=datetime.now().isoformat(),
            processing_date=datetime.now().isoformat()
        )

    async def _extract_comprehensive_features_enhanced(self, domain: str, data_points: List[EnhancedDataPoint]) -> List[EnhancedDataPoint]:
        """
        Extract comprehensive features for data points
        """
        logger.info(f"Extracting comprehensive features for {len(data_points)} data points")

        enhanced_points = []

        for data_point in data_points:
            try:
                enhanced_point = await self._extract_features_for_enhanced_data_point(data_point)
                enhanced_points.append(enhanced_point)

            except Exception as e:
                logger.error(f"Error extracting features for {data_point.event_id}: {e}")
                enhanced_points.append(data_point)  # Keep original if feature extraction fails

        logger.info(f"Enhanced {len(enhanced_points)} data points with comprehensive features")
        return enhanced_points

    async def _extract_features_for_enhanced_data_point(self, data_point: EnhancedDataPoint) -> EnhancedDataPoint:
        """
        Extract comprehensive features for a single enhanced data point
        """
        # Extract sequence features
        if hasattr(data_point, 'guide_sequence') and data_point.guide_sequence:
            # Calculate GC content
            data_point.features.gc_content_guide = self._calculate_gc_content(data_point.guide_sequence)

            # Calculate seed region GC content
            seed_region = data_point.guide_sequence[:10] if len(data_point.guide_sequence) >= 10 else data_point.guide_sequence
            data_point.features.gc_content_seed = self._calculate_gc_content(seed_region)

            # Calculate melting temperature
            data_point.features.tm_guide_dna_duplex = self._calculate_melting_temperature(data_point.guide_sequence)

        # Extract RNA structure features if RNAfold is available
        if self.feature_extractors['rna_structure']['available'] and hasattr(data_point, 'guide_sequence'):
            rna_features = self._calculate_rna_structure_features(data_point.guide_sequence)
            data_point.features.grna_mfe = rna_features.get('rna_mfe_energy_kcal_mol')
            data_point.features.grna_ensemble_diversity = rna_features.get('rna_ensemble_diversity')

        # Extract epigenetic features (placeholders for now)
        # In real implementation, these would be extracted from ATAC-seq, ChIP-seq, etc.
        data_point.features.accessibility_score = self._extract_chromatin_accessibility(data_point)
        data_point.features.h3k27ac_score = self._extract_histone_modifications(data_point, 'H3K27ac')
        data_point.features.cpg_methylation_200bp = self._extract_dna_methylation(data_point)

        return data_point

    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a sequence"""
        if not sequence:
            return 0.0

        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0

    def _calculate_melting_temperature(self, sequence: str) -> Optional[float]:
        """Calculate melting temperature using Wallace rule"""
        if not sequence or len(sequence) < 4:
            return None

        # Wallace rule: Tm = 2°C(A+T) + 4°C(G+C)
        a_count = sequence.upper().count('A')
        t_count = sequence.upper().count('T')
        g_count = sequence.upper().count('G')
        c_count = sequence.upper().count('C')

        return 2 * (a_count + t_count) + 4 * (g_count + c_count)

    def _calculate_cpg_content(self, sequence: str) -> float:
        """Calculate CpG content of a sequence"""
        if not sequence:
            return 0.0

        # Count CpG dinucleotides
        cpg_count = sequence.upper().count('CG')
        total_dinucleotides = len(sequence) - 1 if len(sequence) > 1 else 0

        return cpg_count / total_dinucleotides if total_dinucleotides > 0 else 0.0

    def _calculate_rna_structure_features(self, sequence: str) -> Dict[str, Any]:
        """Calculate RNA structure features using ViennaRNA"""
        if not self.feature_extractors['rna_structure']['available']:
            return {}

        try:
            import subprocess

            # Run RNAfold
            result = subprocess.run(
                ['RNAfold', '--noPS'],
                input=sequence,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Parse RNAfold output
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    structure = lines[1].split()[0]
                    energy = float(lines[1].split()[-1].strip('()'))

                    return {
                        'rna_mfe_structure': structure,
                        'rna_mfe_energy_kcal_mol': energy,
                        'rna_ensemble_diversity': 0.0  # Placeholder
                    }

        except Exception as e:
            logger.error(f"Error calculating RNA structure features: {e}")

        return {}

    def _extract_chromatin_accessibility(self, data_point: EnhancedDataPoint) -> Optional[float]:
        """Extract chromatin accessibility score (placeholder)"""
        # In real implementation, this would query ATAC-seq data
        return None

    def _extract_histone_modifications(self, data_point: EnhancedDataPoint, mark: str) -> Optional[float]:
        """Extract histone modification score (placeholder)"""
        # In real implementation, this would query ChIP-seq data
        return None

    def _extract_dna_methylation(self, data_point: EnhancedDataPoint) -> Optional[float]:
        """Extract DNA methylation score (placeholder)"""
        # In real implementation, this would query WGBS data
        return None

    def _extract_chromatin_3d(self, data_point: EnhancedDataPoint) -> Optional[Dict[str, Any]]:
        """Extract 3D chromatin features (placeholder)"""
        # In real implementation, this would query Hi-C data
        return None

    async def _apply_mvdp_criteria(self, domain: str, data_points: List[EnhancedDataPoint]) -> List[EnhancedDataPoint]:
        """
        Apply Minimum Viable Data Point criteria
        """
        logger.info(f"Applying MVDP criteria to {len(data_points)} data points")

        mvdp_data_points = []

        for data_point in data_points:
            if self._meets_mvdp_criteria(data_point):
                mvdp_data_points.append(data_point)

        logger.info(f"MVDP criteria met by {len(mvdp_data_points)} out of {len(data_points)} data points")
        return mvdp_data_points

    def _meets_mvdp_criteria(self, data_point: EnhancedDataPoint) -> bool:
        """Check if data point meets Minimum Viable Data Point criteria"""

        # Check required fields
        for field in self.mvdp_criteria['required_fields']:
            if not self._has_required_field(data_point, field):
                return False

        # Check sequencing depth
        if (data_point.sequencing_depth and
            data_point.sequencing_depth < self.mvdp_criteria['min_sequencing_depth']):
            return False

        # Check quality score
        quality_score = self._calculate_quality_score(data_point)
        if quality_score < self.mvdp_criteria['min_quality_score']:
            return False

        return True

    def _has_required_field(self, data_point: EnhancedDataPoint, field_path: str) -> bool:
        """Check if data point has a required field"""
        try:
            # Handle nested field paths like 'outcomes.indel_frequency'
            parts = field_path.split('.')
            value = data_point

            for part in parts:
                value = getattr(value, part)
                if value is None:
                    return False

            return True
        except AttributeError:
            return False

    def _calculate_quality_score(self, data_point: EnhancedDataPoint) -> float:
        """Calculate quality score for data point"""
        score = 0.0
        total_checks = 0

        # Check for outcomes
        if data_point.outcomes:
            if data_point.outcomes.indel_frequency is not None:
                score += 0.3
            if data_point.outcomes.precise_editing_efficiency is not None:
                score += 0.3
            if data_point.outcomes.target_base_conversion_efficiency is not None:
                score += 0.3
            total_checks += 3

        # Check for features
        if data_point.features:
            if data_point.features.guide_sequence:
                score += 0.2
            if data_point.features.target_sequence_40bp:
                score += 0.2
            if data_point.features.gc_content_guide is not None:
                score += 0.1
            total_checks += 3

        # Check experimental metadata
        if data_point.sequencing_depth and data_point.sequencing_depth >= 1000:
            score += 0.1
        if data_point.measurement_method:
            score += 0.1
        total_checks += 2

        return score / max(total_checks, 1)

    def _generate_enhanced_report(self) -> Dict[str, Any]:
        """Generate enhanced comprehensive report"""
        return {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'total_data_points': self.stats['total_data_points'],
            'domain_breakdown': {
                domain: {
                    'count': stats['count'],
                    'mvdp_count': stats['mvdp_count'],
                    'experiments_used': len(stats['experiments'])
                }
                for domain, stats in self.stats['domain_stats'].items()
            },
            'processing_time': self.stats['processing_time'],
            'errors': self.stats['errors'],
            'target_goal': '500,000 per domain (1.5M total)',
            'current_progress': f"{self.stats['total_data_points']} / 1,500,000 ({(self.stats['total_data_points']/1500000)*100:.2f}%)",
            'mvdp_progress': f"{sum(stats['mvdp_count'] for stats in self.stats['domain_stats'].values())} / 1,500,000 ({(sum(stats['mvdp_count'] for stats in self.stats['domain_stats'].values())/1500000)*100:.2f}%)"
        }

    def _save_enhanced_results(self):
        """Save enhanced comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create data directory
        data_dir = Path("data/enhanced_comprehensive_mining")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save data points by domain
        for domain, data_points in self.data_points.items():
            if data_points:
                # Convert to DataFrame
                df_data = []
                for dp in data_points:
                    dp_dict = {
                        'event_id': dp.event_id,
                        'editor_name': dp.editor_name,
                        'organism_id': dp.organism_id,
                        'cell_type_id': dp.cell_type_id,
                        'target_gene_id': dp.target_gene_id,
                        'domain': domain,
                        'extraction_date': dp.extraction_date,
                        'processing_date': dp.processing_date,
                        'validation_status': dp.validation_status
                    }

                    # Add outcomes
                    if dp.outcomes:
                        for field, value in asdict(dp.outcomes).items():
                            if isinstance(value, dict):
                                dp_dict[f'outcome_{field}'] = json.dumps(value)
                            else:
                                dp_dict[f'outcome_{field}'] = value

                    # Add features
                    if dp.features:
                        for field, value in asdict(dp.features).items():
                            dp_dict[f'feature_{field}'] = value

                    df_data.append(dp_dict)

                df = pd.DataFrame(df_data)

                # Save as CSV
                csv_file = data_dir / f"{domain}_enhanced_data_points_{timestamp}.csv"
                df.to_csv(csv_file, index=False)

                # Save as JSON
                json_file = data_dir / f"{domain}_enhanced_data_points_{timestamp}.json"
                with open(json_file, 'w') as f:
                    json.dump(df_data, f, indent=2, default=str)

                logger.info(f"Saved {len(data_points)} {domain} enhanced data points to {csv_file}")

        # Save enhanced report
        report = self._generate_enhanced_report()
        report_file = data_dir / f"enhanced_comprehensive_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Enhanced comprehensive report saved to {report_file}")

    def get_domain_stats(self) -> Dict[str, Any]:
        """Get statistics for each domain"""
        return {
            domain: {
                'data_points': len(data_points),
                'mvdp_data_points': self.stats['domain_stats'][domain]['mvdp_count'],
                'experiments_used': len(self.stats['domain_stats'][domain]['experiments']),
                'target': 500000,
                'progress_percent': (len(data_points) / 500000) * 100,
                'mvdp_progress_percent': (self.stats['domain_stats'][domain]['mvdp_count'] / 500000) * 100
            }
            for domain, data_points in self.data_points.items()
        }
