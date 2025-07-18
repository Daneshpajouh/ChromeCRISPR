"""
Enhanced Data Structures for Gene Editing Database
Implements multi-metric outcome framework and comprehensive feature engineering
Based on the technical blueprint for large-scale gene editing database construction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

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

@dataclass
class MVDPCriteria:
    """Minimum Viable Data Point criteria"""
    required_fields: List[str] = field(default_factory=lambda: [
        'event_id', 'editor_name', 'organism_id', 'cell_type_id',
        'outcomes.indel_frequency', 'outcomes.precise_editing_efficiency',
        'outcomes.target_base_conversion_efficiency',
        'features.guide_sequence', 'features.target_sequence_40bp'
    ])
    min_sequencing_depth: int = 1000
    min_quality_score: float = 0.7

@dataclass
class DataAcquisitionConfig:
    """Configuration for large-scale data acquisition"""
    encode: Dict[str, Any] = field(default_factory=lambda: {
        'base_url': "https://www.encodeproject.org/search/",
        'rate_limit': 0.1,  # 10 req/sec
        'headers': {'accept': 'application/json'}
    })

    ucsc: Dict[str, Any] = field(default_factory=lambda: {
        'base_url': "https://api.genome.ucsc.edu/",
        'rate_limit': 1.0  # 1 req/sec
    })

    sra: Dict[str, Any] = field(default_factory=lambda: {
        'prefetch_path': "prefetch",
        'fasterq_dump_path': "fasterq-dump",
        'max_concurrent_downloads': 5
    })

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction tools"""
    rna_structure: Dict[str, Any] = field(default_factory=lambda: {
        'tool': "RNAfold",
        'available': False,
        'parameters': {
            'temperature': 37.0,
            'no_lonely_pairs': True,
            'no_gu': False
        }
    })

    sequence_analysis: Dict[str, bool] = field(default_factory=lambda: {
        'gc_content': True,
        'melting_temperature': True,
        'cpg_content': True
    })

    epigenetic: Dict[str, bool] = field(default_factory=lambda: {
        'chromatin_accessibility': True,
        'histone_modifications': True,
        'dna_methylation': True,
        'chromatin_3d': True
    })
