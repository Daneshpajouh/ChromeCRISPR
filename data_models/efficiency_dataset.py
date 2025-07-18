#!/usr/bin/env python3
"""
Efficiency Prediction Dataset Data Model
Comprehensive data structure for predicting gene editing efficiency
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json

@dataclass
class SourceMetadata:
    """Metadata about the source of the data"""
    title: Optional[str] = None
    authors: Optional[List[Dict[str, str]]] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    keywords: Optional[List[str]] = None
    methods: Optional[str] = None

@dataclass
class ExperimentalConditions:
    """Experimental conditions and parameters"""
    temperature: Optional[float] = None  # °C
    time: Optional[float] = None  # hours
    cell_density: Optional[str] = None  # cells/ml
    transfection_method: Optional[str] = None
    media: Optional[str] = None
    supplements: Optional[List[str]] = None
    ph: Optional[float] = None
    oxygen_level: Optional[float] = None  # %

@dataclass
class HistoneModifications:
    """Histone modification data"""
    h3k27ac: Optional[float] = None
    h3k4me3: Optional[float] = None
    h3k9me3: Optional[float] = None
    h3k27me3: Optional[float] = None
    h3k36me3: Optional[float] = None
    h4k20me3: Optional[float] = None
    h3k79me2: Optional[float] = None
    h3k9ac: Optional[float] = None
    h3k14ac: Optional[float] = None

@dataclass
class StructuralFeatures:
    """RNA/DNA structural features"""
    secondary_structure: Optional[str] = None  # dot-bracket notation
    stem_loops: Optional[List[str]] = None
    pseudoknots: Optional[List[str]] = None
    hairpins: Optional[List[str]] = None
    bulges: Optional[List[str]] = None
    internal_loops: Optional[List[str]] = None
    tertiary_structure: Optional[str] = None
    structure_confidence: Optional[float] = None  # 0-1

@dataclass
class EpigeneticFeatures:
    """Epigenetic and chromatin features"""
    methylation_level: Optional[float] = None  # 0-1
    methylation_sites: Optional[List[str]] = None
    chromatin_accessibility: Optional[float] = None  # 0-1
    atac_seq_score: Optional[float] = None
    faire_seq_score: Optional[float] = None
    dnase_seq_score: Optional[float] = None
    histone_modifications: Optional[HistoneModifications] = None
    enhancer_overlap: Optional[bool] = None
    promoter_overlap: Optional[bool] = None
    insulator_overlap: Optional[bool] = None
    silencer_overlap: Optional[bool] = None
    cpg_islands: Optional[List[str]] = None
    transcription_factor_binding: Optional[List[str]] = None

@dataclass
class ProteinFeatures:
    """Protein-related features"""
    cas_variant: Optional[str] = None  # Cas9, Cas12, etc.
    fusion_proteins: Optional[List[str]] = None
    binding_affinity: Optional[float] = None  # nM
    protein_expression: Optional[float] = None
    protein_stability: Optional[float] = None
    post_translational_modifications: Optional[List[str]] = None
    protein_protein_interactions: Optional[List[str]] = None
    catalytic_activity: Optional[float] = None

@dataclass
class ThermodynamicFeatures:
    """Thermodynamic and physical-chemical features"""
    melting_temperature: Optional[float] = None  # °C
    gc_content: Optional[float] = None  # 0-1
    gibbs_free_energy: Optional[float] = None  # kcal/mol
    enthalpy: Optional[float] = None  # kcal/mol
    entropy: Optional[float] = None  # cal/mol·K
    molecular_weight: Optional[float] = None  # Da
    charge: Optional[float] = None
    solubility: Optional[float] = None  # mg/ml
    hydrophobicity: Optional[float] = None
    isoelectric_point: Optional[float] = None
    extinction_coefficient: Optional[float] = None

@dataclass
class EfficiencyRecord:
    """Complete efficiency prediction record"""

    # Basic identifiers
    persistent_id: str
    editing_technique: str  # CRISPR, Prime, Base

    # Sequence data
    guide_rna_sequence: Optional[str] = None
    target_sequence: Optional[str] = None
    pam_sequence: Optional[str] = None
    peg_rna_sequence: Optional[str] = None  # For Prime Editing
    base_editor_type: Optional[str] = None  # For Base Editing

    # Target information
    target_gene_symbol: Optional[str] = None
    target_gene_ensembl_id: Optional[str] = None
    target_gene_function: Optional[str] = None
    target_gene_pathway: Optional[str] = None
    disease_association: Optional[List[str]] = None

    # Experimental context
    cell_line: Optional[str] = None
    organism: Optional[str] = None
    tissue_type: Optional[str] = None
    delivery_method: Optional[str] = None
    disease_context: Optional[str] = None
    experimental_conditions: Optional[ExperimentalConditions] = None

    # Feature groups
    thermodynamic_features: Optional[ThermodynamicFeatures] = None
    structural_features: Optional[StructuralFeatures] = None
    epigenetic_features: Optional[EpigeneticFeatures] = None
    protein_features: Optional[ProteinFeatures] = None

    # Performance metrics (TARGET)
    efficiency_score: Optional[float] = None  # 0-100%
    activity_score: Optional[float] = None  # 0-100%
    specificity_score: Optional[float] = None  # 0-100%

    # Off-target information
    off_target_sites: Optional[List[str]] = None
    off_target_score: Optional[float] = None  # 0-100%
    off_target_assay: Optional[str] = None
    off_target_results: Optional[str] = None

    # Safety information
    safety_score: Optional[float] = None  # 0-100%
    adverse_events: Optional[List[str]] = None
    cytotoxicity: Optional[float] = None  # 0-100%

    # Source and provenance
    source_metadata: Optional[SourceMetadata] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = {
            'persistent_id': self.persistent_id,
            'editing_technique': self.editing_technique,
            'guide_rna_sequence': self.guide_rna_sequence,
            'target_sequence': self.target_sequence,
            'pam_sequence': self.pam_sequence,
            'peg_rna_sequence': self.peg_rna_sequence,
            'base_editor_type': self.base_editor_type,
            'target_gene_symbol': self.target_gene_symbol,
            'target_gene_ensembl_id': self.target_gene_ensembl_id,
            'target_gene_function': self.target_gene_function,
            'target_gene_pathway': self.target_gene_pathway,
            'disease_association': self.disease_association,
            'cell_line': self.cell_line,
            'organism': self.organism,
            'tissue_type': self.tissue_type,
            'delivery_method': self.delivery_method,
            'disease_context': self.disease_context,
            'efficiency_score': self.efficiency_score,
            'activity_score': self.activity_score,
            'specificity_score': self.specificity_score,
            'off_target_sites': self.off_target_sites,
            'off_target_score': self.off_target_score,
            'off_target_assay': self.off_target_assay,
            'off_target_results': self.off_target_results,
            'safety_score': self.safety_score,
            'adverse_events': self.adverse_events,
            'cytotoxicity': self.cytotoxicity,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'provenance': self.provenance
        }

        # Add nested objects
        if self.experimental_conditions:
            data['experimental_conditions'] = {
                'temperature': self.experimental_conditions.temperature,
                'time': self.experimental_conditions.time,
                'cell_density': self.experimental_conditions.cell_density,
                'transfection_method': self.experimental_conditions.transfection_method,
                'media': self.experimental_conditions.media,
                'supplements': self.experimental_conditions.supplements,
                'ph': self.experimental_conditions.ph,
                'oxygen_level': self.experimental_conditions.oxygen_level
            }

        if self.thermodynamic_features:
            data['thermodynamic_features'] = {
                'melting_temperature': self.thermodynamic_features.melting_temperature,
                'gc_content': self.thermodynamic_features.gc_content,
                'gibbs_free_energy': self.thermodynamic_features.gibbs_free_energy,
                'enthalpy': self.thermodynamic_features.enthalpy,
                'entropy': self.thermodynamic_features.entropy,
                'molecular_weight': self.thermodynamic_features.molecular_weight,
                'charge': self.thermodynamic_features.charge,
                'solubility': self.thermodynamic_features.solubility,
                'hydrophobicity': self.thermodynamic_features.hydrophobicity,
                'isoelectric_point': self.thermodynamic_features.isoelectric_point,
                'extinction_coefficient': self.thermodynamic_features.extinction_coefficient
            }

        if self.structural_features:
            data['structural_features'] = {
                'secondary_structure': self.structural_features.secondary_structure,
                'stem_loops': self.structural_features.stem_loops,
                'pseudoknots': self.structural_features.pseudoknots,
                'hairpins': self.structural_features.hairpins,
                'bulges': self.structural_features.bulges,
                'internal_loops': self.structural_features.internal_loops,
                'tertiary_structure': self.structural_features.tertiary_structure,
                'structure_confidence': self.structural_features.structure_confidence
            }

        if self.epigenetic_features:
            data['epigenetic_features'] = {
                'methylation_level': self.epigenetic_features.methylation_level,
                'methylation_sites': self.epigenetic_features.methylation_sites,
                'chromatin_accessibility': self.epigenetic_features.chromatin_accessibility,
                'atac_seq_score': self.epigenetic_features.atac_seq_score,
                'faire_seq_score': self.epigenetic_features.faire_seq_score,
                'dnase_seq_score': self.epigenetic_features.dnase_seq_score,
                'enhancer_overlap': self.epigenetic_features.enhancer_overlap,
                'promoter_overlap': self.epigenetic_features.promoter_overlap,
                'insulator_overlap': self.epigenetic_features.insulator_overlap,
                'silencer_overlap': self.epigenetic_features.silencer_overlap,
                'cpg_islands': self.epigenetic_features.cpg_islands,
                'transcription_factor_binding': self.epigenetic_features.transcription_factor_binding
            }

            if self.epigenetic_features.histone_modifications:
                data['epigenetic_features']['histone_modifications'] = {
                    'h3k27ac': self.epigenetic_features.histone_modifications.h3k27ac,
                    'h3k4me3': self.epigenetic_features.histone_modifications.h3k4me3,
                    'h3k9me3': self.epigenetic_features.histone_modifications.h3k9me3,
                    'h3k27me3': self.epigenetic_features.histone_modifications.h3k27me3,
                    'h3k36me3': self.epigenetic_features.histone_modifications.h3k36me3,
                    'h4k20me3': self.epigenetic_features.histone_modifications.h4k20me3,
                    'h3k79me2': self.epigenetic_features.histone_modifications.h3k79me2,
                    'h3k9ac': self.epigenetic_features.histone_modifications.h3k9ac,
                    'h3k14ac': self.epigenetic_features.histone_modifications.h3k14ac
                }

        if self.protein_features:
            data['protein_features'] = {
                'cas_variant': self.protein_features.cas_variant,
                'fusion_proteins': self.protein_features.fusion_proteins,
                'binding_affinity': self.protein_features.binding_affinity,
                'protein_expression': self.protein_features.protein_expression,
                'protein_stability': self.protein_features.protein_stability,
                'post_translational_modifications': self.protein_features.post_translational_modifications,
                'protein_protein_interactions': self.protein_features.protein_protein_interactions,
                'catalytic_activity': self.protein_features.catalytic_activity
            }

        if self.source_metadata:
            data['source_metadata'] = {
                'title': self.source_metadata.title,
                'authors': self.source_metadata.authors,
                'journal': self.source_metadata.journal,
                'year': self.source_metadata.year,
                'abstract': self.source_metadata.abstract,
                'doi': self.source_metadata.doi,
                'pmid': self.source_metadata.pmid,
                'keywords': self.source_metadata.keywords,
                'methods': self.source_metadata.methods
            }

        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EfficiencyRecord':
        """Create from dictionary"""
        # Handle nested objects
        experimental_conditions = None
        if 'experimental_conditions' in data and data['experimental_conditions']:
            exp_data = data['experimental_conditions']
            experimental_conditions = ExperimentalConditions(
                temperature=exp_data.get('temperature'),
                time=exp_data.get('time'),
                cell_density=exp_data.get('cell_density'),
                transfection_method=exp_data.get('transfection_method'),
                media=exp_data.get('media'),
                supplements=exp_data.get('supplements'),
                ph=exp_data.get('ph'),
                oxygen_level=exp_data.get('oxygen_level')
            )

        thermodynamic_features = None
        if 'thermodynamic_features' in data and data['thermodynamic_features']:
            thermo_data = data['thermodynamic_features']
            thermodynamic_features = ThermodynamicFeatures(
                melting_temperature=thermo_data.get('melting_temperature'),
                gc_content=thermo_data.get('gc_content'),
                gibbs_free_energy=thermo_data.get('gibbs_free_energy'),
                enthalpy=thermo_data.get('enthalpy'),
                entropy=thermo_data.get('entropy'),
                molecular_weight=thermo_data.get('molecular_weight'),
                charge=thermo_data.get('charge'),
                solubility=thermo_data.get('solubility'),
                hydrophobicity=thermo_data.get('hydrophobicity'),
                isoelectric_point=thermo_data.get('isoelectric_point'),
                extinction_coefficient=thermo_data.get('extinction_coefficient')
            )

        # Create the record
        return cls(
            persistent_id=data['persistent_id'],
            editing_technique=data['editing_technique'],
            guide_rna_sequence=data.get('guide_rna_sequence'),
            target_sequence=data.get('target_sequence'),
            pam_sequence=data.get('pam_sequence'),
            peg_rna_sequence=data.get('peg_rna_sequence'),
            base_editor_type=data.get('base_editor_type'),
            target_gene_symbol=data.get('target_gene_symbol'),
            target_gene_ensembl_id=data.get('target_gene_ensembl_id'),
            target_gene_function=data.get('target_gene_function'),
            target_gene_pathway=data.get('target_gene_pathway'),
            disease_association=data.get('disease_association'),
            cell_line=data.get('cell_line'),
            organism=data.get('organism'),
            tissue_type=data.get('tissue_type'),
            delivery_method=data.get('delivery_method'),
            disease_context=data.get('disease_context'),
            experimental_conditions=experimental_conditions,
            thermodynamic_features=thermodynamic_features,
            efficiency_score=data.get('efficiency_score'),
            activity_score=data.get('activity_score'),
            specificity_score=data.get('specificity_score'),
            off_target_sites=data.get('off_target_sites'),
            off_target_score=data.get('off_target_score'),
            off_target_assay=data.get('off_target_assay'),
            off_target_results=data.get('off_target_results'),
            safety_score=data.get('safety_score'),
            adverse_events=data.get('adverse_events'),
            cytotoxicity=data.get('cytotoxicity'),
            provenance=data.get('provenance', {})
        )
