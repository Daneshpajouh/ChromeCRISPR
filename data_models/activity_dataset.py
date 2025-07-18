#!/usr/bin/env python3
"""
Activity Prediction Dataset Data Model
Extends efficiency dataset with activity-specific features and targets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json

from .efficiency_dataset import (
    EfficiencyRecord, SourceMetadata, ExperimentalConditions,
    HistoneModifications, StructuralFeatures, EpigeneticFeatures,
    ProteinFeatures, ThermodynamicFeatures
)

@dataclass
class ActivityFeatures:
    """Activity-specific features"""
    gene_expression_level: Optional[float] = None  # FPKM/TPM
    protein_expression_level: Optional[float] = None  # abundance
    transcription_rate: Optional[float] = None  # transcripts/hour
    translation_rate: Optional[float] = None  # proteins/hour
    promoter_strength: Optional[float] = None  # 0-1
    enhancer_activity: Optional[float] = None  # 0-1
    transcription_factor_activity: Optional[float] = None  # 0-1
    chromatin_state: Optional[str] = None  # active, repressed, etc.
    cell_cycle_phase: Optional[str] = None  # G1, S, G2, M
    metabolic_state: Optional[str] = None  # quiescent, active, etc.

@dataclass
class ActivityRecord(EfficiencyRecord):
    """Activity prediction record - extends efficiency record"""

    # Activity-specific features
    activity_features: Optional[ActivityFeatures] = None

    # Activity targets (primary target)
    activity_score: Optional[float] = None  # 0-100% [PRIMARY TARGET]
    activation_level: Optional[float] = None  # fold change
    repression_level: Optional[float] = None  # fold change
    gene_knockdown_efficiency: Optional[float] = None  # 0-100%
    gene_knockout_efficiency: Optional[float] = None  # 0-100%

    # Temporal activity data
    time_course_data: Optional[Dict[str, float]] = None  # {time_point: activity}
    activity_persistence: Optional[float] = None  # days

    # Cell-type specific activity
    cell_type_specificity: Optional[Dict[str, float]] = None  # {cell_type: activity}
    tissue_specificity: Optional[Dict[str, float]] = None  # {tissue: activity}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = super().to_dict()

        # Add activity-specific features
        if self.activity_features:
            data['activity_features'] = {
                'gene_expression_level': self.activity_features.gene_expression_level,
                'protein_expression_level': self.activity_features.protein_expression_level,
                'transcription_rate': self.activity_features.transcription_rate,
                'translation_rate': self.activity_features.translation_rate,
                'promoter_strength': self.activity_features.promoter_strength,
                'enhancer_activity': self.activity_features.enhancer_activity,
                'transcription_factor_activity': self.activity_features.transcription_factor_activity,
                'chromatin_state': self.activity_features.chromatin_state,
                'cell_cycle_phase': self.activity_features.cell_cycle_phase,
                'metabolic_state': self.activity_features.metabolic_state
            }

        # Add activity targets
        data.update({
            'activation_level': self.activation_level,
            'repression_level': self.repression_level,
            'gene_knockdown_efficiency': self.gene_knockdown_efficiency,
            'gene_knockout_efficiency': self.gene_knockout_efficiency,
            'time_course_data': self.time_course_data,
            'activity_persistence': self.activity_persistence,
            'cell_type_specificity': self.cell_type_specificity,
            'tissue_specificity': self.tissue_specificity
        })

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActivityRecord':
        """Create from dictionary"""
        # Create base efficiency record
        base_record = super().from_dict(data)

        # Add activity-specific features
        activity_features = None
        if 'activity_features' in data and data['activity_features']:
            act_data = data['activity_features']
            activity_features = ActivityFeatures(
                gene_expression_level=act_data.get('gene_expression_level'),
                protein_expression_level=act_data.get('protein_expression_level'),
                transcription_rate=act_data.get('transcription_rate'),
                translation_rate=act_data.get('translation_rate'),
                promoter_strength=act_data.get('promoter_strength'),
                enhancer_activity=act_data.get('enhancer_activity'),
                transcription_factor_activity=act_data.get('transcription_factor_activity'),
                chromatin_state=act_data.get('chromatin_state'),
                cell_cycle_phase=act_data.get('cell_cycle_phase'),
                metabolic_state=act_data.get('metabolic_state')
            )

        # Create activity record
        return cls(
            persistent_id=base_record.persistent_id,
            editing_technique=base_record.editing_technique,
            guide_rna_sequence=base_record.guide_rna_sequence,
            target_sequence=base_record.target_sequence,
            pam_sequence=base_record.pam_sequence,
            peg_rna_sequence=base_record.peg_rna_sequence,
            base_editor_type=base_record.base_editor_type,
            target_gene_symbol=base_record.target_gene_symbol,
            target_gene_ensembl_id=base_record.target_gene_ensembl_id,
            target_gene_function=base_record.target_gene_function,
            target_gene_pathway=base_record.target_gene_pathway,
            disease_association=base_record.disease_association,
            cell_line=base_record.cell_line,
            organism=base_record.organism,
            tissue_type=base_record.tissue_type,
            delivery_method=base_record.delivery_method,
            disease_context=base_record.disease_context,
            experimental_conditions=base_record.experimental_conditions,
            thermodynamic_features=base_record.thermodynamic_features,
            structural_features=base_record.structural_features,
            epigenetic_features=base_record.epigenetic_features,
            protein_features=base_record.protein_features,
            efficiency_score=base_record.efficiency_score,
            activity_score=data.get('activity_score'),
            specificity_score=base_record.specificity_score,
            off_target_sites=base_record.off_target_sites,
            off_target_score=base_record.off_target_score,
            off_target_assay=base_record.off_target_assay,
            off_target_results=base_record.off_target_results,
            safety_score=base_record.safety_score,
            adverse_events=base_record.adverse_events,
            cytotoxicity=base_record.cytotoxicity,
            source_metadata=base_record.source_metadata,
            provenance=base_record.provenance,
            activity_features=activity_features,
            activation_level=data.get('activation_level'),
            repression_level=data.get('repression_level'),
            gene_knockdown_efficiency=data.get('gene_knockdown_efficiency'),
            gene_knockout_efficiency=data.get('gene_knockout_efficiency'),
            time_course_data=data.get('time_course_data'),
            activity_persistence=data.get('activity_persistence'),
            cell_type_specificity=data.get('cell_type_specificity'),
            tissue_specificity=data.get('tissue_specificity')
        )
