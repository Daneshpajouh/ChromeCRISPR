#!/usr/bin/env python3
"""
Sequence Design Dataset Data Model
Takes disease/gene context as input, outputs optimal sequences and protocols
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json

@dataclass
class ClinicalContext:
    """Clinical and disease context"""
    disease_name: str
    disease_icd10: Optional[str] = None
    disease_omim: Optional[str] = None
    severity: Optional[str] = None  # mild, moderate, severe
    stage: Optional[str] = None  # early, late, metastatic
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    comorbidities: Optional[List[str]] = None
    treatment_history: Optional[List[str]] = None

@dataclass
class TargetGene:
    """Target gene information"""
    symbol: str
    ensembl_id: Optional[str] = None
    function: Optional[str] = None
    pathway: Optional[str] = None
    expression_pattern: Optional[str] = None
    mutation_types: Optional[List[str]] = None
    therapeutic_relevance: Optional[str] = None

@dataclass
class DesignSolution:
    """Complete design solution"""
    editing_technique: str  # CRISPR, Prime, Base
    optimal_guide_sequence: str  # [PRIMARY TARGET]
    delivery_method: str  # [PRIMARY TARGET]
    expected_efficiency: float  # 0-100% [PRIMARY TARGET]
    expected_safety: float  # 0-100% [PRIMARY TARGET]
    protocol: str  # [PRIMARY TARGET] - step-by-step protocol

    # Optional fields
    target_sequence: Optional[str] = None
    pam_sequence: Optional[str] = None
    peg_rna_sequence: Optional[str] = None  # For Prime Editing
    base_editor_type: Optional[str] = None  # For Base Editing
    delivery_vehicle: Optional[str] = None  # AAV, lipid, etc.
    dosage: Optional[str] = None
    administration_route: Optional[str] = None
    confidence_score: Optional[float] = None  # 0-1
    timeline: Optional[str] = None
    quality_control_steps: Optional[List[str]] = None
    supporting_evidence: Optional[List[str]] = None
    references: Optional[List[str]] = None
    clinical_trials: Optional[List[str]] = None

@dataclass
class DesignRecord:
    """Sequence design record"""

    # Input context
    persistent_id: str
    clinical_context: ClinicalContext
    target_gene: TargetGene
    organism: str
    cell_line: Optional[str] = None
    tissue_type: Optional[str] = None

    # Design solutions for each technique
    crispr_solution: Optional[DesignSolution] = None
    prime_solution: Optional[DesignSolution] = None
    base_solution: Optional[DesignSolution] = None

    # Comparative analysis
    technique_comparison: Optional[Dict[str, Dict[str, float]]] = None  # {technique: {metric: value}}
    recommended_technique: Optional[str] = None

    # Regulatory and safety
    regulatory_status: Optional[str] = None  # preclinical, clinical, approved
    safety_profile: Optional[str] = None
    risk_assessment: Optional[str] = None
    contraindications: Optional[List[str]] = None

    # Cost and accessibility
    estimated_cost: Optional[float] = None
    development_time: Optional[str] = None
    availability: Optional[str] = None

    # Source and provenance
    source_metadata: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = {
            'persistent_id': self.persistent_id,
            'organism': self.organism,
            'cell_line': self.cell_line,
            'tissue_type': self.tissue_type,
            'technique_comparison': self.technique_comparison,
            'recommended_technique': self.recommended_technique,
            'regulatory_status': self.regulatory_status,
            'safety_profile': self.safety_profile,
            'risk_assessment': self.risk_assessment,
            'contraindications': self.contraindications,
            'estimated_cost': self.estimated_cost,
            'development_time': self.development_time,
            'availability': self.availability,
            'source_metadata': self.source_metadata,
            'provenance': self.provenance,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

        # Add clinical context
        if self.clinical_context:
            data['clinical_context'] = {
                'disease_name': self.clinical_context.disease_name,
                'disease_icd10': self.clinical_context.disease_icd10,
                'disease_omim': self.clinical_context.disease_omim,
                'severity': self.clinical_context.severity,
                'stage': self.clinical_context.stage,
                'patient_age': self.clinical_context.patient_age,
                'patient_sex': self.clinical_context.patient_sex,
                'comorbidities': self.clinical_context.comorbidities,
                'treatment_history': self.clinical_context.treatment_history
            }

        # Add target gene
        if self.target_gene:
            data['target_gene'] = {
                'symbol': self.target_gene.symbol,
                'ensembl_id': self.target_gene.ensembl_id,
                'function': self.target_gene.function,
                'pathway': self.target_gene.pathway,
                'expression_pattern': self.target_gene.expression_pattern,
                'mutation_types': self.target_gene.mutation_types,
                'therapeutic_relevance': self.target_gene.therapeutic_relevance
            }

        # Add design solutions
        if self.crispr_solution:
            data['crispr_solution'] = self._solution_to_dict(self.crispr_solution)
        if self.prime_solution:
            data['prime_solution'] = self._solution_to_dict(self.prime_solution)
        if self.base_solution:
            data['base_solution'] = self._solution_to_dict(self.base_solution)

        return data

    def _solution_to_dict(self, solution: DesignSolution) -> Dict[str, Any]:
        """Convert design solution to dictionary"""
        return {
            'editing_technique': solution.editing_technique,
            'optimal_guide_sequence': solution.optimal_guide_sequence,
            'target_sequence': solution.target_sequence,
            'pam_sequence': solution.pam_sequence,
            'peg_rna_sequence': solution.peg_rna_sequence,
            'base_editor_type': solution.base_editor_type,
            'delivery_method': solution.delivery_method,
            'delivery_vehicle': solution.delivery_vehicle,
            'dosage': solution.dosage,
            'administration_route': solution.administration_route,
            'expected_efficiency': solution.expected_efficiency,
            'expected_safety': solution.expected_safety,
            'confidence_score': solution.confidence_score,
            'protocol': solution.protocol,
            'timeline': solution.timeline,
            'quality_control_steps': solution.quality_control_steps,
            'supporting_evidence': solution.supporting_evidence,
            'references': solution.references,
            'clinical_trials': solution.clinical_trials
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DesignRecord':
        """Create from dictionary"""
        # Parse clinical context
        clinical_context = None
        if 'clinical_context' in data and data['clinical_context']:
            ctx_data = data['clinical_context']
            clinical_context = ClinicalContext(
                disease_name=ctx_data['disease_name'],
                disease_icd10=ctx_data.get('disease_icd10'),
                disease_omim=ctx_data.get('disease_omim'),
                severity=ctx_data.get('severity'),
                stage=ctx_data.get('stage'),
                patient_age=ctx_data.get('patient_age'),
                patient_sex=ctx_data.get('patient_sex'),
                comorbidities=ctx_data.get('comorbidities'),
                treatment_history=ctx_data.get('treatment_history')
            )

        # Parse target gene
        target_gene = None
        if 'target_gene' in data and data['target_gene']:
            gene_data = data['target_gene']
            target_gene = TargetGene(
                symbol=gene_data['symbol'],
                ensembl_id=gene_data.get('ensembl_id'),
                function=gene_data.get('function'),
                pathway=gene_data.get('pathway'),
                expression_pattern=gene_data.get('expression_pattern'),
                mutation_types=gene_data.get('mutation_types'),
                therapeutic_relevance=gene_data.get('therapeutic_relevance')
            )

        # Parse design solutions
        crispr_solution = None
        if 'crispr_solution' in data and data['crispr_solution']:
            crispr_solution = cls._solution_from_dict(data['crispr_solution'])

        prime_solution = None
        if 'prime_solution' in data and data['prime_solution']:
            prime_solution = cls._solution_from_dict(data['prime_solution'])

        base_solution = None
        if 'base_solution' in data and data['base_solution']:
            base_solution = cls._solution_from_dict(data['base_solution'])

        return cls(
            persistent_id=data['persistent_id'],
            clinical_context=clinical_context,
            target_gene=target_gene,
            organism=data['organism'],
            cell_line=data.get('cell_line'),
            tissue_type=data.get('tissue_type'),
            crispr_solution=crispr_solution,
            prime_solution=prime_solution,
            base_solution=base_solution,
            technique_comparison=data.get('technique_comparison'),
            recommended_technique=data.get('recommended_technique'),
            regulatory_status=data.get('regulatory_status'),
            safety_profile=data.get('safety_profile'),
            risk_assessment=data.get('risk_assessment'),
            contraindications=data.get('contraindications'),
            estimated_cost=data.get('estimated_cost'),
            development_time=data.get('development_time'),
            availability=data.get('availability'),
            source_metadata=data.get('source_metadata'),
            provenance=data.get('provenance', {})
        )

    @classmethod
    def _solution_from_dict(cls, data: Dict[str, Any]) -> DesignSolution:
        """Create design solution from dictionary"""
        return DesignSolution(
            editing_technique=data['editing_technique'],
            optimal_guide_sequence=data['optimal_guide_sequence'],
            target_sequence=data.get('target_sequence'),
            pam_sequence=data.get('pam_sequence'),
            peg_rna_sequence=data.get('peg_rna_sequence'),
            base_editor_type=data.get('base_editor_type'),
            delivery_method=data['delivery_method'],
            delivery_vehicle=data.get('delivery_vehicle'),
            dosage=data.get('dosage'),
            administration_route=data.get('administration_route'),
            expected_efficiency=data['expected_efficiency'],
            expected_safety=data['expected_safety'],
            confidence_score=data.get('confidence_score'),
            protocol=data['protocol'],
            timeline=data.get('timeline'),
            quality_control_steps=data.get('quality_control_steps'),
            supporting_evidence=data.get('supporting_evidence'),
            references=data.get('references'),
            clinical_trials=data.get('clinical_trials')
        )
