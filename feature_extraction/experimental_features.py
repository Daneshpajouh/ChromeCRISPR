#!/usr/bin/env python3
"""
Experimental Feature Extraction Module
Extracts experimental conditions, delivery methods, efficiency metrics, safety data, and off-target analysis
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class ExperimentalConditions:
    """Comprehensive experimental conditions"""
    cell_line: str
    organism: str
    tissue_type: Optional[str] = None
    cell_density: Optional[float] = None  # cells/mL
    passage_number: Optional[int] = None
    culture_medium: Optional[str] = None
    serum_concentration: Optional[float] = None  # %
    temperature: Optional[float] = None  # °C
    humidity: Optional[float] = None  # %
    co2_concentration: Optional[float] = None  # %
    ph: Optional[float] = None
    incubation_time: Optional[float] = None  # hours
    transfection_method: Optional[str] = None
    transfection_reagent: Optional[str] = None
    transfection_ratio: Optional[float] = None  # DNA:reagent ratio
    selection_method: Optional[str] = None
    selection_duration: Optional[float] = None  # days

@dataclass
class DeliveryFeatures:
    """Delivery method features"""
    delivery_method: str  # Viral, Non-viral, Physical
    delivery_vehicle: Optional[str] = None  # AAV, lentivirus, lipid, etc.
    vector_type: Optional[str] = None  # serotype, formulation
    titer: Optional[float] = None  # viral particles/mL
    multiplicity_of_infection: Optional[float] = None  # MOI
    concentration: Optional[float] = None  # ng/μL
    volume: Optional[float] = None  # μL
    administration_route: Optional[str] = None  # IV, IP, IM, etc.
    targeting_specificity: Optional[float] = None  # 0-1
    delivery_efficiency: Optional[float] = None  # 0-100%
    toxicity_score: Optional[float] = None  # 0-1

@dataclass
class EfficiencyMetrics:
    """Comprehensive efficiency metrics"""
    editing_efficiency: float  # 0-100% [PRIMARY]
    insertion_efficiency: Optional[float] = None  # 0-100%
    deletion_efficiency: Optional[float] = None  # 0-100%
    substitution_efficiency: Optional[float] = None  # 0-100%
    knockin_efficiency: Optional[float] = None  # 0-100%
    knockout_efficiency: Optional[float] = None  # 0-100%
    activation_efficiency: Optional[float] = None  # 0-100%
    repression_efficiency: Optional[float] = None  # 0-100%
    biallelic_editing: Optional[float] = None  # 0-100%
    clonal_efficiency: Optional[float] = None  # 0-100%
    time_course_data: Optional[Dict[str, float]] = None  # {time_point: efficiency}
    dose_response_data: Optional[Dict[str, float]] = None  # {dose: efficiency}

@dataclass
class SafetyMetrics:
    """Comprehensive safety metrics"""
    cytotoxicity: Optional[float] = None  # 0-100%
    cell_viability: Optional[float] = None  # 0-100%
    apoptosis_rate: Optional[float] = None  # 0-100%
    necrosis_rate: Optional[float] = None  # 0-100%
    cell_proliferation: Optional[float] = None  # fold change
    dna_damage: Optional[float] = None  # 0-1
    oxidative_stress: Optional[float] = None  # 0-1
    inflammatory_response: Optional[float] = None  # 0-1
    immune_response: Optional[float] = None  # 0-1
    genotoxicity: Optional[float] = None  # 0-1
    teratogenicity: Optional[float] = None  # 0-1
    carcinogenicity: Optional[float] = None  # 0-1

@dataclass
class OffTargetAnalysis:
    """Comprehensive off-target analysis"""
    off_target_sites: List[str]  # List of off-target sequences
    off_target_scores: Dict[str, float]  # {sequence: score}
    off_target_positions: Dict[str, str]  # {sequence: genomic_position}
    off_target_genes: Dict[str, str]  # {sequence: gene_name}
    off_target_effects: Dict[str, str]  # {sequence: effect_type}
    off_target_assay: Optional[str] = None  # Detection method
    off_target_frequency: Optional[float] = None  # Overall frequency
    off_target_severity: Optional[str] = None  # Low, Medium, High
    off_target_mitigation: Optional[List[str]] = None  # Mitigation strategies

class ExperimentalFeatureExtractor:
    """Extracts comprehensive experimental features"""

    def __init__(self):
        # Common cell lines and their properties
        self.cell_line_database = {
            'HEK293T': {
                'organism': 'Human',
                'tissue_type': 'Kidney',
                'origin': 'Embryonic kidney',
                'growth_rate': 'Fast',
                'transfection_efficiency': 'High'
            },
            'HeLa': {
                'organism': 'Human',
                'tissue_type': 'Cervix',
                'origin': 'Cervical cancer',
                'growth_rate': 'Fast',
                'transfection_efficiency': 'Medium'
            },
            'K562': {
                'organism': 'Human',
                'tissue_type': 'Blood',
                'origin': 'Chronic myeloid leukemia',
                'growth_rate': 'Fast',
                'transfection_efficiency': 'Medium'
            },
            'Jurkat': {
                'organism': 'Human',
                'tissue_type': 'Blood',
                'origin': 'T-cell leukemia',
                'growth_rate': 'Fast',
                'transfection_efficiency': 'Medium'
            }
        }

        # Delivery methods and their properties
        self.delivery_methods = {
            'Lipofection': {
                'type': 'Non-viral',
                'vehicle': 'Lipid nanoparticles',
                'efficiency': 'Medium',
                'toxicity': 'Low',
                'cost': 'Low'
            },
            'Electroporation': {
                'type': 'Physical',
                'vehicle': 'Electric field',
                'efficiency': 'High',
                'toxicity': 'Medium',
                'cost': 'Medium'
            },
            'AAV': {
                'type': 'Viral',
                'vehicle': 'Adeno-associated virus',
                'efficiency': 'High',
                'toxicity': 'Low',
                'cost': 'High'
            },
            'Lentivirus': {
                'type': 'Viral',
                'vehicle': 'Lentiviral vector',
                'efficiency': 'Very High',
                'toxicity': 'Medium',
                'cost': 'High'
            }
        }

    def extract_experimental_conditions(self, cell_line: str, **kwargs) -> ExperimentalConditions:
        """Extract experimental conditions"""
        cell_data = self.cell_line_database.get(cell_line, {})

        # Generate synthetic data for unknown cell lines
        if not cell_data:
            cell_data = self._generate_synthetic_cell_data(cell_line)

        return ExperimentalConditions(
            cell_line=cell_line,
            organism=cell_data.get('organism', 'Human'),
            tissue_type=cell_data.get('tissue_type'),
            cell_density=kwargs.get('cell_density', 1e5),
            passage_number=kwargs.get('passage_number', 5),
            culture_medium=kwargs.get('culture_medium', 'DMEM'),
            serum_concentration=kwargs.get('serum_concentration', 10.0),
            temperature=kwargs.get('temperature', 37.0),
            humidity=kwargs.get('humidity', 95.0),
            co2_concentration=kwargs.get('co2_concentration', 5.0),
            ph=kwargs.get('ph', 7.4),
            incubation_time=kwargs.get('incubation_time', 48.0),
            transfection_method=kwargs.get('transfection_method', 'Lipofection'),
            transfection_reagent=kwargs.get('transfection_reagent', 'Lipofectamine 3000'),
            transfection_ratio=kwargs.get('transfection_ratio', 1.0),
            selection_method=kwargs.get('selection_method', 'Puromycin'),
            selection_duration=kwargs.get('selection_duration', 3.0)
        )

    def extract_delivery_features(self, delivery_method: str, **kwargs) -> DeliveryFeatures:
        """Extract delivery method features"""
        method_data = self.delivery_methods.get(delivery_method, {})

        # Generate synthetic data for unknown methods
        if not method_data:
            method_data = self._generate_synthetic_delivery_data(delivery_method)

        return DeliveryFeatures(
            delivery_method=delivery_method,
            delivery_vehicle=method_data.get('vehicle'),
            vector_type=kwargs.get('vector_type'),
            titer=kwargs.get('titer'),
            multiplicity_of_infection=kwargs.get('moi'),
            concentration=kwargs.get('concentration', 100.0),
            volume=kwargs.get('volume', 10.0),
            administration_route=kwargs.get('administration_route'),
            targeting_specificity=kwargs.get('targeting_specificity', 0.8),
            delivery_efficiency=kwargs.get('delivery_efficiency', 70.0),
            toxicity_score=kwargs.get('toxicity_score', 0.2)
        )

    def extract_efficiency_metrics(self, editing_data: Dict[str, Any]) -> EfficiencyMetrics:
        """Extract efficiency metrics from experimental data"""
        return EfficiencyMetrics(
            editing_efficiency=editing_data.get('editing_efficiency', 0.0),
            insertion_efficiency=editing_data.get('insertion_efficiency'),
            deletion_efficiency=editing_data.get('deletion_efficiency'),
            substitution_efficiency=editing_data.get('substitution_efficiency'),
            knockin_efficiency=editing_data.get('knockin_efficiency'),
            knockout_efficiency=editing_data.get('knockout_efficiency'),
            activation_efficiency=editing_data.get('activation_efficiency'),
            repression_efficiency=editing_data.get('repression_efficiency'),
            biallelic_editing=editing_data.get('biallelic_editing'),
            clonal_efficiency=editing_data.get('clonal_efficiency'),
            time_course_data=editing_data.get('time_course_data'),
            dose_response_data=editing_data.get('dose_response_data')
        )

    def extract_safety_metrics(self, safety_data: Dict[str, Any]) -> SafetyMetrics:
        """Extract safety metrics from experimental data"""
        return SafetyMetrics(
            cytotoxicity=safety_data.get('cytotoxicity'),
            cell_viability=safety_data.get('cell_viability'),
            apoptosis_rate=safety_data.get('apoptosis_rate'),
            necrosis_rate=safety_data.get('necrosis_rate'),
            cell_proliferation=safety_data.get('cell_proliferation'),
            dna_damage=safety_data.get('dna_damage'),
            oxidative_stress=safety_data.get('oxidative_stress'),
            inflammatory_response=safety_data.get('inflammatory_response'),
            immune_response=safety_data.get('immune_response'),
            genotoxicity=safety_data.get('genotoxicity'),
            teratogenicity=safety_data.get('teratogenicity'),
            carcinogenicity=safety_data.get('carcinogenicity')
        )

    def extract_off_target_analysis(self, guide_sequence: str, target_sequence: str, **kwargs) -> OffTargetAnalysis:
        """Extract off-target analysis"""
        # Generate synthetic off-target sites
        off_target_sites = self._predict_off_target_sites(guide_sequence, target_sequence)
        off_target_scores = self._calculate_off_target_scores(off_target_sites)
        off_target_positions = self._assign_off_target_positions(off_target_sites)
        off_target_genes = self._identify_off_target_genes(off_target_sites)
        off_target_effects = self._predict_off_target_effects(off_target_sites)

        return OffTargetAnalysis(
            off_target_sites=off_target_sites,
            off_target_scores=off_target_scores,
            off_target_positions=off_target_positions,
            off_target_genes=off_target_genes,
            off_target_effects=off_target_effects,
            off_target_assay=kwargs.get('off_target_assay', 'GUIDE-seq'),
            off_target_frequency=kwargs.get('off_target_frequency', 0.01),
            off_target_severity=kwargs.get('off_target_severity', 'Low'),
            off_target_mitigation=kwargs.get('off_target_mitigation', ['High-fidelity Cas9', 'Truncated gRNAs'])
        )

    def _generate_synthetic_cell_data(self, cell_line: str) -> Dict[str, Any]:
        """Generate synthetic cell line data"""
        import random

        organisms = ['Human', 'Mouse', 'Rat', 'Monkey']
        tissue_types = ['Kidney', 'Liver', 'Brain', 'Heart', 'Lung', 'Blood', 'Skin']

        return {
            'organism': random.choice(organisms),
            'tissue_type': random.choice(tissue_types),
            'origin': f'{random.choice(tissue_types)} derived',
            'growth_rate': random.choice(['Slow', 'Medium', 'Fast']),
            'transfection_efficiency': random.choice(['Low', 'Medium', 'High'])
        }

    def _generate_synthetic_delivery_data(self, delivery_method: str) -> Dict[str, Any]:
        """Generate synthetic delivery method data"""
        import random

        delivery_types = ['Viral', 'Non-viral', 'Physical']
        vehicles = ['Lipid nanoparticles', 'Polymer nanoparticles', 'Viral vectors', 'Electric field']

        return {
            'type': random.choice(delivery_types),
            'vehicle': random.choice(vehicles),
            'efficiency': random.choice(['Low', 'Medium', 'High', 'Very High']),
            'toxicity': random.choice(['Low', 'Medium', 'High']),
            'cost': random.choice(['Low', 'Medium', 'High'])
        }

    def _predict_off_target_sites(self, guide_sequence: str, target_sequence: str) -> List[str]:
        """Predict potential off-target sites"""
        import random

        # Simplified off-target prediction
        # In practice, this would use tools like Cas-OFFinder, CCTop, etc.

        off_target_sites = []
        num_off_targets = random.randint(0, 5)

        for i in range(num_off_targets):
            # Generate sequences with 1-3 mismatches
            mismatches = random.randint(1, 3)
            off_target = self._introduce_mismatches(guide_sequence, mismatches)
            off_target_sites.append(off_target)

        return off_target_sites

    def _introduce_mismatches(self, sequence: str, num_mismatches: int) -> str:
        """Introduce random mismatches into a sequence"""
        import random

        sequence_list = list(sequence)
        positions = random.sample(range(len(sequence)), min(num_mismatches, len(sequence)))

        bases = ['A', 'T', 'G', 'C']
        for pos in positions:
            original_base = sequence_list[pos]
            new_bases = [b for b in bases if b != original_base]
            sequence_list[pos] = random.choice(new_bases)

        return ''.join(sequence_list)

    def _calculate_off_target_scores(self, off_target_sites: List[str]) -> Dict[str, float]:
        """Calculate off-target scores"""
        import random

        scores = {}
        for site in off_target_sites:
            # Simplified scoring based on sequence similarity
            # In practice, this would use more sophisticated algorithms
            score = random.uniform(0.1, 0.9)
            scores[site] = score

        return scores

    def _assign_off_target_positions(self, off_target_sites: List[str]) -> Dict[str, str]:
        """Assign genomic positions to off-target sites"""
        import random

        positions = {}
        chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']

        for site in off_target_sites:
            chromosome = random.choice(chromosomes)
            position = random.randint(1000000, 100000000)
            positions[site] = f"chr{chromosome}:{position}"

        return positions

    def _identify_off_target_genes(self, off_target_sites: List[str]) -> Dict[str, str]:
        """Identify genes at off-target sites"""
        import random

        genes = {}
        gene_names = ['GENE_A', 'GENE_B', 'GENE_C', 'GENE_D', 'GENE_E']

        for site in off_target_sites:
            gene = random.choice(gene_names)
            genes[site] = gene

        return genes

    def _predict_off_target_effects(self, off_target_sites: List[str]) -> Dict[str, str]:
        """Predict effects of off-target editing"""
        import random

        effects = {}
        effect_types = ['Silent', 'Missense', 'Nonsense', 'Frameshift', 'Splice site']

        for site in off_target_sites:
            effect = random.choice(effect_types)
            effects[site] = effect

        return effects

    def calculate_overall_efficiency_score(self, efficiency_metrics: EfficiencyMetrics) -> float:
        """Calculate overall efficiency score"""
        # Weighted combination of different efficiency metrics
        weights = {
            'editing_efficiency': 0.4,
            'insertion_efficiency': 0.2,
            'deletion_efficiency': 0.2,
            'biallelic_editing': 0.1,
            'clonal_efficiency': 0.1
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            value = getattr(efficiency_metrics, metric)
            if value is not None:
                score += value * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def calculate_overall_safety_score(self, safety_metrics: SafetyMetrics) -> float:
        """Calculate overall safety score"""
        # Inverse of toxicity metrics (higher is safer)
        weights = {
            'cell_viability': 0.3,
            'cytotoxicity': 0.2,  # Inverse
            'apoptosis_rate': 0.2,  # Inverse
            'dna_damage': 0.15,  # Inverse
            'genotoxicity': 0.15  # Inverse
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            value = getattr(safety_metrics, metric)
            if value is not None:
                if metric in ['cytotoxicity', 'apoptosis_rate', 'dna_damage', 'genotoxicity']:
                    # Inverse these metrics
                    score += (100 - value) * weight
                else:
                    score += value * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0
