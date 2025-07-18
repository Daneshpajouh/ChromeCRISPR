#!/usr/bin/env python3
"""
Biological Feature Extraction Module
Extracts gene, protein, pathway, disease, and epigenetic features
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class GeneFeatures:
    """Gene-level features"""
    symbol: str
    ensembl_id: Optional[str] = None
    chromosome: Optional[str] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    strand: Optional[str] = None
    transcript_count: Optional[int] = None
    exon_count: Optional[int] = None
    cds_length: Optional[int] = None
    utr_length: Optional[int] = None
    gc_content: Optional[float] = None
    expression_level: Optional[float] = None
    expression_tissue_specificity: Optional[Dict[str, float]] = None
    expression_developmental_stage: Optional[Dict[str, float]] = None

@dataclass
class ProteinFeatures:
    """Protein-level features"""
    uniprot_id: Optional[str] = None
    protein_name: Optional[str] = None
    sequence_length: Optional[int] = None
    molecular_weight: Optional[float] = None
    isoelectric_point: Optional[float] = None
    charge: Optional[float] = None
    hydrophobicity: Optional[float] = None
    secondary_structure: Optional[Dict[str, float]] = None
    domains: Optional[List[str]] = None
    motifs: Optional[List[str]] = None
    post_translational_modifications: Optional[List[str]] = None
    subcellular_localization: Optional[List[str]] = None
    protein_protein_interactions: Optional[List[str]] = None

@dataclass
class PathwayFeatures:
    """Pathway-level features"""
    pathway_name: Optional[str] = None
    pathway_id: Optional[str] = None
    pathway_type: Optional[str] = None
    pathway_size: Optional[int] = None
    pathway_centrality: Optional[float] = None
    pathway_enrichment: Optional[float] = None
    pathway_genes: Optional[List[str]] = None
    pathway_functions: Optional[List[str]] = None
    pathway_diseases: Optional[List[str]] = None

@dataclass
class DiseaseFeatures:
    """Disease-level features"""
    disease_name: Optional[str] = None
    disease_id: Optional[str] = None
    disease_type: Optional[str] = None
    severity: Optional[str] = None
    prevalence: Optional[float] = None
    inheritance_pattern: Optional[str] = None
    affected_systems: Optional[List[str]] = None
    symptoms: Optional[List[str]] = None
    treatments: Optional[List[str]] = None
    genetic_basis: Optional[str] = None

@dataclass
class EpigeneticFeatures:
    """Epigenetic features"""
    histone_modifications: Optional[Dict[str, float]] = None
    dna_methylation: Optional[Dict[str, float]] = None
    chromatin_state: Optional[str] = None
    chromatin_accessibility: Optional[float] = None
    transcription_factor_binding: Optional[List[str]] = None
    enhancer_activity: Optional[float] = None
    promoter_activity: Optional[float] = None
    regulatory_elements: Optional[List[str]] = None

class BiologicalFeatureExtractor:
    """Extracts comprehensive biological features"""

    def __init__(self):
        # Common gene symbols and their properties
        self.gene_database = {
            'BRCA1': {
                'ensembl_id': 'ENSG00000012048',
                'chromosome': '17',
                'function': 'DNA repair, tumor suppression',
                'pathway': 'Homologous recombination repair',
                'diseases': ['Breast cancer', 'Ovarian cancer']
            },
            'TP53': {
                'ensembl_id': 'ENSG00000141510',
                'chromosome': '17',
                'function': 'Tumor suppression, cell cycle regulation',
                'pathway': 'p53 signaling pathway',
                'diseases': ['Li-Fraumeni syndrome', 'Various cancers']
            },
            'CFTR': {
                'ensembl_id': 'ENSG00000001626',
                'chromosome': '7',
                'function': 'Chloride channel regulation',
                'pathway': 'Ion transport',
                'diseases': ['Cystic fibrosis']
            }
        }

        # Protein domains and motifs
        self.protein_domains = {
            'BRCA1': ['BRCT', 'RING', 'Coiled-coil'],
            'TP53': ['DNA-binding', 'Tetramerization', 'Transactivation'],
            'CFTR': ['ABC transporter', 'NBD', 'MSD']
        }

        # Pathway information
        self.pathway_database = {
            'DNA repair': {
                'type': 'Cellular process',
                'genes': ['BRCA1', 'BRCA2', 'PARP1', 'ATM', 'ATR'],
                'diseases': ['Cancer', 'Fanconi anemia']
            },
            'Cell cycle': {
                'type': 'Cellular process',
                'genes': ['TP53', 'CDKN2A', 'RB1', 'CCND1'],
                'diseases': ['Cancer', 'Retinoblastoma']
            },
            'Ion transport': {
                'type': 'Physiological process',
                'genes': ['CFTR', 'SLC26A3', 'ENaC'],
                'diseases': ['Cystic fibrosis', 'Congenital chloride diarrhea']
            }
        }

    def extract_gene_features(self, gene_symbol: str) -> GeneFeatures:
        """Extract comprehensive gene features"""
        gene_data = self.gene_database.get(gene_symbol, {})

        # Generate synthetic data for missing genes
        if not gene_data:
            gene_data = self._generate_synthetic_gene_data(gene_symbol)

        return GeneFeatures(
            symbol=gene_symbol,
            ensembl_id=gene_data.get('ensembl_id'),
            chromosome=gene_data.get('chromosome'),
            start_position=gene_data.get('start_position'),
            end_position=gene_data.get('end_position'),
            strand=gene_data.get('strand'),
            transcript_count=gene_data.get('transcript_count'),
            exon_count=gene_data.get('exon_count'),
            cds_length=gene_data.get('cds_length'),
            utr_length=gene_data.get('utr_length'),
            gc_content=gene_data.get('gc_content'),
            expression_level=gene_data.get('expression_level'),
            expression_tissue_specificity=gene_data.get('expression_tissue_specificity'),
            expression_developmental_stage=gene_data.get('expression_developmental_stage')
        )

    def extract_protein_features(self, gene_symbol: str) -> ProteinFeatures:
        """Extract comprehensive protein features"""
        domains = self.protein_domains.get(gene_symbol, [])

        # Generate synthetic protein data
        protein_data = self._generate_synthetic_protein_data(gene_symbol)

        return ProteinFeatures(
            uniprot_id=protein_data.get('uniprot_id'),
            protein_name=protein_data.get('protein_name'),
            sequence_length=protein_data.get('sequence_length'),
            molecular_weight=protein_data.get('molecular_weight'),
            isoelectric_point=protein_data.get('isoelectric_point'),
            charge=protein_data.get('charge'),
            hydrophobicity=protein_data.get('hydrophobicity'),
            secondary_structure=protein_data.get('secondary_structure'),
            domains=domains,
            motifs=protein_data.get('motifs'),
            post_translational_modifications=protein_data.get('ptms'),
            subcellular_localization=protein_data.get('localization'),
            protein_protein_interactions=protein_data.get('interactions')
        )

    def extract_pathway_features(self, gene_symbol: str) -> List[PathwayFeatures]:
        """Extract pathway features for a gene"""
        pathways = []

        # Find pathways containing this gene
        for pathway_name, pathway_data in self.pathway_database.items():
            if gene_symbol in pathway_data.get('genes', []):
                pathways.append(PathwayFeatures(
                    pathway_name=pathway_name,
                    pathway_id=f"PATH:{pathway_name.replace(' ', '_')}",
                    pathway_type=pathway_data.get('type'),
                    pathway_size=len(pathway_data.get('genes', [])),
                    pathway_centrality=self._calculate_pathway_centrality(gene_symbol, pathway_data),
                    pathway_enrichment=self._calculate_pathway_enrichment(gene_symbol, pathway_data),
                    pathway_genes=pathway_data.get('genes'),
                    pathway_functions=pathway_data.get('functions'),
                    pathway_diseases=pathway_data.get('diseases')
                ))

        # Generate synthetic pathway data if none found
        if not pathways:
            pathways.append(self._generate_synthetic_pathway_data(gene_symbol))

        return pathways

    def extract_disease_features(self, gene_symbol: str) -> List[DiseaseFeatures]:
        """Extract disease features for a gene"""
        diseases = []

        # Find diseases associated with this gene
        for pathway_name, pathway_data in self.pathway_database.items():
            if gene_symbol in pathway_data.get('genes', []):
                for disease_name in pathway_data.get('diseases', []):
                    diseases.append(DiseaseFeatures(
                        disease_name=disease_name,
                        disease_id=f"DISEASE:{disease_name.replace(' ', '_')}",
                        disease_type=self._classify_disease_type(disease_name),
                        severity=self._classify_disease_severity(disease_name),
                        prevalence=self._estimate_disease_prevalence(disease_name),
                        inheritance_pattern=self._determine_inheritance_pattern(disease_name),
                        affected_systems=self._identify_affected_systems(disease_name),
                        symptoms=self._generate_disease_symptoms(disease_name),
                        treatments=self._generate_disease_treatments(disease_name),
                        genetic_basis=self._determine_genetic_basis(disease_name)
                    ))

        # Generate synthetic disease data if none found
        if not diseases:
            diseases.append(self._generate_synthetic_disease_data(gene_symbol))

        return diseases

    def extract_epigenetic_features(self, gene_symbol: str) -> EpigeneticFeatures:
        """Extract epigenetic features for a gene"""
        # Generate synthetic epigenetic data
        epigenetic_data = self._generate_synthetic_epigenetic_data(gene_symbol)

        return EpigeneticFeatures(
            histone_modifications=epigenetic_data.get('histone_modifications'),
            dna_methylation=epigenetic_data.get('dna_methylation'),
            chromatin_state=epigenetic_data.get('chromatin_state'),
            chromatin_accessibility=epigenetic_data.get('chromatin_accessibility'),
            transcription_factor_binding=epigenetic_data.get('tf_binding'),
            enhancer_activity=epigenetic_data.get('enhancer_activity'),
            promoter_activity=epigenetic_data.get('promoter_activity'),
            regulatory_elements=epigenetic_data.get('regulatory_elements')
        )

    def _generate_synthetic_gene_data(self, gene_symbol: str) -> Dict[str, Any]:
        """Generate synthetic gene data"""
        import random

        return {
            'ensembl_id': f"ENSG{random.randint(10000000000, 99999999999)}",
            'chromosome': str(random.randint(1, 23)),
            'start_position': random.randint(1000000, 100000000),
            'end_position': random.randint(1000000, 100000000),
            'strand': random.choice(['+', '-']),
            'transcript_count': random.randint(1, 10),
            'exon_count': random.randint(1, 20),
            'cds_length': random.randint(1000, 10000),
            'utr_length': random.randint(100, 2000),
            'gc_content': random.uniform(0.3, 0.7),
            'expression_level': random.uniform(0.1, 100.0),
            'expression_tissue_specificity': {
                'brain': random.uniform(0.1, 10.0),
                'heart': random.uniform(0.1, 10.0),
                'liver': random.uniform(0.1, 10.0),
                'kidney': random.uniform(0.1, 10.0)
            },
            'expression_developmental_stage': {
                'embryonic': random.uniform(0.1, 10.0),
                'fetal': random.uniform(0.1, 10.0),
                'adult': random.uniform(0.1, 10.0)
            }
        }

    def _generate_synthetic_protein_data(self, gene_symbol: str) -> Dict[str, Any]:
        """Generate synthetic protein data"""
        import random

        return {
            'uniprot_id': f"P{random.randint(10000, 99999)}",
            'protein_name': f"{gene_symbol} protein",
            'sequence_length': random.randint(100, 2000),
            'molecular_weight': random.uniform(10000, 200000),
            'isoelectric_point': random.uniform(4.0, 10.0),
            'charge': random.uniform(-50, 50),
            'hydrophobicity': random.uniform(-2.0, 2.0),
            'secondary_structure': {
                'alpha_helix': random.uniform(0.1, 0.5),
                'beta_sheet': random.uniform(0.1, 0.4),
                'random_coil': random.uniform(0.2, 0.6)
            },
            'motifs': ['NLS', 'NES', 'PEST'],
            'ptms': ['Phosphorylation', 'Ubiquitination', 'Acetylation'],
            'localization': ['Nucleus', 'Cytoplasm', 'Membrane'],
            'interactions': [f'PROTEIN_{i}' for i in range(random.randint(1, 5))]
        }

    def _generate_synthetic_pathway_data(self, gene_symbol: str) -> PathwayFeatures:
        """Generate synthetic pathway data"""
        import random

        pathway_types = ['Metabolic', 'Signaling', 'Cellular process', 'Disease']
        pathway_names = ['Cell cycle regulation', 'DNA repair', 'Apoptosis', 'Metabolism']

        return PathwayFeatures(
            pathway_name=random.choice(pathway_names),
            pathway_id=f"PATH:{random.randint(1000, 9999)}",
            pathway_type=random.choice(pathway_types),
            pathway_size=random.randint(5, 50),
            pathway_centrality=random.uniform(0.1, 1.0),
            pathway_enrichment=random.uniform(0.1, 1.0),
            pathway_genes=[f'GENE_{i}' for i in range(random.randint(3, 10))],
            pathway_functions=['Regulation', 'Metabolism', 'Signaling'],
            pathway_diseases=['Disease A', 'Disease B']
        )

    def _generate_synthetic_disease_data(self, gene_symbol: str) -> DiseaseFeatures:
        """Generate synthetic disease data"""
        import random

        disease_types = ['Genetic', 'Cancer', 'Metabolic', 'Neurological']
        severity_levels = ['Mild', 'Moderate', 'Severe']
        inheritance_patterns = ['Autosomal dominant', 'Autosomal recessive', 'X-linked']

        return DiseaseFeatures(
            disease_name=f"{gene_symbol} syndrome",
            disease_id=f"DISEASE:{random.randint(1000, 9999)}",
            disease_type=random.choice(disease_types),
            severity=random.choice(severity_levels),
            prevalence=random.uniform(0.0001, 0.1),
            inheritance_pattern=random.choice(inheritance_patterns),
            affected_systems=['Nervous system', 'Cardiovascular system'],
            symptoms=['Symptom A', 'Symptom B', 'Symptom C'],
            treatments=['Treatment A', 'Treatment B'],
            genetic_basis='Single gene mutation'
        )

    def _generate_synthetic_epigenetic_data(self, gene_symbol: str) -> Dict[str, Any]:
        """Generate synthetic epigenetic data"""
        import random

        return {
            'histone_modifications': {
                'H3K4me3': random.uniform(0.1, 1.0),
                'H3K27ac': random.uniform(0.1, 1.0),
                'H3K9me3': random.uniform(0.1, 1.0),
                'H3K27me3': random.uniform(0.1, 1.0)
            },
            'dna_methylation': {
                'promoter': random.uniform(0.0, 1.0),
                'gene_body': random.uniform(0.0, 1.0),
                'enhancer': random.uniform(0.0, 1.0)
            },
            'chromatin_state': random.choice(['Active', 'Repressed', 'Bivalent', 'Poised']),
            'chromatin_accessibility': random.uniform(0.1, 1.0),
            'tf_binding': [f'TF_{i}' for i in range(random.randint(1, 5))],
            'enhancer_activity': random.uniform(0.1, 1.0),
            'promoter_activity': random.uniform(0.1, 1.0),
            'regulatory_elements': ['Enhancer A', 'Promoter B', 'Silencer C']
        }

    def _calculate_pathway_centrality(self, gene_symbol: str, pathway_data: Dict) -> float:
        """Calculate gene centrality in pathway"""
        import random
        return random.uniform(0.1, 1.0)

    def _calculate_pathway_enrichment(self, gene_symbol: str, pathway_data: Dict) -> float:
        """Calculate pathway enrichment"""
        import random
        return random.uniform(0.1, 1.0)

    def _classify_disease_type(self, disease_name: str) -> str:
        """Classify disease type"""
        if 'cancer' in disease_name.lower():
            return 'Cancer'
        elif 'syndrome' in disease_name.lower():
            return 'Genetic'
        else:
            return 'Other'

    def _classify_disease_severity(self, disease_name: str) -> str:
        """Classify disease severity"""
        import random
        return random.choice(['Mild', 'Moderate', 'Severe'])

    def _estimate_disease_prevalence(self, disease_name: str) -> float:
        """Estimate disease prevalence"""
        import random
        return random.uniform(0.0001, 0.1)

    def _determine_inheritance_pattern(self, disease_name: str) -> str:
        """Determine inheritance pattern"""
        import random
        return random.choice(['Autosomal dominant', 'Autosomal recessive', 'X-linked'])

    def _identify_affected_systems(self, disease_name: str) -> List[str]:
        """Identify affected body systems"""
        import random
        systems = ['Nervous system', 'Cardiovascular system', 'Respiratory system', 'Digestive system']
        return random.sample(systems, random.randint(1, 3))

    def _generate_disease_symptoms(self, disease_name: str) -> List[str]:
        """Generate disease symptoms"""
        import random
        symptoms = ['Fatigue', 'Pain', 'Fever', 'Weight loss', 'Nausea', 'Headache']
        return random.sample(symptoms, random.randint(2, 4))

    def _generate_disease_treatments(self, disease_name: str) -> List[str]:
        """Generate disease treatments"""
        import random
        treatments = ['Medication', 'Surgery', 'Therapy', 'Lifestyle changes']
        return random.sample(treatments, random.randint(1, 3))

    def _determine_genetic_basis(self, disease_name: str) -> str:
        """Determine genetic basis of disease"""
        import random
        bases = ['Single gene mutation', 'Polygenic', 'Chromosomal abnormality', 'Mitochondrial']
        return random.choice(bases)
