#!/usr/bin/env python3
"""
Sequence Feature Extraction Module
Extracts and calculates comprehensive sequence-related features
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class SequenceFeatures:
    """Comprehensive sequence features"""
    sequence: str
    gc_content: float
    melting_temperature: float
    molecular_weight: float
    charge: float
    secondary_structure: str
    stem_loops: List[str]
    pseudoknots: List[str]
    hairpins: List[str]
    bulges: List[str]
    internal_loops: List[str]
    structure_confidence: float

class SequenceFeatureExtractor:
    """Extracts comprehensive sequence features"""

    def __init__(self):
        # Thermodynamic parameters for DNA/RNA
        self.dna_thermo = {
            'AA': -1.00, 'AT': -0.88, 'AG': -1.44, 'AC': -1.28,
            'TA': -0.58, 'TT': -1.00, 'TG': -1.45, 'TC': -1.30,
            'GA': -1.29, 'GT': -1.44, 'GG': -1.84, 'GC': -2.17,
            'CA': -1.11, 'CT': -1.28, 'CG': -2.24, 'CC': -1.84
        }

        self.rna_thermo = {
            'AA': -0.93, 'AU': -0.88, 'AG': -1.10, 'AC': -1.69,
            'UA': -1.10, 'UU': -0.93, 'UG': -1.33, 'UC': -1.69,
            'GA': -1.33, 'GU': -1.10, 'GG': -1.84, 'GC': -2.24,
            'CA': -1.69, 'CU': -1.69, 'CG': -2.11, 'CC': -1.84
        }

        # Molecular weights (g/mol)
        self.base_weights = {
            'A': 313.21, 'T': 304.2, 'U': 290.17,
            'G': 329.21, 'C': 289.18
        }

        # Charges at pH 7
        self.base_charges = {
            'A': -1.0, 'T': -1.0, 'U': -1.0,
            'G': -1.0, 'C': -1.0
        }

    def extract_features(self, sequence: str, sequence_type: str = 'RNA') -> SequenceFeatures:
        """Extract all sequence features"""
        sequence = sequence.upper()

        # Basic sequence features
        gc_content = self._calculate_gc_content(sequence)
        melting_temp = self._calculate_melting_temperature(sequence, sequence_type)
        molecular_weight = self._calculate_molecular_weight(sequence)
        charge = self._calculate_charge(sequence)

        # Secondary structure prediction
        secondary_structure = self._predict_secondary_structure(sequence)
        stem_loops = self._extract_stem_loops(secondary_structure)
        pseudoknots = self._extract_pseudoknots(secondary_structure)
        hairpins = self._extract_hairpins(secondary_structure)
        bulges = self._extract_bulges(secondary_structure)
        internal_loops = self._extract_internal_loops(secondary_structure)
        structure_confidence = self._calculate_structure_confidence(sequence)

        return SequenceFeatures(
            sequence=sequence,
            gc_content=gc_content,
            melting_temperature=melting_temp,
            molecular_weight=molecular_weight,
            charge=charge,
            secondary_structure=secondary_structure,
            stem_loops=stem_loops,
            pseudoknots=pseudoknots,
            hairpins=hairpins,
            bulges=bulges,
            internal_loops=internal_loops,
            structure_confidence=structure_confidence
        )

    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content"""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if sequence else 0.0

    def _calculate_melting_temperature(self, sequence: str, sequence_type: str) -> float:
        """Calculate melting temperature using nearest neighbor model"""
        if len(sequence) < 2:
            return 0.0

        # Get thermodynamic parameters
        thermo_params = self.rna_thermo if sequence_type.upper() == 'RNA' else self.dna_thermo

        # Calculate enthalpy and entropy
        enthalpy = 0.0
        entropy = 0.0

        for i in range(len(sequence) - 1):
            dinucleotide = sequence[i:i+2]
            if dinucleotide in thermo_params:
                enthalpy += thermo_params[dinucleotide]
                entropy += -0.0024  # Approximate entropy contribution

        # Melting temperature calculation
        if entropy != 0:
            tm = enthalpy / entropy + 273.15  # Convert to Celsius
            return max(0, tm)
        return 0.0

    def _calculate_molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight"""
        weight = 0.0
        for base in sequence:
            if base in self.base_weights:
                weight += self.base_weights[base]
        return weight

    def _calculate_charge(self, sequence: str) -> float:
        """Calculate net charge at pH 7"""
        charge = 0.0
        for base in sequence:
            if base in self.base_charges:
                charge += self.base_charges[base]
        return charge

    def _predict_secondary_structure(self, sequence: str) -> str:
        """Predict secondary structure using simplified algorithm"""
        # This is a simplified implementation
        # In practice, you'd use tools like RNAfold, ViennaRNA, or similar

        structure = '.' * len(sequence)

        # Simple stem prediction (find complementary regions)
        for i in range(len(sequence) - 4):
            for j in range(i + 4, len(sequence)):
                if self._is_complementary(sequence[i:j+1]):
                    # Mark as paired
                    for k in range(i, j+1):
                        if k < len(structure):
                            if k < (i + j) // 2:
                                structure = structure[:k] + '(' + structure[k+1:]
                            else:
                                structure = structure[:k] + ')' + structure[k+1:]

        return structure

    def _is_complementary(self, seq: str) -> bool:
        """Check if sequence is self-complementary"""
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        rev_comp = ''.join(complement.get(base, base) for base in reversed(seq))
        return seq == rev_comp

    def _extract_stem_loops(self, structure: str) -> List[str]:
        """Extract stem-loop motifs"""
        stem_loops = []
        # Find regions with balanced parentheses
        stack = []
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    start = stack.pop()
                    if i - start > 4:  # Minimum stem length
                        stem_loops.append(f"{start}-{i}")
        return stem_loops

    def _extract_pseudoknots(self, structure: str) -> List[str]:
        """Extract pseudoknot motifs"""
        # Simplified pseudoknot detection
        pseudoknots = []
        # This would require more sophisticated analysis
        return pseudoknots

    def _extract_hairpins(self, structure: str) -> List[str]:
        """Extract hairpin motifs"""
        hairpins = []
        # Find hairpin patterns: (....)
        pattern = r'\([^()]*\)'
        matches = re.finditer(pattern, structure)
        for match in matches:
            hairpins.append(match.group())
        return hairpins

    def _extract_bulges(self, structure: str) -> List[str]:
        """Extract bulge motifs"""
        bulges = []
        # Simplified bulge detection
        return bulges

    def _extract_internal_loops(self, structure: str) -> List[str]:
        """Extract internal loop motifs"""
        internal_loops = []
        # Simplified internal loop detection
        return internal_loops

    def _calculate_structure_confidence(self, sequence: str) -> float:
        """Calculate confidence in structure prediction"""
        # Simplified confidence calculation
        # In practice, this would be based on multiple structure predictions
        gc_content = self._calculate_gc_content(sequence)
        length_factor = min(1.0, len(sequence) / 100.0)
        return (gc_content * 0.5 + length_factor * 0.5)

    def extract_guide_rna_features(self, guide_sequence: str) -> Dict[str, Any]:
        """Extract features specific to guide RNAs"""
        features = self.extract_features(guide_sequence, 'RNA')

        # Guide RNA specific features
        pam_sequence = self._extract_pam_sequence(guide_sequence)
        target_complement = self._get_target_complement(guide_sequence)

        return {
            'guide_sequence': guide_sequence,
            'pam_sequence': pam_sequence,
            'target_complement': target_complement,
            'gc_content': features.gc_content,
            'melting_temperature': features.melting_temperature,
            'secondary_structure': features.secondary_structure,
            'structure_confidence': features.structure_confidence
        }

    def _extract_pam_sequence(self, guide_sequence: str) -> str:
        """Extract PAM sequence (simplified)"""
        # For Cas9, PAM is typically NGG
        # This is a simplified implementation
        if len(guide_sequence) >= 2:
            return guide_sequence[-2:]  # Last 2 bases as PAM
        return ""

    def _get_target_complement(self, guide_sequence: str) -> str:
        """Get target DNA complement"""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'U': 'A'}
        return ''.join(complement.get(base, base) for base in guide_sequence)

    def extract_peg_rna_features(self, peg_sequence: str) -> Dict[str, Any]:
        """Extract features specific to pegRNAs (Prime Editing)"""
        features = self.extract_features(peg_sequence, 'RNA')

        # pegRNA specific features
        primer_binding_site = self._extract_primer_binding_site(peg_sequence)
        reverse_transcriptase_template = self._extract_rt_template(peg_sequence)

        return {
            'peg_sequence': peg_sequence,
            'primer_binding_site': primer_binding_site,
            'rt_template': reverse_transcriptase_template,
            'gc_content': features.gc_content,
            'melting_temperature': features.melting_temperature,
            'secondary_structure': features.secondary_structure
        }

    def _extract_primer_binding_site(self, peg_sequence: str) -> str:
        """Extract primer binding site from pegRNA"""
        # Simplified: assume first 20 bases are PBS
        return peg_sequence[:20] if len(peg_sequence) >= 20 else peg_sequence

    def _extract_rt_template(self, peg_sequence: str) -> str:
        """Extract reverse transcriptase template from pegRNA"""
        # Simplified: assume remaining sequence after PBS is RT template
        pbs_length = 20
        return peg_sequence[pbs_length:] if len(peg_sequence) > pbs_length else ""

    def download_nltk_resources(self):
        """Download required NLTK resources"""
        try:
            import nltk
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
            nltk.download('stopwords')
            print("NLTK resources downloaded successfully")
        except Exception as e:
            print(f"Warning: Could not download NLTK resources: {e}")
