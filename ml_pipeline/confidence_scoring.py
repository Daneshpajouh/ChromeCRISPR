"""
Confidence Scoring and Evidence Tracking Framework
Based on GeneX Phase 1 Research Report 3/3

This module implements the confidence-scoring and evidence-tracking framework
recommended in the research report to handle ambiguity and conflicting findings
in scientific literature.
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of sources as defined in the research report."""
    PEER_REVIEWED_JOURNAL = "peer_reviewed_journal"
    PREPRINT = "preprint"
    PATENT = "patent"
    CLINICAL_TRIAL = "clinical_trial"
    CONFERENCE_ABSTRACT = "conference_abstract"
    REVIEW_ARTICLE = "review_article"


class ConfidenceLevel(Enum):
    """Confidence levels for extracted facts."""
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


@dataclass
class EvidenceMetadata:
    """Metadata for evidence tracking as recommended in the research report."""
    source_paper_id: str
    source_type: SourceType
    publication_date: str
    evidence_sentence: str
    extraction_confidence: float
    model_used: str
    extraction_timestamp: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    figure_reference: Optional[str] = None
    table_reference: Optional[str] = None


@dataclass
class ConfidenceScore:
    """Comprehensive confidence scoring for extracted facts."""
    overall_confidence: float
    model_confidence: float
    source_reliability: float
    evidence_strength: float
    temporal_relevance: float
    replication_status: Optional[str] = None
    conflicting_evidence: List[str] = None
    supporting_evidence: List[str] = None


@dataclass
class ExtractedFact:
    """Enhanced fact representation with confidence and evidence tracking."""
    subject: str
    predicate: str
    object: str
    confidence_score: ConfidenceScore
    evidence_metadata: EvidenceMetadata
    fact_id: str
    extraction_timestamp: str
    last_updated: str
    version: int = 1

    def __post_init__(self):
        if self.confidence_score.conflicting_evidence is None:
            self.confidence_score.conflicting_evidence = []
        if self.confidence_score.supporting_evidence is None:
            self.confidence_score.supporting_evidence = []


class ConfidenceScoringFramework:
    """
    Implements the confidence-scoring framework recommended in the research report.

    This framework addresses the challenge of handling conflicting findings and
    varying levels of evidence in scientific literature by providing a nuanced
    approach to knowledge representation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.source_reliability_weights = {
            SourceType.PEER_REVIEWED_JOURNAL: 1.0,
            SourceType.REVIEW_ARTICLE: 0.9,
            SourceType.CLINICAL_TRIAL: 0.8,
            SourceType.PATENT: 0.7,
            SourceType.PREPRINT: 0.5,
            SourceType.CONFERENCE_ABSTRACT: 0.4
        }

        # Model confidence weights based on benchmarking results
        self.model_confidence_weights = {
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract": 0.9,
            "dmis-lab/biobert-base-cased-v1.2": 0.85,
            "en_core_sci_sm": 0.8,
            "NCBI/bluebert_pubmed_mimic_uncased_L-12_H-768_A-768_A-12": 0.85,
            "bert-base-uncased": 0.7
        }

    def calculate_source_reliability(self, source_type: SourceType,
                                   publication_date: str,
                                   citation_count: Optional[int] = None) -> float:
        """
        Calculate source reliability based on type and metadata.

        Args:
            source_type: Type of source (peer-reviewed, preprint, etc.)
            publication_date: Publication date in YYYY-MM-DD format
            citation_count: Number of citations (if available)

        Returns:
            Reliability score between 0 and 1
        """
        base_reliability = self.source_reliability_weights.get(source_type, 0.5)

        # Temporal relevance: newer papers get slight boost
        try:
            pub_date = datetime.strptime(publication_date, "%Y-%m-%d")
            current_date = datetime.now()
            years_old = (current_date - pub_date).days / 365.25

            # Papers from last 2 years get slight boost, older papers slight penalty
            if years_old <= 2:
                temporal_factor = 1.05
            elif years_old <= 5:
                temporal_factor = 1.0
            else:
                temporal_factor = 0.95

        except ValueError:
            temporal_factor = 1.0

        # Citation impact (if available)
        citation_factor = 1.0
        if citation_count is not None:
            if citation_count > 100:
                citation_factor = 1.1
            elif citation_count > 50:
                citation_factor = 1.05
            elif citation_count < 5:
                citation_factor = 0.95

        reliability = base_reliability * temporal_factor * citation_factor
        return min(reliability, 1.0)

    def calculate_evidence_strength(self, evidence_sentence: str,
                                  has_figure: bool = False,
                                  has_table: bool = False,
                                  has_supplementary: bool = False) -> float:
        """
        Calculate evidence strength based on supporting materials.

        Args:
            evidence_sentence: The sentence containing the evidence
            has_figure: Whether the claim is supported by a figure
            has_table: Whether the claim is supported by a table
            has_supplementary: Whether the claim is supported by supplementary data

        Returns:
            Evidence strength score between 0 and 1
        """
        # Base strength from sentence quality
        sentence_length = len(evidence_sentence.split())
        if sentence_length > 20:
            base_strength = 0.8
        elif sentence_length > 10:
            base_strength = 0.7
        else:
            base_strength = 0.6

        # Boost for supporting materials
        material_boost = 0.0
        if has_figure:
            material_boost += 0.1
        if has_table:
            material_boost += 0.1
        if has_supplementary:
            material_boost += 0.05

        strength = base_strength + material_boost
        return min(strength, 1.0)

    def calculate_temporal_relevance(self, publication_date: str,
                                   fact_type: str) -> float:
        """
        Calculate temporal relevance based on fact type and publication date.

        Args:
            publication_date: Publication date in YYYY-MM-DD format
            fact_type: Type of fact (efficiency, safety, etc.)

        Returns:
            Temporal relevance score between 0 and 1
        """
        try:
            pub_date = datetime.strptime(publication_date, "%Y-%m-%d")
            current_date = datetime.now()
            years_old = (current_date - pub_date).days / 365.25

            # Different decay rates for different fact types
            if fact_type in ["efficiency", "safety", "off_target"]:
                # Technical facts decay faster
                if years_old <= 1:
                    return 1.0
                elif years_old <= 3:
                    return 0.9
                elif years_old <= 5:
                    return 0.8
                else:
                    return max(0.6, 1.0 - (years_old - 5) * 0.05)
            else:
                # Conceptual facts decay slower
                if years_old <= 5:
                    return 1.0
                elif years_old <= 10:
                    return 0.9
                else:
                    return max(0.7, 1.0 - (years_old - 10) * 0.02)

        except ValueError:
            return 0.8

    def calculate_overall_confidence(self, model_confidence: float,
                                   source_reliability: float,
                                   evidence_strength: float,
                                   temporal_relevance: float) -> float:
        """
        Calculate overall confidence using weighted combination.

        Args:
            model_confidence: NLP model confidence score
            source_reliability: Reliability of the source
            evidence_strength: Strength of the evidence
            temporal_relevance: Temporal relevance of the fact

        Returns:
            Overall confidence score between 0 and 1
        """
        # Weights based on research report recommendations
        weights = {
            'model': 0.3,
            'source': 0.3,
            'evidence': 0.25,
            'temporal': 0.15
        }

        overall_confidence = (
            model_confidence * weights['model'] +
            source_reliability * weights['source'] +
            evidence_strength * weights['evidence'] +
            temporal_relevance * weights['temporal']
        )

        return min(overall_confidence, 1.0)

    def create_extracted_fact(self, subject: str, predicate: str, object: str,
                            model_confidence: float, model_used: str,
                            source_paper_id: str, source_type: SourceType,
                            publication_date: str, evidence_sentence: str,
                            **kwargs) -> ExtractedFact:
        """
        Create an ExtractedFact with comprehensive confidence scoring.

        Args:
            subject: Subject of the fact
            predicate: Predicate/relation
            object: Object of the fact
            model_confidence: Confidence from NLP model
            model_used: Name of the model used
            source_paper_id: ID of the source paper
            source_type: Type of source
            publication_date: Publication date
            evidence_sentence: Sentence containing the evidence
            **kwargs: Additional metadata

        Returns:
            ExtractedFact with confidence scoring
        """
        # Calculate component scores
        source_reliability = self.calculate_source_reliability(
            source_type, publication_date, kwargs.get('citation_count')
        )

        evidence_strength = self.calculate_evidence_strength(
            evidence_sentence,
            kwargs.get('has_figure', False),
            kwargs.get('has_table', False),
            kwargs.get('has_supplementary', False)
        )

        temporal_relevance = self.calculate_temporal_relevance(
            publication_date, kwargs.get('fact_type', 'general')
        )

        # Calculate overall confidence
        overall_confidence = self.calculate_overall_confidence(
            model_confidence, source_reliability, evidence_strength, temporal_relevance
        )

        # Create confidence score object
        confidence_score = ConfidenceScore(
            overall_confidence=overall_confidence,
            model_confidence=model_confidence,
            source_reliability=source_reliability,
            evidence_strength=evidence_strength,
            temporal_relevance=temporal_relevance,
            replication_status=kwargs.get('replication_status'),
            conflicting_evidence=kwargs.get('conflicting_evidence', []),
            supporting_evidence=kwargs.get('supporting_evidence', [])
        )

        # Create evidence metadata
        evidence_metadata = EvidenceMetadata(
            source_paper_id=source_paper_id,
            source_type=source_type,
            publication_date=publication_date,
            evidence_sentence=evidence_sentence,
            extraction_confidence=model_confidence,
            model_used=model_used,
            extraction_timestamp=datetime.now().isoformat(),
            page_number=kwargs.get('page_number'),
            section=kwargs.get('section'),
            figure_reference=kwargs.get('figure_reference'),
            table_reference=kwargs.get('table_reference')
        )

        # Generate unique fact ID
        fact_id = self._generate_fact_id(subject, predicate, object, source_paper_id)

        # Create extracted fact
        extracted_fact = ExtractedFact(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence_score=confidence_score,
            evidence_metadata=evidence_metadata,
            fact_id=fact_id,
            extraction_timestamp=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )

        return extracted_fact

    def _generate_fact_id(self, subject: str, predicate: str, object: str,
                         source_paper_id: str) -> str:
        """Generate unique fact ID based on content and source."""
        content = f"{subject}|{predicate}|{object}|{source_paper_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def detect_conflicting_facts(self, new_fact: ExtractedFact,
                               existing_facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """
        Detect conflicting facts based on the research report recommendations.

        Args:
            new_fact: Newly extracted fact
            existing_facts: List of existing facts

        Returns:
            List of potentially conflicting facts
        """
        conflicting_facts = []

        for existing_fact in existing_facts:
            # Check for same subject-predicate pair
            if (existing_fact.subject == new_fact.subject and
                existing_fact.predicate == new_fact.predicate):

                # Check for conflicting objects
                if self._is_conflicting_object(new_fact.object, existing_fact.object):
                    conflicting_facts.append(existing_fact)

        return conflicting_facts

    def _is_conflicting_object(self, obj1: str, obj2: str) -> bool:
        """
        Determine if two objects are conflicting.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            True if objects are conflicting
        """
        # Try to extract numeric values for comparison
        try:
            # Extract numbers from strings like "95% efficiency" vs "50% efficiency"
            import re

            # Look for percentage patterns
            pct1 = re.findall(r'(\d+(?:\.\d+)?)\s*%', obj1)
            pct2 = re.findall(r'(\d+(?:\.\d+)?)\s*%', obj2)

            if pct1 and pct2:
                val1, val2 = float(pct1[0]), float(pct2[0])
                # Consider conflicting if difference > 20%
                return abs(val1 - val2) > 20

            # Look for fold change patterns
            fold1 = re.findall(r'(\d+(?:\.\d+)?)\s*-?fold', obj1.lower())
            fold2 = re.findall(r'(\d+(?:\.\d+)?)\s*-?fold', obj2.lower())

            if fold1 and fold2:
                val1, val2 = float(fold1[0]), float(fold2[0])
                # Consider conflicting if ratio > 2 or < 0.5
                ratio = max(val1, val2) / min(val1, val2)
                return ratio > 2

            # Check for opposite qualitative terms
            positive_terms = ['high', 'efficient', 'successful', 'good', 'strong']
            negative_terms = ['low', 'inefficient', 'unsuccessful', 'poor', 'weak']

            obj1_lower, obj2_lower = obj1.lower(), obj2.lower()

            obj1_positive = any(term in obj1_lower for term in positive_terms)
            obj1_negative = any(term in obj1_lower for term in negative_terms)
            obj2_positive = any(term in obj2_lower for term in positive_terms)
            obj2_negative = any(term in obj2_lower for term in negative_terms)

            if (obj1_positive and obj2_negative) or (obj1_negative and obj2_positive):
                return True

        except Exception as e:
            logger.warning(f"Error comparing objects '{obj1}' and '{obj2}': {e}")

        return False

    def resolve_conflicts(self, conflicting_facts: List[ExtractedFact]) -> Dict[str, Any]:
        """
        Resolve conflicts using the framework from the research report.

        Args:
            conflicting_facts: List of conflicting facts

        Returns:
            Resolution strategy and recommended fact
        """
        if not conflicting_facts:
            return {"strategy": "no_conflict", "recommended_fact": None}

        # Sort by overall confidence
        sorted_facts = sorted(conflicting_facts,
                            key=lambda x: x.confidence_score.overall_confidence,
                            reverse=True)

        highest_confidence = sorted_facts[0]
        second_highest = sorted_facts[1] if len(sorted_facts) > 1 else None

        # Determine resolution strategy
        if second_highest is None:
            strategy = "single_fact"
        elif (highest_confidence.confidence_score.overall_confidence -
              second_highest.confidence_score.overall_confidence > 0.2):
            strategy = "high_confidence_difference"
        else:
            strategy = "context_dependent"

        return {
            "strategy": strategy,
            "recommended_fact": highest_confidence,
            "all_facts": conflicting_facts,
            "confidence_difference": (
                highest_confidence.confidence_score.overall_confidence -
                second_highest.confidence_score.overall_confidence
                if second_highest else 0
            )
        }

    def to_dict(self, fact: ExtractedFact) -> Dict[str, Any]:
        """Convert ExtractedFact to dictionary for storage."""
        return {
            "fact_id": fact.fact_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "confidence_score": asdict(fact.confidence_score),
            "evidence_metadata": asdict(fact.evidence_metadata),
            "extraction_timestamp": fact.extraction_timestamp,
            "last_updated": fact.last_updated,
            "version": fact.version
        }

    def from_dict(self, data: Dict[str, Any]) -> ExtractedFact:
        """Create ExtractedFact from dictionary."""
        confidence_score = ConfidenceScore(**data["confidence_score"])
        evidence_metadata = EvidenceMetadata(**data["evidence_metadata"])

        return ExtractedFact(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence_score=confidence_score,
            evidence_metadata=evidence_metadata,
            fact_id=data["fact_id"],
            extraction_timestamp=data["extraction_timestamp"],
            last_updated=data["last_updated"],
            version=data.get("version", 1)
        )
