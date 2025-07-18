"""
MVDP (Minimum Viable Data Point) Validator
Comprehensive data quality assessment framework for gene editing domains
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for data point assessment"""
    completeness_score: float
    accuracy_score: float
    relevance_score: float
    overall_score: float
    missing_fields: List[str]
    quality_issues: List[str]
    organism_type: str
    therapeutic_relevance: bool


@dataclass
class DomainThresholds:
    """Domain-specific quality thresholds"""
    min_efficiency: float
    min_precision: float
    min_conversion: float
    required_fields: int
    min_completeness: float
    human_priority_bonus: float


class MVDPValidator:
    """
    Validates a mined gene editing experiment record against the Minimum Viable Data Product (MVDP) standard.
    Returns (is_valid, compliance_dict) where is_valid is True if compliance rate meets minimum threshold.
    """
    REQUIRED_FIELDS = [
        "persistent_id", "source_metadata", "organism_metadata", "editing_technique",
        "target_gene", "guide_rna_sequence", "delivery_method", "efficiency_metric",
        "efficiency_assay", "off_target_assessment", "off_target_assay", "off_target_results"
    ]

    def validate(self, record):
        """
        Validates a mined gene editing experiment record against the Minimum Viable Data Product (MVDP) standard.
        Returns (is_valid, compliance_dict) where is_valid is True if compliance rate meets minimum threshold.
        """
        compliance = {}
        present_count = 0

        for field in self.REQUIRED_FIELDS:
            present = field in record and record[field] not in (None, "", [], {})
            compliance[field] = present
            if present:
                present_count += 1

        # Calculate compliance rate
        compliance_rate = present_count / len(self.REQUIRED_FIELDS)

        # Accept records with at least 40% compliance (5 out of 12 fields)
        # This allows for real-world data where some fields may be missing
        is_valid = compliance_rate >= 0.4

        return is_valid, compliance

    def __init__(self):
        # Domain-specific quality thresholds based on research
        self.quality_thresholds = {
            'crispr': DomainThresholds(
                min_efficiency=25.0,
                min_precision=20.0,
                min_conversion=0.0,  # Not applicable for CRISPR
                required_fields=8,
                min_completeness=0.8,
                human_priority_bonus=0.1
            ),
            'prime_editing': DomainThresholds(
                min_efficiency=20.0,
                min_precision=20.0,
                min_conversion=0.0,  # Not applicable for Prime Editing
                required_fields=10,
                min_completeness=0.85,
                human_priority_bonus=0.15
            ),
            'base_editing': DomainThresholds(
                min_efficiency=15.0,
                min_precision=15.0,
                min_conversion=15.0,
                required_fields=9,
                min_completeness=0.8,
                human_priority_bonus=0.1
            )
        }

        # Required fields for each domain
        self.required_fields = {
            'crispr': [
                'title', 'abstract', 'year', 'authors', 'journal',
                'efficiency', 'organism', 'target_gene', 'experimental_method'
            ],
            'prime_editing': [
                'title', 'abstract', 'year', 'authors', 'journal',
                'efficiency', 'precision', 'organism', 'target_gene',
                'pegRNA_design', 'experimental_method'
            ],
            'base_editing': [
                'title', 'abstract', 'year', 'authors', 'journal',
                'efficiency', 'conversion_rate', 'organism', 'target_gene',
                'base_editor_type', 'experimental_method'
            ]
        }

        # Human organism identifiers
        self.human_identifiers = [
            'homo sapiens', 'human', 'human cell', 'human cells',
            'human tissue', 'human organoid', 'human patient',
            'clinical trial', 'therapeutic'
        ]

        # Disease-related keywords for therapeutic relevance
        self.therapeutic_keywords = [
            'cancer', 'tumor', 'leukemia', 'lymphoma', 'sarcoma',
            'cystic fibrosis', 'sickle cell', 'thalassemia',
            'huntington', 'parkinson', 'alzheimer', 'diabetes',
            'cardiovascular', 'muscular dystrophy', 'hemophilia',
            'therapeutic', 'treatment', 'therapy', 'clinical',
            'disease', 'disorder', 'syndrome'
        ]

    def assess_compliance_rate(self, domain: str, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive MVDP compliance rates per domain

        Args:
            domain: Gene editing domain (crispr, prime_editing, base_editing)
            data_points: List of data point dictionaries

        Returns:
            Dictionary with compliance metrics and analysis
        """
        if not data_points:
            return {
                'compliance_rate': 0.0,
                'total_points': 0,
                'valid_points': 0,
                'quality_distribution': {},
                'organism_distribution': {},
                'therapeutic_relevance_rate': 0.0,
                'common_issues': []
            }

        valid_points = 0
        quality_scores = []
        organism_counts = {}
        therapeutic_count = 0
        all_issues = []

        for dp in data_points:
            metrics = self._validate_data_point(domain, dp)

            if metrics.overall_score >= self.quality_thresholds[domain].min_completeness:
                valid_points += 1

            quality_scores.append(metrics.overall_score)
            organism_counts[metrics.organism_type] = organism_counts.get(metrics.organism_type, 0) + 1

            if metrics.therapeutic_relevance:
                therapeutic_count += 1

            all_issues.extend(metrics.quality_issues)

        # Calculate compliance rate
        compliance_rate = (valid_points / len(data_points)) * 100

        # Quality distribution analysis
        quality_distribution = self._analyze_quality_distribution(quality_scores)

        # Common issues analysis
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'compliance_rate': compliance_rate,
            'total_points': len(data_points),
            'valid_points': valid_points,
            'quality_distribution': quality_distribution,
            'organism_distribution': organism_counts,
            'therapeutic_relevance_rate': (therapeutic_count / len(data_points)) * 100,
            'common_issues': common_issues,
            'average_quality_score': statistics.mean(quality_scores) if quality_scores else 0.0,
            'quality_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
        }

    def _validate_data_point(self, domain: str, data_point: Dict[str, Any]) -> QualityMetrics:
        """
        Validate individual data point with domain-specific criteria

        Args:
            domain: Gene editing domain
            data_point: Data point dictionary

        Returns:
            QualityMetrics object with comprehensive assessment
        """
        missing_fields = []
        quality_issues = []

        # Check required fields
        required = self.required_fields[domain]
        for field in required:
            if not self._has_valid_field(data_point, field):
                missing_fields.append(field)

        # Calculate completeness score
        completeness_score = (len(required) - len(missing_fields)) / len(required)

        # Check efficiency metrics
        accuracy_score = self._assess_efficiency_accuracy(domain, data_point)

        # Assess organism and therapeutic relevance
        organism_type = self._classify_organism(data_point)
        therapeutic_relevance = self._assess_therapeutic_relevance(data_point)

        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(
            domain, data_point, organism_type, therapeutic_relevance
        )

        # Overall score with domain-specific weighting
        overall_score = self._calculate_overall_score(
            completeness_score, accuracy_score, relevance_score,
            organism_type, therapeutic_relevance, domain
        )

        # Identify quality issues
        quality_issues = self._identify_quality_issues(
            domain, data_point, missing_fields, accuracy_score
        )

        return QualityMetrics(
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            relevance_score=relevance_score,
            overall_score=overall_score,
            missing_fields=missing_fields,
            quality_issues=quality_issues,
            organism_type=organism_type,
            therapeutic_relevance=therapeutic_relevance
        )

    def _has_valid_field(self, data_point: Dict[str, Any], field: str) -> bool:
        """Check if field exists and has valid content"""
        if field not in data_point:
            return False

        value = data_point[field]
        if value is None:
            return False

        if isinstance(value, str) and not value.strip():
            return False

        if isinstance(value, (int, float)) and value <= 0:
            return False

        return True

    def _assess_efficiency_accuracy(self, domain: str, data_point: Dict[str, Any]) -> float:
        """Assess accuracy of efficiency metrics based on domain"""
        thresholds = self.quality_thresholds[domain]

        if domain == 'crispr':
            efficiency = data_point.get('efficiency', 0)
            return 1.0 if efficiency >= thresholds.min_efficiency else 0.5

        elif domain == 'prime_editing':
            efficiency = data_point.get('efficiency', 0)
            precision = data_point.get('precision', 0)
            return 1.0 if (efficiency >= thresholds.min_efficiency and
                          precision >= thresholds.min_precision) else 0.5

        elif domain == 'base_editing':
            efficiency = data_point.get('efficiency', 0)
            conversion = data_point.get('conversion_rate', 0)
            return 1.0 if (efficiency >= thresholds.min_efficiency and
                          conversion >= thresholds.min_conversion) else 0.5

        return 0.5

    def _classify_organism(self, data_point: Dict[str, Any]) -> str:
        """Classify organism type from data point"""
        text_content = ' '.join([
            str(data_point.get('title', '')),
            str(data_point.get('abstract', '')),
            str(data_point.get('organism', ''))
        ]).lower()

        # Check for human identifiers
        for identifier in self.human_identifiers:
            if identifier in text_content:
                return 'human'

        # Check for model organisms
        if any(org in text_content for org in ['mouse', 'rat', 'murine']):
            return 'mouse'
        elif any(org in text_content for org in ['drosophila', 'fruit fly']):
            return 'drosophila'
        elif any(org in text_content for org in ['c. elegans', 'nematode']):
            return 'c_elegans'
        elif any(org in text_content for org in ['zebrafish', 'danio']):
            return 'zebrafish'
        elif any(org in text_content for org in ['yeast', 'saccharomyces']):
            return 'yeast'
        else:
            return 'other'

    def _assess_therapeutic_relevance(self, data_point: Dict[str, Any]) -> bool:
        """Assess therapeutic relevance based on disease keywords"""
        text_content = ' '.join([
            str(data_point.get('title', '')),
            str(data_point.get('abstract', ''))
        ]).lower()

        return any(keyword in text_content for keyword in self.therapeutic_keywords)

    def _calculate_relevance_score(self, domain: str, data_point: Dict[str, Any],
                                 organism_type: str, therapeutic_relevance: bool) -> float:
        """Calculate relevance score based on organism and therapeutic focus"""
        score = 0.5  # Base score

        # Human organism bonus
        if organism_type == 'human':
            score += 0.3

        # Therapeutic relevance bonus
        if therapeutic_relevance:
            score += 0.2

        # Domain-specific relevance adjustments
        if domain == 'prime_editing' and therapeutic_relevance:
            score += 0.1  # Prime editing has high therapeutic potential

        return min(score, 1.0)

    def _calculate_overall_score(self, completeness: float, accuracy: float,
                               relevance: float, organism_type: str,
                               therapeutic_relevance: bool, domain: str) -> float:
        """Calculate overall quality score with domain-specific weighting"""
        thresholds = self.quality_thresholds[domain]

        # Weighted score (completeness 40%, accuracy 30%, relevance 30%)
        base_score = (completeness * 0.4 + accuracy * 0.3 + relevance * 0.3)

        # Human priority bonus
        if organism_type == 'human':
            base_score += thresholds.human_priority_bonus

        # Therapeutic relevance bonus
        if therapeutic_relevance:
            base_score += 0.05

        return min(base_score, 1.0)

    def _identify_quality_issues(self, domain: str, data_point: Dict[str, Any],
                               missing_fields: List[str], accuracy_score: float) -> List[str]:
        """Identify specific quality issues"""
        issues = []

        # Missing required fields
        if missing_fields:
            issues.append(f"Missing required fields: {', '.join(missing_fields)}")

        # Low efficiency metrics
        if accuracy_score < 0.5:
            issues.append("Efficiency metrics below domain thresholds")

        # Missing organism information
        if not self._has_valid_field(data_point, 'organism'):
            issues.append("Missing organism information")

        # Missing experimental details
        if not self._has_valid_field(data_point, 'experimental_method'):
            issues.append("Missing experimental methodology")

        return issues

    def _analyze_quality_distribution(self, quality_scores: List[float]) -> Dict[str, int]:
        """Analyze distribution of quality scores"""
        distribution = {
            'excellent': 0,  # 0.9-1.0
            'good': 0,       # 0.7-0.89
            'fair': 0,       # 0.5-0.69
            'poor': 0        # 0.0-0.49
        }

        for score in quality_scores:
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.7:
                distribution['good'] += 1
            elif score >= 0.5:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1

        return distribution

    def generate_quality_report(self, domain: str, compliance_results: Dict[str, Any]) -> str:
        """Generate comprehensive quality report"""
        report = f"""
=== MVDP Quality Report for {domain.upper()} Domain ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

COMPLIANCE METRICS:
- Compliance Rate: {compliance_results['compliance_rate']:.2f}%
- Total Data Points: {compliance_results['total_points']}
- Valid Data Points: {compliance_results['valid_points']}
- Average Quality Score: {compliance_results['average_quality_score']:.3f}
- Quality Standard Deviation: {compliance_results['quality_std']:.3f}

QUALITY DISTRIBUTION:
- Excellent (0.9-1.0): {compliance_results['quality_distribution']['excellent']}
- Good (0.7-0.89): {compliance_results['quality_distribution']['good']}
- Fair (0.5-0.69): {compliance_results['quality_distribution']['fair']}
- Poor (0.0-0.49): {compliance_results['quality_distribution']['poor']}

ORGANISM DISTRIBUTION:
"""
        for organism, count in compliance_results['organism_distribution'].items():
            percentage = (count / compliance_results['total_points']) * 100
            report += f"- {organism.title()}: {count} ({percentage:.1f}%)\n"

        report += f"""
THERAPEUTIC RELEVANCE:
- Therapeutic Studies: {compliance_results['therapeutic_relevance_rate']:.1f}%

TOP QUALITY ISSUES:
"""
        for issue, count in compliance_results['common_issues']:
            percentage = (count / compliance_results['total_points']) * 100
            report += f"- {issue}: {count} occurrences ({percentage:.1f}%)\n"

        return report
