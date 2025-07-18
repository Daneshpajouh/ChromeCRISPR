"""
Quality Controller for GeneX Phase 1
Comprehensive quality assessment and scoring for paper data
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityController:
    """Comprehensive quality controller for paper data assessment"""

    def __init__(self):
        """Initialize the quality controller with scoring criteria"""
        self.quality_weights = {
            'completeness': 0.25,
            'content_quality': 0.30,
            'source_reliability': 0.20,
            'recency': 0.15,
            'impact': 0.10
        }

        self.completeness_criteria = {
            'title': 0.15,
            'abstract': 0.25,
            'authors': 0.15,
            'year': 0.10,
            'doi': 0.10,
            'journal': 0.10,
            'keywords': 0.05,
            'publication_date': 0.05,
            'citation_count': 0.05
        }

        self.content_quality_criteria = {
            'title_length': 0.20,
            'abstract_length': 0.30,
            'author_count': 0.15,
            'language_quality': 0.20,
            'technical_depth': 0.15
        }

        self.source_reliability_scores = {
            'PubMed': 0.95,
            'Semantic Scholar': 0.90,
            'CrossRef': 0.85,
            'bioRxiv': 0.80,
            'medRxiv': 0.80,
            'arXiv': 0.75,
            'Unknown': 0.50
        }

    def calculate_quality_score(self, paper: Dict[str, Any]) -> float:
        """
        Calculate comprehensive quality score for a paper

        Args:
            paper: Paper data dictionary

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            scores = {
                'completeness': self._calculate_completeness_score(paper),
                'content_quality': self._calculate_content_quality_score(paper),
                'source_reliability': self._calculate_source_reliability_score(paper),
                'recency': self._calculate_recency_score(paper),
                'impact': self._calculate_impact_score(paper)
            }

            # Calculate weighted average
            total_score = 0.0
            for component, score in scores.items():
                weight = self.quality_weights[component]
                total_score += score * weight

            # Ensure score is between 0 and 1
            total_score = max(0.0, min(1.0, total_score))

            # Store individual scores for debugging
            paper['quality_scores'] = scores
            paper['quality_score'] = total_score

            return total_score

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0

    def _calculate_completeness_score(self, paper: Dict[str, Any]) -> float:
        """Calculate completeness score based on available fields"""
        score = 0.0

        for field, weight in self.completeness_criteria.items():
            if self._is_field_complete(paper, field):
                score += weight

        return score

    def _is_field_complete(self, paper: Dict[str, Any], field: str) -> bool:
        """Check if a field is complete and meaningful"""
        value = paper.get(field)

        if not value:
            return False

        if isinstance(value, str):
            return len(value.strip()) > 0
        elif isinstance(value, list):
            return len(value) > 0
        elif isinstance(value, int):
            return value > 0
        elif isinstance(value, float):
            return value > 0.0

        return True

    def _calculate_content_quality_score(self, paper: Dict[str, Any]) -> float:
        """Calculate content quality score"""
        score = 0.0

        # Title length quality
        title = paper.get('title', '') or ''
        title_length = len(title.strip())
        if 20 <= title_length <= 200:
            score += self.content_quality_criteria['title_length']
        elif 10 <= title_length < 20 or 200 < title_length <= 300:
            score += self.content_quality_criteria['title_length'] * 0.7
        else:
            score += self.content_quality_criteria['title_length'] * 0.3

        # Abstract length quality
        abstract = paper.get('abstract', '') or ''
        abstract_length = len(abstract.strip())
        if 100 <= abstract_length <= 2000:
            score += self.content_quality_criteria['abstract_length']
        elif 50 <= abstract_length < 100 or 2000 < abstract_length <= 3000:
            score += self.content_quality_criteria['abstract_length'] * 0.8
        else:
            score += self.content_quality_criteria['abstract_length'] * 0.4

        # Author count quality
        authors = paper.get('authors', [])
        author_count = len(authors)
        if 1 <= author_count <= 10:
            score += self.content_quality_criteria['author_count']
        elif 11 <= author_count <= 20:
            score += self.content_quality_criteria['author_count'] * 0.8
        else:
            score += self.content_quality_criteria['author_count'] * 0.5

        # Language quality
        language_score = self._assess_language_quality(paper)
        score += language_score * self.content_quality_criteria['language_quality']

        # Technical depth
        technical_score = self._assess_technical_depth(paper)
        score += technical_score * self.content_quality_criteria['technical_depth']

        return score

    def _assess_language_quality(self, paper: Dict[str, Any]) -> float:
        """Assess language quality of the paper"""
        title = paper.get('title', '') or ''
        abstract = paper.get('abstract', '') or ''
        text = f"{title} {abstract}"
        text = text.lower()

        # Check for common quality indicators
        quality_indicators = [
            'method', 'analysis', 'results', 'conclusion', 'study', 'research',
            'experiment', 'data', 'analysis', 'evaluation', 'assessment'
        ]

        # Check for poor quality indicators
        poor_indicators = [
            'error', 'failed', 'invalid', 'missing', 'unknown', 'placeholder',
            'test', 'dummy', 'sample', 'example'
        ]

        quality_count = sum(1 for indicator in quality_indicators if indicator in text)
        poor_count = sum(1 for indicator in poor_indicators if indicator in text)

        # Calculate score
        if poor_count > 0:
            return max(0.0, 0.5 - (poor_count * 0.1))
        else:
            return min(1.0, 0.5 + (quality_count * 0.1))

    def _assess_technical_depth(self, paper: Dict[str, Any]) -> float:
        """Assess technical depth of the paper"""
        title = paper.get('title', '') or ''
        abstract = paper.get('abstract', '') or ''
        text = f"{title} {abstract}"
        text = text.lower()

        # Technical terms that indicate depth
        technical_terms = [
            'crispr', 'genome', 'editing', 'gene', 'dna', 'rna', 'protein',
            'mutation', 'sequence', 'expression', 'regulation', 'pathway',
            'mechanism', 'function', 'structure', 'analysis', 'methodology'
        ]

        technical_count = sum(1 for term in technical_terms if term in text)

        # Normalize to 0-1 scale
        return min(1.0, technical_count / 10.0)

    def _calculate_source_reliability_score(self, paper: Dict[str, Any]) -> float:
        """Calculate source reliability score"""
        source = paper.get('source', 'Unknown')
        return self.source_reliability_scores.get(source, 0.50)

    def _calculate_recency_score(self, paper: Dict[str, Any]) -> float:
        """Calculate recency score based on publication year"""
        year = paper.get('year')
        if not year:
            return 0.5  # Neutral score for unknown year

        try:
            year_int = int(year)
            current_year = datetime.now().year

            # Calculate years since publication
            years_ago = current_year - year_int

            if years_ago <= 2:
                return 1.0
            elif years_ago <= 5:
                return 0.9
            elif years_ago <= 10:
                return 0.8
            elif years_ago <= 15:
                return 0.7
            elif years_ago <= 20:
                return 0.6
            else:
                return 0.5

        except (ValueError, TypeError):
            return 0.5

    def _calculate_impact_score(self, paper: Dict[str, Any]) -> float:
        """Calculate impact score based on citation count"""
        citation_count = paper.get('citation_count', 0)

        if citation_count >= 100:
            return 1.0
        elif citation_count >= 50:
            return 0.9
        elif citation_count >= 20:
            return 0.8
        elif citation_count >= 10:
            return 0.7
        elif citation_count >= 5:
            return 0.6
        elif citation_count >= 1:
            return 0.5
        else:
            return 0.3

    def get_quality_summary(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get quality summary for a collection of papers"""
        if not papers:
            return {'total_papers': 0, 'average_quality': 0.0}

        quality_scores = []
        for paper in papers:
            score = self.calculate_quality_score(paper)
            quality_scores.append(score)

        avg_quality = sum(quality_scores) / len(quality_scores)

        # Categorize papers by quality
        high_quality = len([s for s in quality_scores if s >= 0.8])
        medium_quality = len([s for s in quality_scores if 0.6 <= s < 0.8])
        low_quality = len([s for s in quality_scores if s < 0.6])

        return {
            'total_papers': len(papers),
            'average_quality': round(avg_quality, 3),
            'quality_distribution': {
                'high_quality': high_quality,
                'medium_quality': medium_quality,
                'low_quality': low_quality
            },
            'quality_percentages': {
                'high_quality': round((high_quality / len(papers)) * 100, 1),
                'medium_quality': round((medium_quality / len(papers)) * 100, 1),
                'low_quality': round((low_quality / len(papers)) * 100, 1)
            }
        }

    def filter_by_quality(self, papers: List[Dict[str, Any]],
                         min_quality: float = 0.5) -> List[Dict[str, Any]]:
        """Filter papers by minimum quality threshold"""
        filtered_papers = []

        for paper in papers:
            quality_score = self.calculate_quality_score(paper)
            if quality_score >= min_quality:
                filtered_papers.append(paper)

        logger.info(f"Quality filtering: {len(filtered_papers)}/{len(papers)} papers passed threshold {min_quality}")
        return filtered_papers

    def get_quality_report(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed quality report for a single paper"""
        quality_score = self.calculate_quality_score(paper)

        return {
            'overall_score': round(quality_score, 3),
            'component_scores': paper.get('quality_scores', {}),
            'recommendations': self._get_quality_recommendations(paper, quality_score),
            'strengths': self._identify_strengths(paper),
            'weaknesses': self._identify_weaknesses(paper)
        }

    def _get_quality_recommendations(self, paper: Dict[str, Any], score: float) -> List[str]:
        """Get recommendations for improving paper quality"""
        recommendations = []

        if score < 0.5:
            recommendations.append("Overall quality is low - consider excluding from analysis")

        if not paper.get('abstract'):
            recommendations.append("Missing abstract - consider fetching from source")

        if not paper.get('doi'):
            recommendations.append("Missing DOI - consider manual verification")

        if not paper.get('year'):
            recommendations.append("Missing publication year - consider manual verification")

        if len(paper.get('authors', [])) == 0:
            recommendations.append("Missing authors - consider manual verification")

        return recommendations

    def _identify_strengths(self, paper: Dict[str, Any]) -> List[str]:
        """Identify strengths of the paper"""
        strengths = []

        if paper.get('abstract') and len(paper['abstract']) > 200:
            strengths.append("Comprehensive abstract")

        if paper.get('doi'):
            strengths.append("Has DOI for verification")

        if paper.get('citation_count', 0) > 10:
            strengths.append("Well-cited paper")

        if paper.get('year') and int(paper['year']) >= 2020:
            strengths.append("Recent publication")

        return strengths

    def _identify_weaknesses(self, paper: Dict[str, Any]) -> List[str]:
        """Identify weaknesses of the paper"""
        weaknesses = []

        if not paper.get('abstract'):
            weaknesses.append("Missing abstract")

        if not paper.get('doi'):
            weaknesses.append("Missing DOI")

        if not paper.get('year'):
            weaknesses.append("Missing publication year")

        if len(paper.get('authors', [])) == 0:
            weaknesses.append("Missing authors")

        if paper.get('citation_count', 0) == 0:
            weaknesses.append("No citations")

        return weaknesses

    async def assess_paper_quality_async(self, paper: dict) -> float:
        """Async wrapper for assess_paper_quality or calculate_quality_score."""
        if hasattr(self, 'assess_paper_quality'):
            return self.assess_paper_quality(paper)
        elif hasattr(self, 'calculate_quality_score'):
            return self.calculate_quality_score(paper)
        return 0.0

    def assess_paper_quality(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess paper quality and return comprehensive quality metrics

        Args:
            paper: Paper data dictionary

        Returns:
            Dictionary with quality assessment results
        """
        try:
            # Calculate overall quality score
            overall_score = self.calculate_quality_score(paper)

            # Get detailed quality report
            quality_report = self.get_quality_report(paper)

            # Get quality recommendations
            recommendations = self._get_quality_recommendations(paper, overall_score)

            # Identify strengths and weaknesses
            strengths = self._identify_strengths(paper)
            weaknesses = self._identify_weaknesses(paper)

            return {
                'overall_score': overall_score,
                'quality_level': self._get_quality_level(overall_score),
                'component_scores': paper.get('quality_scores', {}),
                'strengths': strengths,
                'weaknesses': weaknesses,
                'recommendations': recommendations,
                'assessment_date': datetime.now().isoformat(),
                'details': quality_report
            }

        except Exception as e:
            logger.error(f"Error assessing paper quality: {e}")
            return {
                'overall_score': 0.0,
                'quality_level': 'unknown',
                'component_scores': {},
                'strengths': [],
                'weaknesses': ['quality_assessment_error'],
                'recommendations': ['Unable to assess quality due to error'],
                'assessment_date': datetime.now().isoformat(),
                'details': {}
            }

    def _get_quality_level(self, score: float) -> str:
        """Convert quality score to quality level"""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'very_good'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.6:
            return 'fair'
        elif score >= 0.5:
            return 'acceptable'
        else:
            return 'poor'
