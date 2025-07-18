"""
Data Validator for GeneX Phase 1
Comprehensive validation of paper data quality and completeness
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validator for paper metadata"""

    def __init__(self):
        """Initialize the data validator with validation rules"""
        self.validation_rules = {
            'required_fields': ['title', 'authors', 'year'],
            'optional_fields': ['abstract', 'doi', 'journal', 'publication_date'],
            'min_title_length': 10,
            'max_title_length': 500,
            'min_abstract_length': 50,
            'max_abstract_length': 5000,
            'valid_year_range': (1900, 2030),
            'min_authors': 1,
            'max_authors': 100
        }

    def validate_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single paper against all validation rules

        Args:
            paper: Paper data dictionary

        Returns:
            Dictionary with validation results including 'is_valid' and 'score'
        """
        try:
            validation_results = {
                'required_fields': self._validate_required_fields(paper),
                'title_validation': self._validate_title(paper),
                'abstract_validation': self._validate_abstract(paper),
                'year_validation': self._validate_year(paper),
                'authors_validation': self._validate_authors(paper),
                'doi_validation': self._validate_doi(paper),
                'content_validation': self._validate_content(paper)
            }

            # Calculate overall validation score
            passed_validations = sum(validation_results.values())
            total_validations = len(validation_results)
            validation_score = passed_validations / total_validations if total_validations > 0 else 0.0

            # All validations must pass for paper to be considered valid
            is_valid = all(validation_results.values())

            if not is_valid:
                failed_validations = [k for k, v in validation_results.items() if not v]
                logger.debug(f"Paper validation failed for: {paper.get('title', 'Unknown')}")
                logger.debug(f"Failed validations: {failed_validations}")

            return {
                'is_valid': is_valid,
                'score': validation_score,
                'errors': [k for k, v in validation_results.items() if not v],
                'details': validation_results
            }

        except Exception as e:
            logger.error(f"Error during paper validation: {e}")
            return {
                'is_valid': False,
                'score': 0.0,
                'errors': ['validation_error'],
                'details': {}
            }

    def _validate_required_fields(self, paper: Dict[str, Any]) -> bool:
        """Validate that all required fields are present and non-empty"""
        for field in self.validation_rules['required_fields']:
            value = paper.get(field)
            if not value:
                logger.debug(f"Missing required field: {field}")
                return False

            # Check for empty strings, empty lists, etc.
            if isinstance(value, str) and not value.strip():
                logger.debug(f"Empty required field: {field}")
                return False
            elif isinstance(value, list) and len(value) == 0:
                logger.debug(f"Empty required list field: {field}")
                return False

        return True

    def _validate_title(self, paper: Dict[str, Any]) -> bool:
        """Validate paper title"""
        title = paper.get('title', '')
        if not title:
            return False

        title_length = len(title.strip())

        # Check length constraints
        if title_length < self.validation_rules['min_title_length']:
            logger.debug(f"Title too short: {title_length} characters")
            return False

        if title_length > self.validation_rules['max_title_length']:
            logger.debug(f"Title too long: {title_length} characters")
            return False

        # Check for suspicious patterns
        if self._contains_suspicious_patterns(title):
            logger.debug(f"Title contains suspicious patterns: {title}")
            return False

        return True

    def _validate_abstract(self, paper: Dict[str, Any]) -> bool:
        """Validate paper abstract (optional field)"""
        abstract = paper.get('abstract', '')
        if not abstract:
            return True  # Abstract is optional

        abstract_length = len(abstract.strip())

        # Check length constraints
        if abstract_length < self.validation_rules['min_abstract_length']:
            logger.debug(f"Abstract too short: {abstract_length} characters")
            return False

        if abstract_length > self.validation_rules['max_abstract_length']:
            logger.debug(f"Abstract too long: {abstract_length} characters")
            return False

        # Check for suspicious patterns
        if self._contains_suspicious_patterns(abstract):
            logger.debug(f"Abstract contains suspicious patterns")
            return False

        return True

    def _validate_year(self, paper: Dict[str, Any]) -> bool:
        """Validate publication year"""
        year = paper.get('year')
        if not year:
            return False

        try:
            year_int = int(year)
            min_year, max_year = self.validation_rules['valid_year_range']

            if not (min_year <= year_int <= max_year):
                logger.debug(f"Year out of valid range: {year_int}")
                return False

            return True
        except (ValueError, TypeError):
            logger.debug(f"Invalid year format: {year}")
            return False

    def _validate_authors(self, paper: Dict[str, Any]) -> bool:
        """Validate authors list"""
        authors = paper.get('authors', [])
        if not authors:
            return False

        if not isinstance(authors, list):
            logger.debug(f"Authors is not a list: {type(authors)}")
            return False

        author_count = len(authors)
        min_authors = self.validation_rules['min_authors']
        max_authors = self.validation_rules['max_authors']

        if author_count < min_authors:
            logger.debug(f"Too few authors: {author_count}")
            return False

        if author_count > max_authors:
            logger.debug(f"Too many authors: {author_count}")
            return False

        # Validate each author name
        for author in authors:
            if not self._validate_author_name(author):
                logger.debug(f"Invalid author name: {author}")
                return False

        return True

    def _validate_author_name(self, author: str) -> bool:
        """Validate individual author name"""
        if not author or not isinstance(author, str):
            return False

        author_clean = author.strip()
        if len(author_clean) < 2:
            return False

        # Check for reasonable author name patterns
        if not re.match(r'^[A-Za-z\s\-\.\']+$', author_clean):
            return False

        return True

    def _validate_doi(self, paper: Dict[str, Any]) -> bool:
        """Validate DOI format (optional field)"""
        doi = paper.get('doi', '')
        if not doi:
            return True  # DOI is optional

        # Basic DOI format validation
        doi_pattern = r'^10\.\d{4,}/.+$'
        if not re.match(doi_pattern, doi):
            logger.debug(f"Invalid DOI format: {doi}")
            return False

        return True

    def _validate_content(self, paper: Dict[str, Any]) -> bool:
        """Validate content quality and consistency"""
        # Check for duplicate content
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')

        # Convert to lowercase only if not None
        title_lower = title.lower() if title else ''
        abstract_lower = abstract.lower() if abstract else ''

        if title_lower and abstract_lower:
            # Check if title is contained in abstract (suspicious)
            if title_lower in abstract_lower and len(title_lower) > 20:
                logger.debug("Title appears to be duplicated in abstract")
                return False

        # Check for reasonable content ratios
        if abstract_lower and title_lower:
            title_words = len(title_lower.split())
            abstract_words = len(abstract_lower.split())

            if title_words > abstract_words:
                logger.debug("Title longer than abstract (suspicious)")
                return False

        return True

    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check for suspicious patterns in text"""
        suspicious_patterns = [
            r'\[.*?\]',  # Square brackets (often indicate placeholders)
            r'\{.*?\}',  # Curly brackets
            r'<.*?>',    # HTML tags
            r'http[s]?://',  # URLs in title/abstract
            r'www\.',    # URLs without protocol
            r'\d{4}-\d{4}',  # Year ranges (suspicious in titles)
            r'[^\w\s\-\.\',:;()]',  # Allow common scientific punctuation
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text):
                return True

        return False

    def get_validation_summary(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get validation summary for a collection of papers"""
        total_papers = len(papers)
        if total_papers == 0:
            return {'total_papers': 0, 'validation_rate': 0.0}

        valid_papers = sum(1 for paper in papers if self.validate_paper(paper))
        validation_rate = (valid_papers / total_papers) * 100

        return {
            'total_papers': total_papers,
            'valid_papers': valid_papers,
            'invalid_papers': total_papers - valid_papers,
            'validation_rate': round(validation_rate, 2)
        }

    def validate_batch(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a batch of papers and return only valid ones"""
        valid_papers = []

        for paper in papers:
            if self.validate_paper(paper):
                valid_papers.append(paper)
            else:
                logger.debug(f"Paper failed validation: {paper.get('title', 'Unknown')}")

        logger.info(f"Batch validation: {len(valid_papers)}/{len(papers)} papers passed validation")
        return valid_papers

    async def validate_paper_async(self, paper: dict) -> bool:
        """Async wrapper for validate_paper."""
        return self.validate_paper(paper)
