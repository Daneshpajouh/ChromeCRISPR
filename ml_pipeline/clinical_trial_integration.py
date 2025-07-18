"""
Clinical Trial Integration Module
Based on GeneX Phase 1 Research Report 3/3

This module implements specialized extraction for clinical trial registries
to integrate translational and human safety data into the GeneX knowledge base.
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import requests
from urllib.parse import urlencode
import time

logger = logging.getLogger(__name__)


class TrialPhase(Enum):
    """Clinical trial phases as defined in the research report."""
    PHASE_1 = "Phase 1"
    PHASE_2 = "Phase 2"
    PHASE_3 = "Phase 3"
    PHASE_4 = "Phase 4"
    PHASE_1_2 = "Phase 1/2"
    PHASE_2_3 = "Phase 2/3"
    EARLY_PHASE_1 = "Early Phase 1"
    NOT_APPLICABLE = "N/A"


class TrialStatus(Enum):
    """Clinical trial status."""
    RECRUITING = "Recruiting"
    ACTIVE_NOT_RECRUITING = "Active, not recruiting"
    ENROLLING_BY_INVITATION = "Enrolling by invitation"
    NOT_YET_RECRUITING = "Not yet recruiting"
    SUSPENDED = "Suspended"
    TERMINATED = "Terminated"
    COMPLETED = "Completed"
    WITHDRAWN = "Withdrawn"
    UNKNOWN = "Unknown"


@dataclass
class ClinicalTrialMetadata:
    """Metadata for clinical trials."""
    trial_id: str
    title: str
    brief_title: str
    official_title: str
    phase: TrialPhase
    status: TrialStatus
    start_date: str
    completion_date: str
    primary_completion_date: str
    enrollment: int
    sponsor: str
    collaborators: List[str]
    principal_investigator: str
    study_type: str
    allocation: str
    intervention_model: str
    masking: str
    primary_purpose: str
    conditions: List[str]
    interventions: List[str]
    outcome_measures: List[str]
    eligibility_criteria: str
    locations: List[str]
    countries: List[str]


@dataclass
class TrialOutcome:
    """Clinical trial outcome data."""
    outcome_measure: str
    outcome_type: str  # primary, secondary, other
    outcome_description: str
    time_frame: str
    safety_issue: bool
    reported_value: str
    statistical_analysis: str
    p_value: Optional[float] = None
    confidence_interval: Optional[str] = None


@dataclass
class ClinicalTrialDocument:
    """Complete clinical trial document with structured extraction."""
    metadata: ClinicalTrialMetadata
    outcomes: List[TrialOutcome]
    adverse_events: List[Dict[str, Any]]
    gene_editing_technologies: List[str]
    therapeutic_applications: List[str]
    delivery_methods: List[str]
    safety_data: Dict[str, Any]
    efficacy_data: Dict[str, Any]
    extracted_entities: Dict[str, List[str]]
    extracted_relations: List[Dict[str, str]]


class ClinicalTrialsAPIClient:
    """Client for ClinicalTrials.gov API."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = "https://clinicaltrials.gov/api/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GeneX-Clinical-Trials/1.0'
        })

    def search_gene_editing_trials(self, max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Search for gene editing related clinical trials.

        Args:
            max_results: Maximum number of trials to retrieve

        Returns:
            List of trial metadata
        """
        search_terms = [
            "CRISPR",
            "gene editing",
            "genome editing",
            "prime editing",
            "base editing",
            "TALEN",
            "ZFN",
            "nuclease",
            "gene therapy"
        ]

        all_trials = []

        for term in search_terms:
            try:
                params = {
                    'query': f'"{term}"',
                    'fields': 'NCTId,BriefTitle,OfficialTitle,Phase,Status,StartDate,CompletionDate,Enrollment,SponsorName,Condition,InterventionName',
                    'min_rnk': 1,
                    'max_rnk': min(max_results // len(search_terms), 100),
                    'fmt': 'json'
                }

                response = self.session.get(f"{self.base_url}/studies", params=params)
                response.raise_for_status()

                data = response.json()
                if 'studies' in data:
                    all_trials.extend(data['studies'])

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error searching ClinicalTrials.gov for '{term}': {e}")

        return all_trials[:max_results]

    def get_trial_details(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific clinical trial.

        Args:
            trial_id: NCT ID of the trial

        Returns:
            Detailed trial information or None if not found
        """
        try:
            params = {
                'fields': 'NCTId,BriefTitle,OfficialTitle,Phase,Status,StartDate,CompletionDate,PrimaryCompletionDate,Enrollment,SponsorName,CollaboratorName,PrincipalInvestigator,StudyType,Allocation,InterventionModel,Masking,PrimaryPurpose,Condition,InterventionName,OutcomeMeasure,EligibilityCriteria,LocationCountry'
            }

            response = self.session.get(f"{self.base_url}/studies/{trial_id}", params=params)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error getting trial details for {trial_id}: {e}")
            return None

    def get_trial_outcomes(self, trial_id: str) -> List[Dict[str, Any]]:
        """
        Get outcome data for a clinical trial.

        Args:
            trial_id: NCT ID of the trial

        Returns:
            List of trial outcomes
        """
        try:
            params = {
                'fields': 'OutcomeMeasure,OutcomeType,OutcomeDescription,TimeFrame,SafetyIssue,OutcomeValue,StatisticalAnalysis,PValue,ConfidenceInterval'
            }

            response = self.session.get(f"{self.base_url}/studies/{trial_id}/outcomes", params=params)
            response.raise_for_status()

            data = response.json()
            return data.get('outcomes', [])

        except Exception as e:
            logger.error(f"Error getting trial outcomes for {trial_id}: {e}")
            return []


class ClinicalTrialExtractor:
    """
    Specialized extractor for clinical trial data.

    Based on the research report recommendation to integrate clinical trial
    registries for translational and human safety data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def extract_trial_entities(self, trial_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract entities from clinical trial data.

        Args:
            trial_data: Raw trial data from API

        Returns:
            Dictionary of extracted entities by category
        """
        entities = {}

        # Extract gene editing technologies
        entities['gene_editing_technologies'] = self._extract_gene_editing_tech(trial_data)

        # Extract therapeutic applications
        entities['therapeutic_applications'] = self._extract_therapeutic_apps(trial_data)

        # Extract delivery methods
        entities['delivery_methods'] = self._extract_delivery_methods(trial_data)

        # Extract safety indicators
        entities['safety_indicators'] = self._extract_safety_indicators(trial_data)

        # Extract efficacy indicators
        entities['efficacy_indicators'] = self._extract_efficacy_indicators(trial_data)

        return entities

    def _extract_gene_editing_tech(self, trial_data: Dict[str, Any]) -> List[str]:
        """Extract gene editing technologies from trial data."""
        import re

        technologies = []

        # Check intervention names
        interventions = trial_data.get('interventions', [])
        for intervention in interventions:
            intervention_name = intervention.get('name', '')

            # CRISPR-related
            if any(term in intervention_name.lower() for term in ['crispr', 'cas9', 'cas12', 'cas13']):
                technologies.append(intervention_name)

            # Base editing
            if any(term in intervention_name.lower() for term in ['base editor', 'cbe', 'abe']):
                technologies.append(intervention_name)

            # Prime editing
            if any(term in intervention_name.lower() for term in ['prime editor', 'peg']):
                technologies.append(intervention_name)

            # Other nucleases
            if any(term in intervention_name.lower() for term in ['talen', 'zfn', 'nuclease']):
                technologies.append(intervention_name)

        # Check title and description
        title = trial_data.get('brief_title', '') + ' ' + trial_data.get('official_title', '')

        crispr_patterns = [
            r'CRISPR[-\s]?Cas\d*',
            r'clustered regularly interspaced short palindromic repeats',
            r'guide RNA',
            r'sgRNA'
        ]

        for pattern in crispr_patterns:
            matches = re.findall(pattern, title, re.IGNORECASE)
            technologies.extend(matches)

        base_editing_patterns = [
            r'base editor',
            r'cytosine base editor',
            r'adenine base editor',
            r'BE\d*',
            r'CBE',
            r'ABE'
        ]

        for pattern in base_editing_patterns:
            matches = re.findall(pattern, title, re.IGNORECASE)
            technologies.extend(matches)

        return list(set(technologies))

    def _extract_therapeutic_apps(self, trial_data: Dict[str, Any]) -> List[str]:
        """Extract therapeutic applications from trial data."""
        applications = []

        # Extract from conditions
        conditions = trial_data.get('conditions', [])
        for condition in conditions:
            if isinstance(condition, dict):
                applications.append(condition.get('name', ''))
            else:
                applications.append(condition)

        return list(set(applications))

    def _extract_delivery_methods(self, trial_data: Dict[str, Any]) -> List[str]:
        """Extract delivery methods from trial data."""
        import re

        delivery_methods = []

        # Check intervention descriptions
        interventions = trial_data.get('interventions', [])
        for intervention in interventions:
            description = intervention.get('description', '')

            # Viral vectors
            viral_patterns = [
                r'AAV',
                r'adeno-associated virus',
                r'lentivirus',
                r'retrovirus',
                r'viral vector'
            ]

            for pattern in viral_patterns:
                if re.search(pattern, description, re.IGNORECASE):
                    delivery_methods.append(pattern)

            # Non-viral methods
            nonviral_patterns = [
                r'lipid nanoparticle',
                r'LNP',
                r'electroporation',
                r'microinjection',
                r'nanoparticle'
            ]

            for pattern in nonviral_patterns:
                if re.search(pattern, description, re.IGNORECASE):
                    delivery_methods.append(pattern)

        return list(set(delivery_methods))

    def _extract_safety_indicators(self, trial_data: Dict[str, Any]) -> List[str]:
        """Extract safety indicators from trial data."""
        safety_indicators = []

        # Check outcome measures for safety endpoints
        outcomes = trial_data.get('outcomes', [])
        for outcome in outcomes:
            measure = outcome.get('measure', '')
            if any(term in measure.lower() for term in ['safety', 'adverse', 'toxicity', 'side effect']):
                safety_indicators.append(measure)

        return safety_indicators

    def _extract_efficacy_indicators(self, trial_data: Dict[str, Any]) -> List[str]:
        """Extract efficacy indicators from trial data."""
        efficacy_indicators = []

        # Check outcome measures for efficacy endpoints
        outcomes = trial_data.get('outcomes', [])
        for outcome in outcomes:
            measure = outcome.get('measure', '')
            if any(term in measure.lower() for term in ['efficacy', 'effectiveness', 'response', 'improvement']):
                efficacy_indicators.append(measure)

        return efficacy_indicators

    def extract_trial_relations(self, trial_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract relations from clinical trial data.

        Args:
            trial_data: Raw trial data from API

        Returns:
            List of extracted relations
        """
        relations = []

        # Technology-application relations
        tech_app_relations = self._extract_tech_application_relations(trial_data)
        relations.extend(tech_app_relations)

        # Technology-delivery relations
        tech_delivery_relations = self._extract_tech_delivery_relations(trial_data)
        relations.extend(tech_delivery_relations)

        # Safety relations
        safety_relations = self._extract_safety_relations(trial_data)
        relations.extend(safety_relations)

        return relations

    def _extract_tech_application_relations(self, trial_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract technology-application relations."""
        relations = []

        technologies = self._extract_gene_editing_tech(trial_data)
        applications = self._extract_therapeutic_apps(trial_data)

        for tech in technologies:
            for app in applications:
                relations.append({
                    'subject': tech,
                    'predicate': 'TREATS',
                    'object': app,
                    'source_type': 'clinical_trial',
                    'source_id': trial_data.get('nct_id', ''),
                    'phase': trial_data.get('phase', ''),
                    'status': trial_data.get('status', '')
                })

        return relations

    def _extract_tech_delivery_relations(self, trial_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract technology-delivery method relations."""
        relations = []

        technologies = self._extract_gene_editing_tech(trial_data)
        delivery_methods = self._extract_delivery_methods(trial_data)

        for tech in technologies:
            for method in delivery_methods:
                relations.append({
                    'subject': tech,
                    'predicate': 'DELIVERED_BY',
                    'object': method,
                    'source_type': 'clinical_trial',
                    'source_id': trial_data.get('nct_id', ''),
                    'phase': trial_data.get('phase', ''),
                    'status': trial_data.get('status', '')
                })

        return relations

    def _extract_safety_relations(self, trial_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract safety-related relations."""
        relations = []

        technologies = self._extract_gene_editing_tech(trial_data)
        safety_indicators = self._extract_safety_indicators(trial_data)

        for tech in technologies:
            for indicator in safety_indicators:
                relations.append({
                    'subject': tech,
                    'predicate': 'HAS_SAFETY_MEASURE',
                    'object': indicator,
                    'source_type': 'clinical_trial',
                    'source_id': trial_data.get('nct_id', ''),
                    'phase': trial_data.get('phase', ''),
                    'status': trial_data.get('status', '')
                })

        return relations


class ClinicalTrialIntegrationPipeline:
    """
    Main pipeline for integrating clinical trial data into the GeneX knowledge base.

    This implements the research report recommendation to integrate clinical
    trial registries for translational and human safety data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = ClinicalTrialsAPIClient(config)
        self.extractor = ClinicalTrialExtractor(config)

    def search_gene_editing_trials(self, max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Search for gene editing related clinical trials.

        Args:
            max_results: Maximum number of trials to retrieve

        Returns:
            List of trial metadata
        """
        logger.info("Searching ClinicalTrials.gov for gene editing trials...")
        trials = self.client.search_gene_editing_trials(max_results)
        logger.info(f"Found {len(trials)} gene editing trials")
        return trials

    def process_trial_document(self, trial_data: Dict[str, Any]) -> Optional[ClinicalTrialDocument]:
        """
        Process a clinical trial document and extract structured information.

        Args:
            trial_data: Raw trial data from API

        Returns:
            Structured ClinicalTrialDocument or None if processing fails
        """
        try:
            # Extract metadata
            metadata = self._extract_metadata(trial_data)

            # Get detailed trial information
            trial_id = trial_data.get('nct_id', '')
            detailed_data = self.client.get_trial_details(trial_id)
            if detailed_data:
                trial_data.update(detailed_data)

            # Get outcomes
            outcomes = self.client.get_trial_outcomes(trial_id)
            trial_outcomes = self._extract_outcomes(outcomes)

            # Extract entities and relations
            entities = self.extractor.extract_trial_entities(trial_data)
            relations = self.extractor.extract_trial_relations(trial_data)

            # Extract safety and efficacy data
            safety_data = self._extract_safety_data(trial_data, outcomes)
            efficacy_data = self._extract_efficacy_data(trial_data, outcomes)

            # Create trial document
            trial_doc = ClinicalTrialDocument(
                metadata=metadata,
                outcomes=trial_outcomes,
                adverse_events=[],  # Would need additional API calls
                gene_editing_technologies=entities.get('gene_editing_technologies', []),
                therapeutic_applications=entities.get('therapeutic_applications', []),
                delivery_methods=entities.get('delivery_methods', []),
                safety_data=safety_data,
                efficacy_data=efficacy_data,
                extracted_entities=entities,
                extracted_relations=relations
            )

            return trial_doc

        except Exception as e:
            logger.error(f"Error processing trial document: {e}")
            return None

    def _extract_metadata(self, trial_data: Dict[str, Any]) -> ClinicalTrialMetadata:
        """Extract metadata from trial data."""
        return ClinicalTrialMetadata(
            trial_id=trial_data.get('nct_id', ''),
            title=trial_data.get('brief_title', ''),
            brief_title=trial_data.get('brief_title', ''),
            official_title=trial_data.get('official_title', ''),
            phase=TrialPhase(trial_data.get('phase', 'N/A')),
            status=TrialStatus(trial_data.get('status', 'Unknown')),
            start_date=trial_data.get('start_date', ''),
            completion_date=trial_data.get('completion_date', ''),
            primary_completion_date=trial_data.get('primary_completion_date', ''),
            enrollment=trial_data.get('enrollment', 0),
            sponsor=trial_data.get('sponsor_name', ''),
            collaborators=trial_data.get('collaborator_name', []),
            principal_investigator=trial_data.get('principal_investigator', ''),
            study_type=trial_data.get('study_type', ''),
            allocation=trial_data.get('allocation', ''),
            intervention_model=trial_data.get('intervention_model', ''),
            masking=trial_data.get('masking', ''),
            primary_purpose=trial_data.get('primary_purpose', ''),
            conditions=trial_data.get('conditions', []),
            interventions=trial_data.get('interventions', []),
            outcome_measures=trial_data.get('outcomes', []),
            eligibility_criteria=trial_data.get('eligibility_criteria', ''),
            locations=trial_data.get('locations', []),
            countries=trial_data.get('countries', [])
        )

    def _extract_outcomes(self, outcomes_data: List[Dict[str, Any]]) -> List[TrialOutcome]:
        """Extract trial outcomes."""
        outcomes = []

        for outcome_data in outcomes_data:
            outcome = TrialOutcome(
                outcome_measure=outcome_data.get('measure', ''),
                outcome_type=outcome_data.get('type', ''),
                outcome_description=outcome_data.get('description', ''),
                time_frame=outcome_data.get('time_frame', ''),
                safety_issue=outcome_data.get('safety_issue', False),
                reported_value=outcome_data.get('value', ''),
                statistical_analysis=outcome_data.get('statistical_analysis', ''),
                p_value=outcome_data.get('p_value'),
                confidence_interval=outcome_data.get('confidence_interval')
            )
            outcomes.append(outcome)

        return outcomes

    def _extract_safety_data(self, trial_data: Dict[str, Any],
                           outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract safety-related data."""
        safety_data = {
            'safety_outcomes': [],
            'adverse_events': [],
            'safety_monitoring': []
        }

        for outcome in outcomes:
            if outcome.get('safety_issue', False):
                safety_data['safety_outcomes'].append(outcome)

        return safety_data

    def _extract_efficacy_data(self, trial_data: Dict[str, Any],
                             outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract efficacy-related data."""
        efficacy_data = {
            'primary_outcomes': [],
            'secondary_outcomes': [],
            'efficacy_measures': []
        }

        for outcome in outcomes:
            outcome_type = outcome.get('type', '').lower()
            if 'primary' in outcome_type:
                efficacy_data['primary_outcomes'].append(outcome)
            elif 'secondary' in outcome_type:
                efficacy_data['secondary_outcomes'].append(outcome)

        return efficacy_data

    def integrate_with_academic_literature(self, trial_docs: List[ClinicalTrialDocument],
                                         academic_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Integrate clinical trial data with academic literature facts.

        Args:
            trial_docs: List of processed trial documents
            academic_facts: List of facts from academic literature

        Returns:
            Integrated facts with trial-academic links
        """
        integrated_facts = []

        for trial_doc in trial_docs:
            # Create trial-specific facts
            trial_facts = self._create_trial_facts(trial_doc)
            integrated_facts.extend(trial_facts)

            # Link to academic literature
            academic_links = self._find_academic_links(trial_doc, academic_facts)
            integrated_facts.extend(academic_links)

        return integrated_facts

    def _create_trial_facts(self, trial_doc: ClinicalTrialDocument) -> List[Dict[str, Any]]:
        """Create facts from trial document."""
        facts = []

        # Technology facts
        for tech in trial_doc.gene_editing_technologies:
            facts.append({
                'subject': tech,
                'predicate': 'TESTED_IN_CLINICAL_TRIAL',
                'object': trial_doc.metadata.trial_id,
                'source_type': 'clinical_trial',
                'source_id': trial_doc.metadata.trial_id,
                'publication_date': trial_doc.metadata.start_date,
                'phase': trial_doc.metadata.phase.value,
                'status': trial_doc.metadata.status.value
            })

        # Application facts
        for app in trial_doc.therapeutic_applications:
            facts.append({
                'subject': trial_doc.gene_editing_technologies[0] if trial_doc.gene_editing_technologies else 'Gene Editing Technology',
                'predicate': 'TREATS',
                'object': app,
                'source_type': 'clinical_trial',
                'source_id': trial_doc.metadata.trial_id,
                'publication_date': trial_doc.metadata.start_date,
                'phase': trial_doc.metadata.phase.value,
                'status': trial_doc.metadata.status.value
            })

        # Safety facts
        if trial_doc.safety_data.get('safety_outcomes'):
            facts.append({
                'subject': trial_doc.gene_editing_technologies[0] if trial_doc.gene_editing_technologies else 'Gene Editing Technology',
                'predicate': 'HAS_SAFETY_DATA',
                'object': f"{len(trial_doc.safety_data['safety_outcomes'])} safety outcomes",
                'source_type': 'clinical_trial',
                'source_id': trial_doc.metadata.trial_id,
                'publication_date': trial_doc.metadata.start_date,
                'phase': trial_doc.metadata.phase.value,
                'status': trial_doc.metadata.status.value
            })

        return facts

    def _find_academic_links(self, trial_doc: ClinicalTrialDocument,
                           academic_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find links between clinical trial and academic literature."""
        links = []

        # Simple string matching for now - could be enhanced with semantic similarity
        for fact in academic_facts:
            for tech in trial_doc.gene_editing_technologies:
                if tech.lower() in fact.get('subject', '').lower():
                    links.append({
                        'subject': trial_doc.metadata.trial_id,
                        'predicate': 'RELATES_TO_ACADEMIC',
                        'object': fact.get('source_id', ''),
                        'source_type': 'integration',
                        'source_id': f"link_{trial_doc.metadata.trial_id}_{fact.get('source_id', '')}",
                        'publication_date': trial_doc.metadata.start_date,
                        'phase': trial_doc.metadata.phase.value,
                        'status': trial_doc.metadata.status.value
                    })

        return links
