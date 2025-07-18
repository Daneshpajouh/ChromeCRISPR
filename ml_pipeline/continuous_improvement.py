"""
Continuous Improvement Module
Based on GeneX Phase 1 Research Report 3/3

This module implements the human-in-the-loop feedback mechanism and continuous
improvement flywheel recommended in the research report.
"""

import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import hashlib
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback as defined in the research report."""
    ENTITY_CORRECTION = "entity_correction"
    RELATION_CORRECTION = "relation_correction"
    MISSING_INFORMATION = "missing_information"
    FACTUAL_ERROR = "factual_error"
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"
    SOURCE_VERIFICATION = "source_verification"


class FeedbackStatus(Enum):
    """Status of feedback processing."""
    PENDING = "pending"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


@dataclass
class UserFeedback:
    """User feedback for continuous improvement."""
    feedback_id: str
    user_id: str
    feedback_type: FeedbackType
    fact_id: str
    original_fact: Dict[str, Any]
    suggested_correction: Dict[str, Any]
    feedback_text: str
    confidence_score: float
    timestamp: str
    status: FeedbackStatus
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None
    implementation_date: Optional[str] = None


@dataclass
class ModelPerformance:
    """Model performance tracking for continuous improvement."""
    model_name: str
    task_type: str  # NER, RE, summarization
    dataset: str
    performance_metrics: Dict[str, float]
    training_date: str
    evaluation_date: str
    improvement_needed: bool
    areas_for_improvement: List[str]


@dataclass
class ActiveLearningExample:
    """Example for active learning based on user feedback."""
    example_id: str
    text: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    feedback_source: str
    confidence_threshold: float
    priority_score: float
    added_date: str


class FeedbackDatabase:
    """Database for storing and managing user feedback."""

    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the feedback database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                original_fact TEXT NOT NULL,
                suggested_correction TEXT NOT NULL,
                feedback_text TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                reviewed_by TEXT,
                review_notes TEXT,
                implementation_date TEXT
            )
        ''')

        # Create model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                model_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                dataset TEXT NOT NULL,
                performance_metrics TEXT NOT NULL,
                training_date TEXT NOT NULL,
                evaluation_date TEXT NOT NULL,
                improvement_needed BOOLEAN NOT NULL,
                areas_for_improvement TEXT NOT NULL,
                PRIMARY KEY (model_name, task_type, evaluation_date)
            )
        ''')

        # Create active learning examples table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_learning_examples (
                example_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                entities TEXT NOT NULL,
                relations TEXT NOT NULL,
                feedback_source TEXT NOT NULL,
                confidence_threshold REAL NOT NULL,
                priority_score REAL NOT NULL,
                added_date TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def store_feedback(self, feedback: UserFeedback):
        """Store user feedback in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO user_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.feedback_id,
            feedback.user_id,
            feedback.feedback_type.value,
            feedback.fact_id,
            json.dumps(feedback.original_fact),
            json.dumps(feedback.suggested_correction),
            feedback.feedback_text,
            feedback.confidence_score,
            feedback.timestamp,
            feedback.status.value,
            feedback.reviewed_by,
            feedback.review_notes,
            feedback.implementation_date
        ))

        conn.commit()
        conn.close()

    def get_feedback(self, feedback_id: str) -> Optional[UserFeedback]:
        """Retrieve feedback by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM user_feedback WHERE feedback_id = ?', (feedback_id,))
        row = cursor.fetchone()

        conn.close()

        if row:
            return self._row_to_feedback(row)
        return None

    def get_pending_feedback(self) -> List[UserFeedback]:
        """Get all pending feedback for review."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM user_feedback WHERE status = ?', (FeedbackStatus.PENDING.value,))
        rows = cursor.fetchall()

        conn.close()

        return [self._row_to_feedback(row) for row in rows]

    def _row_to_feedback(self, row) -> UserFeedback:
        """Convert database row to UserFeedback object."""
        return UserFeedback(
            feedback_id=row[0],
            user_id=row[1],
            feedback_type=FeedbackType(row[2]),
            fact_id=row[3],
            original_fact=json.loads(row[4]),
            suggested_correction=json.loads(row[5]),
            feedback_text=row[6],
            confidence_score=row[7],
            timestamp=row[8],
            status=FeedbackStatus(row[9]),
            reviewed_by=row[10],
            review_notes=row[11],
            implementation_date=row[12]
        )


class ActiveLearningManager:
    """
    Manages active learning based on user feedback.

    This implements the research report recommendation for using user feedback
    as high-quality training data for model improvement.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feedback_db = FeedbackDatabase()
        self.min_confidence_threshold = config.get('active_learning', {}).get('min_confidence', 0.7)
        self.priority_weights = {
            FeedbackType.ENTITY_CORRECTION: 1.0,
            FeedbackType.RELATION_CORRECTION: 1.2,
            FeedbackType.MISSING_INFORMATION: 0.8,
            FeedbackType.FACTUAL_ERROR: 1.5,
            FeedbackType.CONFIDENCE_ADJUSTMENT: 0.6,
            FeedbackType.SOURCE_VERIFICATION: 0.7
        }

    def generate_training_examples(self, feedback: List[UserFeedback]) -> List[ActiveLearningExample]:
        """
        Generate training examples from user feedback.

        Args:
            feedback: List of approved user feedback

        Returns:
            List of active learning examples
        """
        examples = []

        for fb in feedback:
            if fb.status == FeedbackStatus.APPROVED:
                example = self._create_training_example(fb)
                if example:
                    examples.append(example)

        return examples

    def _create_training_example(self, feedback: UserFeedback) -> Optional[ActiveLearningExample]:
        """Create a training example from feedback."""
        try:
            # Extract text context from original fact
            original_fact = feedback.original_fact
            evidence_sentence = original_fact.get('evidence_metadata', {}).get('evidence_sentence', '')

            if not evidence_sentence:
                return None

            # Generate entities and relations based on feedback type
            entities = []
            relations = []

            if feedback.feedback_type == FeedbackType.ENTITY_CORRECTION:
                entities = self._extract_entities_from_correction(feedback)
            elif feedback.feedback_type == FeedbackType.RELATION_CORRECTION:
                relations = self._extract_relations_from_correction(feedback)

            # Calculate priority score
            priority_score = self._calculate_priority_score(feedback)

            # Create example
            example = ActiveLearningExample(
                example_id=self._generate_example_id(feedback),
                text=evidence_sentence,
                entities=entities,
                relations=relations,
                feedback_source=feedback.feedback_id,
                confidence_threshold=self.min_confidence_threshold,
                priority_score=priority_score,
                added_date=datetime.now().isoformat()
            )

            return example

        except Exception as e:
            logger.error(f"Error creating training example from feedback {feedback.feedback_id}: {e}")
            return None

    def _extract_entities_from_correction(self, feedback: UserFeedback) -> List[Dict[str, Any]]:
        """Extract entities from feedback correction."""
        entities = []

        # Extract from suggested correction
        correction = feedback.suggested_correction

        # Add corrected entities
        if 'subject' in correction:
            entities.append({
                'text': correction['subject'],
                'type': 'corrected_entity',
                'start': 0,  # Would need proper text alignment
                'end': len(correction['subject'])
            })

        if 'object' in correction:
            entities.append({
                'text': correction['object'],
                'type': 'corrected_entity',
                'start': 0,  # Would need proper text alignment
                'end': len(correction['object'])
            })

        return entities

    def _extract_relations_from_correction(self, feedback: UserFeedback) -> List[Dict[str, Any]]:
        """Extract relations from feedback correction."""
        relations = []

        correction = feedback.suggested_correction

        if all(key in correction for key in ['subject', 'predicate', 'object']):
            relations.append({
                'subject': correction['subject'],
                'predicate': correction['predicate'],
                'object': correction['object'],
                'type': 'corrected_relation'
            })

        return relations

    def _calculate_priority_score(self, feedback: UserFeedback) -> float:
        """Calculate priority score for training example."""
        base_weight = self.priority_weights.get(feedback.feedback_type, 1.0)
        confidence_factor = feedback.confidence_score
        time_factor = self._calculate_time_factor(feedback.timestamp)

        priority_score = base_weight * confidence_factor * time_factor
        return min(priority_score, 1.0)

    def _calculate_time_factor(self, timestamp: str) -> float:
        """Calculate time-based factor for priority scoring."""
        try:
            feedback_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            days_old = (current_time - feedback_time).days

            # Recent feedback gets higher priority
            if days_old <= 7:
                return 1.0
            elif days_old <= 30:
                return 0.9
            elif days_old <= 90:
                return 0.8
            else:
                return 0.7

        except ValueError:
            return 0.8

    def _generate_example_id(self, feedback: UserFeedback) -> str:
        """Generate unique example ID."""
        content = f"{feedback.feedback_id}_{feedback.feedback_type.value}_{feedback.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ContinuousLiteratureMonitor:
    """
    Monitors literature sources for new publications.

    This implements the research report recommendation for continuous
    literature monitoring to keep the knowledge base current.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitored_sources = config.get('continuous_monitoring', {}).get('sources', [])
        self.check_interval_hours = config.get('continuous_monitoring', {}).get('check_interval_hours', 24)
        self.last_check_file = Path("data/last_literature_check.json")
        self._init_last_check()

    def _init_last_check(self):
        """Initialize last check tracking."""
        if not self.last_check_file.exists():
            self.last_check_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_last_check(datetime.now())

    def _save_last_check(self, check_time: datetime):
        """Save last check time."""
        with open(self.last_check_file, 'w') as f:
            json.dump({
                'last_check': check_time.isoformat(),
                'next_check': (check_time + timedelta(hours=self.check_interval_hours)).isoformat()
            }, f)

    def _load_last_check(self) -> datetime:
        """Load last check time."""
        try:
            with open(self.last_check_file, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data['last_check'])
        except (FileNotFoundError, json.JSONDecodeError):
            return datetime.now() - timedelta(hours=self.check_interval_hours + 1)

    def should_check_for_updates(self) -> bool:
        """Determine if it's time to check for new literature."""
        last_check = self._load_last_check()
        next_check = last_check + timedelta(hours=self.check_interval_hours)
        return datetime.now() >= next_check

    def check_for_new_literature(self) -> List[Dict[str, Any]]:
        """
        Check for new literature from monitored sources.

        Returns:
            List of new publications found
        """
        if not self.should_check_for_updates():
            return []

        new_publications = []

        for source in self.monitored_sources:
            try:
                source_publications = self._check_source(source)
                new_publications.extend(source_publications)

            except Exception as e:
                logger.error(f"Error checking source {source}: {e}")

        # Update last check time
        self._save_last_check(datetime.now())

        logger.info(f"Found {len(new_publications)} new publications")
        return new_publications

    def _check_source(self, source: str) -> List[Dict[str, Any]]:
        """Check a specific source for new publications."""
        # This would implement source-specific checking logic
        # For now, return empty list as placeholder
        return []


class ContinuousImprovementPipeline:
    """
    Main pipeline for continuous improvement of the GeneX system.

    This implements the research report recommendation for a human-in-the-loop
    feedback mechanism and continuous improvement flywheel.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feedback_db = FeedbackDatabase()
        self.active_learning_manager = ActiveLearningManager(config)
        self.literature_monitor = ContinuousLiteratureMonitor(config)
        self.improvement_threshold = config.get('continuous_improvement', {}).get('improvement_threshold', 0.1)

    def process_user_feedback(self, feedback: UserFeedback) -> Dict[str, Any]:
        """
        Process user feedback and determine next steps.

        Args:
            feedback: User feedback to process

        Returns:
            Processing result with next steps
        """
        try:
            # Store feedback
            self.feedback_db.store_feedback(feedback)

            # Analyze feedback for immediate actions
            immediate_actions = self._analyze_feedback_for_immediate_actions(feedback)

            # Check if feedback should trigger model retraining
            retraining_needed = self._check_retraining_needed(feedback)

            # Generate response
            result = {
                'feedback_id': feedback.feedback_id,
                'status': 'processed',
                'immediate_actions': immediate_actions,
                'retraining_needed': retraining_needed,
                'next_steps': self._determine_next_steps(feedback, retraining_needed)
            }

            return result

        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
            return {
                'feedback_id': feedback.feedback_id,
                'status': 'error',
                'error': str(e)
            }

    def _analyze_feedback_for_immediate_actions(self, feedback: UserFeedback) -> List[str]:
        """Analyze feedback for immediate actions that can be taken."""
        actions = []

        # High-confidence corrections can be applied immediately
        if (feedback.confidence_score > 0.9 and
            feedback.feedback_type in [FeedbackType.ENTITY_CORRECTION, FeedbackType.RELATION_CORRECTION]):
            actions.append('apply_immediate_correction')

        # Missing information can trigger additional searches
        if feedback.feedback_type == FeedbackType.MISSING_INFORMATION:
            actions.append('trigger_additional_search')

        # Factual errors can trigger source verification
        if feedback.feedback_type == FeedbackType.FACTUAL_ERROR:
            actions.append('trigger_source_verification')

        return actions

    def _check_retraining_needed(self, feedback: UserFeedback) -> bool:
        """Check if feedback indicates need for model retraining."""
        # Check feedback volume
        pending_feedback = self.feedback_db.get_pending_feedback()

        # If we have significant feedback volume, consider retraining
        if len(pending_feedback) >= 100:
            return True

        # Check for patterns in feedback types
        feedback_types = [fb.feedback_type for fb in pending_feedback]
        type_counts = {}
        for fb_type in feedback_types:
            type_counts[fb_type] = type_counts.get(fb_type, 0) + 1

        # If a specific type of feedback is dominant, consider retraining
        for fb_type, count in type_counts.items():
            if count >= 50:  # Threshold for retraining
                return True

        return False

    def _determine_next_steps(self, feedback: UserFeedback, retraining_needed: bool) -> List[str]:
        """Determine next steps based on feedback and retraining needs."""
        steps = []

        if retraining_needed:
            steps.append('schedule_model_retraining')
            steps.append('prepare_training_data')

        if feedback.status == FeedbackStatus.PENDING:
            steps.append('schedule_feedback_review')

        steps.append('update_knowledge_graph')
        steps.append('notify_relevant_users')

        return steps

    def generate_retraining_dataset(self) -> Dict[str, Any]:
        """
        Generate dataset for model retraining from approved feedback.

        Returns:
            Dataset for retraining with metadata
        """
        # Get approved feedback
        # This would query the database for approved feedback
        approved_feedback = []  # Placeholder

        # Generate training examples
        training_examples = self.active_learning_manager.generate_training_examples(approved_feedback)

        # Prepare dataset
        dataset = {
            'examples': [asdict(example) for example in training_examples],
            'metadata': {
                'total_examples': len(training_examples),
                'feedback_sources': list(set(example.feedback_source for example in training_examples)),
                'generation_date': datetime.now().isoformat(),
                'priority_scores': [example.priority_score for example in training_examples]
            }
        }

        return dataset

    def monitor_and_update(self) -> Dict[str, Any]:
        """
        Monitor for new literature and system updates.

        Returns:
            Update status and actions taken
        """
        update_status = {
            'literature_check': False,
            'new_publications': 0,
            'feedback_processing': False,
            'retraining_needed': False
        }

        # Check for new literature
        if self.literature_monitor.should_check_for_updates():
            new_publications = self.literature_monitor.check_for_new_literature()
            update_status['literature_check'] = True
            update_status['new_publications'] = len(new_publications)

        # Check for pending feedback
        pending_feedback = self.feedback_db.get_pending_feedback()
        if pending_feedback:
            update_status['feedback_processing'] = True

            # Check if retraining is needed
            if self._check_retraining_needed(pending_feedback[0]):  # Use first feedback as sample
                update_status['retraining_needed'] = True

        return update_status

    def get_system_health_metrics(self) -> Dict[str, Any]:
        """
        Get system health metrics for monitoring.

        Returns:
            System health metrics
        """
        # Get feedback statistics
        # This would query the database for statistics
        feedback_stats = {
            'total_feedback': 0,
            'pending_feedback': 0,
            'approved_feedback': 0,
            'rejected_feedback': 0
        }

        # Get model performance metrics
        # This would query model performance database
        model_performance = {
            'current_models': [],
            'performance_trends': [],
            'improvement_areas': []
        }

        # Get literature monitoring status
        literature_status = {
            'last_check': self.literature_monitor._load_last_check().isoformat(),
            'next_check': (self.literature_monitor._load_last_check() +
                          timedelta(hours=self.literature_monitor.check_interval_hours)).isoformat(),
            'monitored_sources': len(self.literature_monitor.monitored_sources)
        }

        return {
            'feedback_statistics': feedback_stats,
            'model_performance': model_performance,
            'literature_monitoring': literature_status,
            'system_status': 'healthy'  # Would implement actual health check
        }
