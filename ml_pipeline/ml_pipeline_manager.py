"""
ML Pipeline Manager for GeneX Project

Orchestrates all ML/DL/AI components for comprehensive scientific paper
analysis and knowledge extraction in the GeneX project.
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .nlp_processor import NLPProcessor, NLPFeatures
from .deep_learning_models import DeepLearningModels, ModelConfig
from .ai_analyzer import AIAnalyzer, AIInsights
from .knowledge_extractor import KnowledgeExtractor, ExtractedKnowledge
from .cluster_executor import ClusterExecutor, ClusterConfig

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Result from ML pipeline execution"""
    # Input data
    input_papers: List[Dict[str, Any]]

    # Processing results
    nlp_results: List[NLPFeatures]
    ml_classifications: List[Dict[str, Any]]
    quality_predictions: List[Dict[str, Any]]
    ai_insights: List[AIInsights]
    knowledge_extractions: List[ExtractedKnowledge]

    # Summary statistics
    total_papers: int
    processing_time: float
    success_rate: float

    # Metadata
    pipeline_version: str
    execution_timestamp: str
    cluster_info: Dict[str, Any]

class MLPipelineManager:
    """
    Comprehensive ML pipeline manager that orchestrates all AI/ML/DL
    components for the GeneX project.
    """

    def __init__(self,
                 config: Dict[str, Any] = None,
                 use_cluster: bool = True,
                 cluster_name: str = "graham"):
        """
        Initialize ML pipeline manager.

        Args:
            config: Configuration dictionary
            use_cluster: Whether to use cluster execution
            cluster_name: Name of cluster to use
        """
        self.config = config or {}
        self.use_cluster = use_cluster
        self.cluster_name = cluster_name

        # Initialize components
        self.nlp_processor = None
        self.ml_models = None
        self.ai_analyzer = None
        self.knowledge_extractor = None
        self.cluster_executor = None

        # Pipeline state
        self.pipeline_version = "1.0.0"
        self.is_initialized = False

        logger.info("ML Pipeline Manager initialized")

    def initialize_pipeline(self,
                          nlp_model: str = "allenai/scibert_scivocab_uncased",
                          ml_config: ModelConfig = None,
                          openai_api_key: str = None):
        """
        Initialize all pipeline components.

        Args:
            nlp_model: NLP model to use
            ml_config: ML model configuration
            openai_api_key: OpenAI API key for AI analysis
        """
        logger.info("Initializing ML pipeline components")

        try:
            # Initialize NLP processor
            logger.info("Initializing NLP processor")
            self.nlp_processor = NLPProcessor(
                model_name=nlp_model,
                device="auto"
            )

            # Initialize ML models
            logger.info("Initializing ML models")
            ml_config = ml_config or ModelConfig()
            self.ml_models = DeepLearningModels(ml_config)

            # Initialize AI analyzer
            logger.info("Initializing AI analyzer")
            self.ai_analyzer = AIAnalyzer(
                openai_api_key=openai_api_key,
                device="auto"
            )

            # Initialize knowledge extractor
            logger.info("Initializing knowledge extractor")
            self.knowledge_extractor = KnowledgeExtractor(
                model_name=nlp_model,
                device="auto"
            )

            # Initialize cluster executor if needed
            if self.use_cluster:
                logger.info("Initializing cluster executor")
                cluster_config = ClusterConfig(
                    cluster_name=self.cluster_name,
                    username=os.getenv('USER', 'amird'),
                    project_dir=f"/home/{os.getenv('USER', 'amird')}/PhD",
                    scratch_dir=f"/scratch/{os.getenv('USER', 'amird')}/genex"
                )
                self.cluster_executor = ClusterExecutor(cluster_config)

            self.is_initialized = True
            logger.info("ML pipeline components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise

    def run_comprehensive_pipeline(self,
                                 papers: List[Dict[str, Any]],
                                 pipeline_type: str = "comprehensive") -> PipelineResult:
        """
        Run comprehensive ML/DL/AI pipeline on papers.

        Args:
            papers: List of paper data
            pipeline_type: Type of pipeline to run

        Returns:
            PipelineResult with all analysis results
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

        logger.info(f"Running {pipeline_type} pipeline on {len(papers)} papers")
        start_time = time.time()

        try:
            if self.use_cluster and self.cluster_executor:
                return self._run_pipeline_on_cluster(papers, pipeline_type)
            else:
                return self._run_pipeline_locally(papers, pipeline_type)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    def _run_pipeline_on_cluster(self, papers: List[Dict[str, Any]],
                                pipeline_type: str) -> PipelineResult:
        """Run pipeline on HPC cluster."""
        logger.info("Executing pipeline on cluster")

        # Submit job to cluster
        job_id = self.cluster_executor.submit_ml_pipeline_job(
            input_papers=papers,
            pipeline_type=pipeline_type,
            output_dir=f"ml_results/{pipeline_type}_{int(time.time())}"
        )

        # Wait for completion
        logger.info(f"Waiting for cluster job {job_id} to complete...")
        results = self.cluster_executor.wait_for_completion([job_id], timeout=7200)

        if job_id not in results:
            raise RuntimeError(f"Cluster job {job_id} failed")

        # Process results
        cluster_result = results[job_id]

        # Convert cluster results to PipelineResult format
        pipeline_result = PipelineResult(
            input_papers=papers,
            nlp_results=cluster_result.get('nlp_results', []),
            ml_classifications=cluster_result.get('ml_classifications', []),
            quality_predictions=cluster_result.get('quality_predictions', []),
            ai_insights=cluster_result.get('ai_insights', []),
            knowledge_extractions=cluster_result.get('knowledge_extractions', []),
            total_papers=len(papers),
            processing_time=cluster_result.get('processing_time', 0),
            success_rate=cluster_result.get('success_rate', 0),
            pipeline_version=self.pipeline_version,
            execution_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            cluster_info={'cluster_name': self.cluster_name, 'job_id': job_id}
        )

        return pipeline_result

    def _run_pipeline_locally(self, papers: List[Dict[str, Any]],
                             pipeline_type: str) -> PipelineResult:
        """Run pipeline locally."""
        logger.info("Executing pipeline locally")

        nlp_results = []
        ml_classifications = []
        quality_predictions = []
        ai_insights = []
        knowledge_extractions = []

        successful_papers = 0

        for i, paper in enumerate(papers):
            try:
                logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")

                # Extract text content
                text = paper.get('abstract', '') or paper.get('content', '')
                title = paper.get('title', '')

                if not text:
                    logger.warning(f"Paper {i+1} has no text content, skipping")
                    continue

                # NLP processing
                if pipeline_type in ['comprehensive', 'nlp']:
                    nlp_features = self.nlp_processor.extract_features(text, title)
                    nlp_results.append(nlp_features)

                # ML classification
                if pipeline_type in ['comprehensive', 'classification']:
                    classification = self.ml_models.predict_project(text)
                    ml_classifications.append(classification)

                # Quality prediction
                if pipeline_type in ['comprehensive', 'quality']:
                    quality_pred = self.ml_models.predict_quality(text)
                    quality_predictions.append(quality_pred)

                # AI analysis
                if pipeline_type in ['comprehensive', 'ai_analysis']:
                    ai_insight = self.ai_analyzer.analyze_paper(text, title)
                    ai_insights.append(ai_insight)

                # Knowledge extraction
                if pipeline_type in ['comprehensive', 'knowledge']:
                    knowledge = self.knowledge_extractor.extract_knowledge(text, title)
                    knowledge_extractions.append(knowledge)

                successful_papers += 1

            except Exception as e:
                logger.error(f"Error processing paper {i+1}: {e}")
                continue

        processing_time = time.time() - start_time
        success_rate = successful_papers / len(papers) if papers else 0

        return PipelineResult(
            input_papers=papers,
            nlp_results=nlp_results,
            ml_classifications=ml_classifications,
            quality_predictions=quality_predictions,
            ai_insights=ai_insights,
            knowledge_extractions=knowledge_extractions,
            total_papers=len(papers),
            processing_time=processing_time,
            success_rate=success_rate,
            pipeline_version=self.pipeline_version,
            execution_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            cluster_info={'execution_mode': 'local'}
        )

    def run_batch_processing(self,
                           papers_batch: List[List[Dict[str, Any]]],
                           pipeline_type: str = "comprehensive") -> List[PipelineResult]:
        """
        Run pipeline on multiple batches of papers.

        Args:
            papers_batch: List of paper batches
            pipeline_type: Type of pipeline to run

        Returns:
            List of PipelineResult objects
        """
        logger.info(f"Running batch processing on {len(papers_batch)} batches")

        results = []

        for i, batch in enumerate(papers_batch):
            logger.info(f"Processing batch {i+1}/{len(papers_batch)} with {len(batch)} papers")

            try:
                batch_result = self.run_comprehensive_pipeline(batch, pipeline_type)
                results.append(batch_result)

            except Exception as e:
                logger.error(f"Batch {i+1} failed: {e}")
                # Create empty result for failed batch
                empty_result = PipelineResult(
                    input_papers=batch,
                    nlp_results=[],
                    ml_classifications=[],
                    quality_predictions=[],
                    ai_insights=[],
                    knowledge_extractions=[],
                    total_papers=len(batch),
                    processing_time=0,
                    success_rate=0,
                    pipeline_version=self.pipeline_version,
                    execution_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    cluster_info={'batch_id': i, 'status': 'failed'}
                )
                results.append(empty_result)

        return results

    def save_results(self, results: PipelineResult, output_dir: str):
        """
        Save pipeline results to disk.

        Args:
            results: Pipeline results to save
            output_dir: Output directory
        """
        logger.info(f"Saving results to {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Save comprehensive results
        results_dict = {
            'pipeline_version': results.pipeline_version,
            'execution_timestamp': results.execution_timestamp,
            'total_papers': results.total_papers,
            'processing_time': results.processing_time,
            'success_rate': results.success_rate,
            'cluster_info': results.cluster_info,
            'summary_statistics': self._generate_summary_statistics(results)
        }

        with open(f"{output_dir}/pipeline_summary.json", 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        # Save detailed results
        if results.nlp_results:
            nlp_data = [self._nlp_features_to_dict(features) for features in results.nlp_results]
            with open(f"{output_dir}/nlp_results.json", 'w') as f:
                json.dump(nlp_data, f, indent=2, default=str)

        if results.ml_classifications:
            with open(f"{output_dir}/ml_classifications.json", 'w') as f:
                json.dump(results.ml_classifications, f, indent=2, default=str)

        if results.quality_predictions:
            with open(f"{output_dir}/quality_predictions.json", 'w') as f:
                json.dump(results.quality_predictions, f, indent=2, default=str)

        if results.ai_insights:
            ai_data = [self._ai_insights_to_dict(insights) for insights in results.ai_insights]
            with open(f"{output_dir}/ai_insights.json", 'w') as f:
                json.dump(ai_data, f, indent=2, default=str)

        if results.knowledge_extractions:
            knowledge_data = [self._knowledge_extraction_to_dict(knowledge) for knowledge in results.knowledge_extractions]
            with open(f"{output_dir}/knowledge_extractions.json", 'w') as f:
                json.dump(knowledge_data, f, indent=2, default=str)

        # Save as CSV for analysis
        self._save_results_csv(results, output_dir)

        logger.info(f"Results saved to {output_dir}")

    def _nlp_features_to_dict(self, features: NLPFeatures) -> Dict[str, Any]:
        """Convert NLPFeatures to dictionary."""
        return {
            'sentiment_score': features.sentiment_score,
            'complexity_score': features.complexity_score,
            'readability_score': features.readability_score,
            'topics': features.topics,
            'genes': features.genes,
            'proteins': features.proteins,
            'organisms': features.organisms,
            'techniques': features.techniques,
            'sentence_count': features.sentence_count,
            'word_count': features.word_count,
            'avg_sentence_length': features.avg_sentence_length,
            'unique_words': features.unique_words,
            'vocabulary_diversity': features.vocabulary_diversity
        }

    def _ai_insights_to_dict(self, insights: AIInsights) -> Dict[str, Any]:
        """Convert AIInsights to dictionary."""
        return {
            'key_findings': insights.key_findings,
            'research_gaps': insights.research_gaps,
            'future_directions': insights.future_directions,
            'potential_impact': insights.potential_impact,
            'clinical_relevance': insights.clinical_relevance,
            'commercial_potential': insights.commercial_potential,
            'methodology_insights': insights.methodology_insights,
            'experimental_design': insights.experimental_design,
            'statistical_analysis': insights.statistical_analysis,
            'related_techniques': insights.related_techniques,
            'competing_approaches': insights.competing_approaches
        }

    def _knowledge_extraction_to_dict(self, knowledge: ExtractedKnowledge) -> Dict[str, Any]:
        """Convert ExtractedKnowledge to dictionary."""
        return {
            'genes': knowledge.genes,
            'proteins': knowledge.proteins,
            'organisms': knowledge.organisms,
            'techniques': knowledge.techniques,
            'diseases': knowledge.diseases,
            'chemicals': knowledge.chemicals,
            'gene_protein_relationships': knowledge.gene_protein_relationships,
            'gene_disease_relationships': knowledge.gene_disease_relationships,
            'technique_gene_relationships': knowledge.technique_gene_relationships,
            'experimental_results': knowledge.experimental_results,
            'statistical_data': knowledge.statistical_data,
            'methodology_details': knowledge.methodology_details,
            'extraction_confidence': knowledge.extraction_confidence,
            'knowledge_completeness': knowledge.knowledge_completeness
        }

    def _generate_summary_statistics(self, results: PipelineResult) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        stats = {
            'total_papers': results.total_papers,
            'successful_papers': int(results.total_papers * results.success_rate),
            'processing_time_minutes': results.processing_time / 60,
            'papers_per_minute': results.total_papers / (results.processing_time / 60) if results.processing_time > 0 else 0
        }

        # NLP statistics
        if results.nlp_results:
            stats['nlp'] = {
                'avg_sentiment': np.mean([r.sentiment_score for r in results.nlp_results]),
                'avg_complexity': np.mean([r.complexity_score for r in results.nlp_results]),
                'avg_readability': np.mean([r.readability_score for r in results.nlp_results]),
                'total_genes_found': sum(len(r.genes) for r in results.nlp_results),
                'total_proteins_found': sum(len(r.proteins) for r in results.nlp_results),
                'total_techniques_found': sum(len(r.techniques) for r in results.nlp_results)
            }

        # ML classification statistics
        if results.ml_classifications:
            project_counts = {}
            for classification in results.ml_classifications:
                project = classification.get('predicted_project', 'Unknown')
                project_counts[project] = project_counts.get(project, 0) + 1
            stats['ml_classification'] = {
                'project_distribution': project_counts,
                'avg_confidence': np.mean([c.get('confidence', 0) for c in results.ml_classifications])
            }

        # Quality prediction statistics
        if results.quality_predictions:
            stats['quality_prediction'] = {
                'avg_quality_score': np.mean([q.get('quality_score', 0) for q in results.quality_predictions]),
                'avg_completeness_score': np.mean([q.get('completeness_score', 0) for q in results.quality_predictions]),
                'avg_relevance_score': np.mean([q.get('relevance_score', 0) for q in results.quality_predictions])
            }

        # Knowledge extraction statistics
        if results.knowledge_extractions:
            stats['knowledge_extraction'] = {
                'avg_extraction_confidence': np.mean([k.extraction_confidence for k in results.knowledge_extractions]),
                'avg_knowledge_completeness': np.mean([k.knowledge_completeness for k in results.knowledge_extractions]),
                'total_entities_extracted': sum(len(k.genes) + len(k.proteins) + len(k.organisms) + len(k.techniques) + len(k.diseases) + len(k.chemicals) for k in results.knowledge_extractions),
                'total_relationships_extracted': sum(len(k.gene_protein_relationships) + len(k.gene_disease_relationships) + len(k.technique_gene_relationships) for k in results.knowledge_extractions)
            }

        return stats

    def _save_results_csv(self, results: PipelineResult, output_dir: str):
        """Save results as CSV files for analysis."""
        try:
            # Create summary DataFrame
            summary_data = []

            for i, paper in enumerate(results.input_papers):
                row = {
                    'paper_id': i,
                    'title': paper.get('title', ''),
                    'source': paper.get('source', ''),
                    'year': paper.get('year', ''),
                    'successful_processing': i < len(results.nlp_results) if results.nlp_results else False
                }

                # Add NLP features
                if i < len(results.nlp_results):
                    nlp = results.nlp_results[i]
                    row.update({
                        'sentiment_score': nlp.sentiment_score,
                        'complexity_score': nlp.complexity_score,
                        'readability_score': nlp.readability_score,
                        'sentence_count': nlp.sentence_count,
                        'word_count': nlp.word_count,
                        'genes_count': len(nlp.genes),
                        'proteins_count': len(nlp.proteins),
                        'techniques_count': len(nlp.techniques)
                    })

                # Add ML classification
                if i < len(results.ml_classifications):
                    ml = results.ml_classifications[i]
                    row.update({
                        'predicted_project': ml.get('predicted_project', ''),
                        'classification_confidence': ml.get('confidence', 0)
                    })

                # Add quality prediction
                if i < len(results.quality_predictions):
                    quality = results.quality_predictions[i]
                    row.update({
                        'quality_score': quality.get('quality_score', 0),
                        'completeness_score': quality.get('completeness_score', 0),
                        'relevance_score': quality.get('relevance_score', 0)
                    })

                summary_data.append(row)

            # Save as CSV
            df = pd.DataFrame(summary_data)
            df.to_csv(f"{output_dir}/pipeline_summary.csv", index=False)

        except Exception as e:
            logger.error(f"Error saving CSV results: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'is_initialized': self.is_initialized,
            'pipeline_version': self.pipeline_version,
            'use_cluster': self.use_cluster,
            'cluster_name': self.cluster_name,
            'components_loaded': {
                'nlp_processor': self.nlp_processor is not None,
                'ml_models': self.ml_models is not None,
                'ai_analyzer': self.ai_analyzer is not None,
                'knowledge_extractor': self.knowledge_extractor is not None,
                'cluster_executor': self.cluster_executor is not None
            }
        }
