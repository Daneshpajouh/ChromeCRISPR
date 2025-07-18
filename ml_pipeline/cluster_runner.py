#!/usr/bin/env python3
"""
Cluster Runner for GeneX ML/DL/AI Pipeline

This script is executed on HPC clusters to run the ML/DL/AI pipeline
for processing scientific papers and extracting knowledge.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml_pipeline.nlp_processor import NLPProcessor
from src.ml_pipeline.deep_learning_models import DeepLearningModels, ModelConfig
from src.ml_pipeline.ai_analyzer import AIAnalyzer
from src.ml_pipeline.knowledge_extractor import KnowledgeExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cluster_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the environment for ML pipeline execution."""
    logger.info("Setting up ML pipeline environment")

    # Set CUDA device
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Set memory growth
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Create output directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    logger.info("Environment setup complete")

def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input papers from JSON file."""
    logger.info(f"Loading input data from {input_file}")

    with open(input_file, 'r') as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} papers")
    return papers

def run_nlp_processing(papers: List[Dict[str, Any]],
                      nlp_processor: NLPProcessor) -> List[Dict[str, Any]]:
    """Run NLP processing on papers."""
    logger.info("Starting NLP processing")

    processed_papers = []

    for i, paper in enumerate(papers):
        try:
            logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")

            # Extract text content
            text = paper.get('abstract', '') or paper.get('content', '')
            title = paper.get('title', '')

            if not text:
                logger.warning(f"Paper {i+1} has no text content, skipping")
                continue

            # Extract NLP features
            nlp_features = nlp_processor.extract_features(text, title)

            # Add NLP features to paper
            paper['nlp_features'] = {
                'sentiment_score': nlp_features.sentiment_score,
                'complexity_score': nlp_features.complexity_score,
                'readability_score': nlp_features.readability_score,
                'topics': nlp_features.topics,
                'genes': nlp_features.genes,
                'proteins': nlp_features.proteins,
                'organisms': nlp_features.organisms,
                'techniques': nlp_features.techniques,
                'sentence_count': nlp_features.sentence_count,
                'word_count': nlp_features.word_count,
                'avg_sentence_length': nlp_features.avg_sentence_length,
                'unique_words': nlp_features.unique_words,
                'vocabulary_diversity': nlp_features.vocabulary_diversity,
                'methodology_indicators': nlp_features.methodology_indicators,
                'result_indicators': nlp_features.result_indicators,
                'conclusion_indicators': nlp_features.conclusion_indicators
            }

            # Add embeddings (save as separate files to avoid large JSON)
            embeddings_dir = f"embeddings/paper_{i}"
            os.makedirs(embeddings_dir, exist_ok=True)

            import numpy as np
            np.save(f"{embeddings_dir}/sentence_embeddings.npy", nlp_features.sentence_embeddings)
            np.save(f"{embeddings_dir}/document_embedding.npy", nlp_features.document_embedding)
            np.save(f"{embeddings_dir}/keyword_embeddings.npy", nlp_features.keyword_embeddings)

            paper['embeddings_path'] = embeddings_dir

            processed_papers.append(paper)

        except Exception as e:
            logger.error(f"Error processing paper {i+1}: {e}")
            continue

    logger.info(f"NLP processing completed for {len(processed_papers)} papers")
    return processed_papers

def run_ml_classification(papers: List[Dict[str, Any]],
                         ml_models: DeepLearningModels) -> List[Dict[str, Any]]:
    """Run ML classification on papers."""
    logger.info("Starting ML classification")

    # Prepare data for classification
    texts = []
    labels = []  # Will be generated based on content analysis

    for paper in papers:
        text = paper.get('abstract', '') or paper.get('content', '')
        if text:
            texts.append(text)

            # Generate pseudo-labels based on content analysis
            # In a real scenario, these would come from human annotation
            nlp_features = paper.get('nlp_features', {})
            techniques = nlp_features.get('techniques', [])

            # Simple rule-based classification for demonstration
            if 'CRISPR' in techniques:
                labels.append(0)  # CRISPR Gene Editing
            elif 'base editing' in [t.lower() for t in techniques]:
                labels.append(1)  # Base Editing
            elif 'prime editing' in [t.lower() for t in techniques]:
                labels.append(2)  # Prime Editing
            else:
                labels.append(0)  # Default to CRISPR

    if not texts:
        logger.warning("No texts available for classification")
        return papers

    # Train classifier
    logger.info("Training classifier")
    training_results = ml_models.train_classifier(texts, labels)
    logger.info(f"Classifier training completed: {training_results['best_val_accuracy']:.4f} accuracy")

    # Predict for all papers
    for i, paper in enumerate(papers):
        try:
            text = paper.get('abstract', '') or paper.get('content', '')
            if text:
                prediction = ml_models.predict_project(text)
                paper['ml_classification'] = prediction
        except Exception as e:
            logger.error(f"Error classifying paper {i}: {e}")

    logger.info("ML classification completed")
    return papers

def run_quality_prediction(papers: List[Dict[str, Any]],
                          ml_models: DeepLearningModels) -> List[Dict[str, Any]]:
    """Run quality prediction on papers."""
    logger.info("Starting quality prediction")

    # Prepare data for quality prediction
    texts = []
    quality_scores = []
    completeness_scores = []
    relevance_scores = []

    for paper in papers:
        text = paper.get('abstract', '') or paper.get('content', '')
        if text:
            texts.append(text)

            # Generate pseudo-quality scores based on content analysis
            nlp_features = paper.get('nlp_features', {})

            # Quality based on text length and complexity
            word_count = nlp_features.get('word_count', 0)
            quality_score = min(1.0, word_count / 1000)  # Normalize to 0-1

            # Completeness based on presence of key sections
            methodology = len(nlp_features.get('methodology_indicators', []))
            results = len(nlp_features.get('result_indicators', []))
            conclusions = len(nlp_features.get('conclusion_indicators', []))
            completeness_score = min(1.0, (methodology + results + conclusions) / 10)

            # Relevance based on gene editing terms
            techniques = nlp_features.get('techniques', [])
            relevance_score = min(1.0, len(techniques) / 5)

            quality_scores.append(quality_score)
            completeness_scores.append(completeness_score)
            relevance_scores.append(relevance_score)

    if not texts:
        logger.warning("No texts available for quality prediction")
        return papers

    # Train quality predictor
    logger.info("Training quality predictor")
    training_results = ml_models.train_quality_predictor(
        texts, quality_scores, completeness_scores, relevance_scores
    )
    logger.info(f"Quality predictor training completed: {training_results['best_val_loss']:.4f} loss")

    # Predict quality for all papers
    for i, paper in enumerate(papers):
        try:
            text = paper.get('abstract', '') or paper.get('content', '')
            if text:
                quality_prediction = ml_models.predict_quality(text)
                paper['quality_prediction'] = quality_prediction
        except Exception as e:
            logger.error(f"Error predicting quality for paper {i}: {e}")

    logger.info("Quality prediction completed")
    return papers

def run_ai_analysis(papers: List[Dict[str, Any]],
                   ai_analyzer: AIAnalyzer) -> List[Dict[str, Any]]:
    """Run AI analysis on papers."""
    logger.info("Starting AI analysis")

    for i, paper in enumerate(papers):
        try:
            logger.info(f"AI analysis for paper {i+1}/{len(papers)}")

            text = paper.get('abstract', '') or paper.get('content', '')
            if not text:
                continue

            # Run AI analysis
            ai_results = ai_analyzer.analyze_paper(text)
            paper['ai_analysis'] = ai_results

        except Exception as e:
            logger.error(f"Error in AI analysis for paper {i}: {e}")

    logger.info("AI analysis completed")
    return papers

def run_knowledge_extraction(papers: List[Dict[str, Any]],
                           knowledge_extractor: KnowledgeExtractor) -> List[Dict[str, Any]]:
    """Run knowledge extraction on papers."""
    logger.info("Starting knowledge extraction")

    for i, paper in enumerate(papers):
        try:
            logger.info(f"Knowledge extraction for paper {i+1}/{len(papers)}")

            text = paper.get('abstract', '') or paper.get('content', '')
            if not text:
                continue

            # Extract knowledge
            knowledge = knowledge_extractor.extract_knowledge(text)
            paper['extracted_knowledge'] = knowledge

        except Exception as e:
            logger.error(f"Error in knowledge extraction for paper {i}: {e}")

    logger.info("Knowledge extraction completed")
    return papers

def save_results(papers: List[Dict[str, Any]], output_dir: str, job_id: str):
    """Save processing results."""
    logger.info(f"Saving results to {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Save processed papers
    results_file = f"{output_dir}/processed_papers.json"
    with open(results_file, 'w') as f:
        json.dump(papers, f, indent=2, default=str)

    # Save summary statistics
    summary = {
        'job_id': job_id,
        'total_papers': len(papers),
        'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'statistics': {
            'papers_with_nlp_features': len([p for p in papers if 'nlp_features' in p]),
            'papers_with_classification': len([p for p in papers if 'ml_classification' in p]),
            'papers_with_quality_prediction': len([p for p in papers if 'quality_prediction' in p]),
            'papers_with_ai_analysis': len([p for p in papers if 'ai_analysis' in p]),
            'papers_with_knowledge': len([p for p in papers if 'extracted_knowledge' in p])
        }
    }

    summary_file = f"{output_dir}/summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")

def main():
    """Main function for cluster runner."""
    parser = argparse.ArgumentParser(description='GeneX ML/DL/AI Pipeline Cluster Runner')
    parser.add_argument('--job-id', required=True, help='Job ID')
    parser.add_argument('--pipeline-type', required=True,
                       choices=['comprehensive', 'classification', 'quality', 'knowledge'],
                       help='Type of pipeline to run')
    parser.add_argument('--input-file', required=True, help='Input JSON file with papers')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--scratch-dir', help='Scratch directory for temporary files')

    args = parser.parse_args()

    logger.info(f"Starting GeneX ML pipeline: {args.job_id}")
    logger.info(f"Pipeline type: {args.pipeline_type}")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output directory: {args.output_dir}")

    # Setup environment
    setup_environment()

    # Load input data
    papers = load_input_data(args.input_file)

    # Initialize components based on pipeline type
    if args.pipeline_type in ['comprehensive', 'classification', 'quality']:
        logger.info("Initializing NLP processor")
        nlp_processor = NLPProcessor()

        logger.info("Initializing ML models")
        ml_config = ModelConfig()
        ml_models = DeepLearningModels(ml_config)

    if args.pipeline_type in ['comprehensive', 'ai_analysis']:
        logger.info("Initializing AI analyzer")
        ai_analyzer = AIAnalyzer()

    if args.pipeline_type in ['comprehensive', 'knowledge']:
        logger.info("Initializing knowledge extractor")
        knowledge_extractor = KnowledgeExtractor()

    # Run pipeline components
    if args.pipeline_type in ['comprehensive', 'nlp']:
        papers = run_nlp_processing(papers, nlp_processor)

    if args.pipeline_type in ['comprehensive', 'classification']:
        papers = run_ml_classification(papers, ml_models)

    if args.pipeline_type in ['comprehensive', 'quality']:
        papers = run_quality_prediction(papers, ml_models)

    if args.pipeline_type in ['comprehensive', 'ai_analysis']:
        papers = run_ai_analysis(papers, ai_analyzer)

    if args.pipeline_type in ['comprehensive', 'knowledge']:
        papers = run_knowledge_extraction(papers, knowledge_extractor)

    # Save results
    save_results(papers, args.output_dir, args.job_id)

    logger.info(f"GeneX ML pipeline completed successfully: {args.job_id}")

if __name__ == "__main__":
    main()
