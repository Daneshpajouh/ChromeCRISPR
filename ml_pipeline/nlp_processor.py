"""
NLP Processor for GeneX Project

Advanced NLP processing using transformer models and deep learning techniques
for scientific text analysis, feature extraction, and knowledge discovery.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModel,
    pipeline,
    TextClassificationPipeline,
    TokenClassificationPipeline,
    QuestionAnsweringPipeline
)
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

@dataclass
class NLPFeatures:
    """Comprehensive NLP features extracted from scientific text"""
    # Embeddings
    sentence_embeddings: np.ndarray
    document_embedding: np.ndarray
    keyword_embeddings: np.ndarray

    # Text analysis
    sentiment_score: float
    complexity_score: float
    readability_score: float

    # Topic modeling
    topics: List[str]
    topic_weights: np.ndarray

    # Named entities
    genes: List[str]
    proteins: List[str]
    organisms: List[str]
    techniques: List[str]

    # Linguistic features
    sentence_count: int
    word_count: int
    avg_sentence_length: float
    unique_words: int
    vocabulary_diversity: float

    # Scientific features
    methodology_indicators: List[str]
    result_indicators: List[str]
    conclusion_indicators: List[str]

class NLPProcessor:
    """
    Advanced NLP processor using transformer models and deep learning
    for scientific text analysis in the GeneX project.
    """

    def __init__(self,
                 model_name: str = "allenai/scibert_scivocab_uncased",
                 sentence_model: str = "all-MiniLM-L6-v2",
                 device: str = "auto"):
        """
        Initialize NLP processor with transformer models.

        Args:
            model_name: Pre-trained transformer model for scientific text
            sentence_model: Sentence transformer for embeddings
            device: Computing device (auto, cpu, cuda)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        logger.info(f"Initializing NLP processor on device: {self.device}")

        # Load transformer models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Sentence transformer for embeddings
        self.sentence_transformer = SentenceTransformer(sentence_model)

        # SpaCy for NER and linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize pipelines
        self._setup_pipelines()

        # Download NLTK data
        self._download_nltk_data()

        logger.info("NLP processor initialized successfully")

    def _setup_pipelines(self):
        """Setup various NLP pipelines for different tasks."""
        try:
            # Text classification pipeline
            self.classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                device=0 if torch.cuda.is_available() else -1
            )

            # Named entity recognition pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=0 if torch.cuda.is_available() else -1
            )

            # Question answering pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if torch.cuda.is_available() else -1
            )

        except Exception as e:
            logger.warning(f"Some pipelines failed to load: {e}")

    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")

    def extract_features(self, text: str, title: str = "") -> NLPFeatures:
        """
        Extract comprehensive NLP features from scientific text.

        Args:
            text: Main text content
            title: Paper title

        Returns:
            NLPFeatures object with all extracted features
        """
        logger.info("Extracting NLP features from text")

        # Combine title and text for analysis
        full_text = f"{title}. {text}" if title else text

        # Get embeddings
        sentence_embeddings = self._get_sentence_embeddings(text)
        document_embedding = self._get_document_embedding(full_text)
        keyword_embeddings = self._get_keyword_embeddings(text)

        # Text analysis
        sentiment_score = self._analyze_sentiment(full_text)
        complexity_score = self._analyze_complexity(text)
        readability_score = self._analyze_readability(text)

        # Topic modeling
        topics, topic_weights = self._extract_topics(text)

        # Named entities
        genes = self._extract_genes(text)
        proteins = self._extract_proteins(text)
        organisms = self._extract_organisms(text)
        techniques = self._extract_techniques(text)

        # Linguistic features
        linguistic_features = self._analyze_linguistics(text)

        # Scientific features
        methodology_indicators = self._extract_methodology_indicators(text)
        result_indicators = self._extract_result_indicators(text)
        conclusion_indicators = self._extract_conclusion_indicators(text)

        return NLPFeatures(
            sentence_embeddings=sentence_embeddings,
            document_embedding=document_embedding,
            keyword_embeddings=keyword_embeddings,
            sentiment_score=sentiment_score,
            complexity_score=complexity_score,
            readability_score=readability_score,
            topics=topics,
            topic_weights=topic_weights,
            genes=genes,
            proteins=proteins,
            organisms=organisms,
            techniques=techniques,
            **linguistic_features,
            methodology_indicators=methodology_indicators,
            result_indicators=result_indicators,
            conclusion_indicators=conclusion_indicators
        )

    def _get_sentence_embeddings(self, text: str) -> np.ndarray:
        """Extract sentence-level embeddings."""
        sentences = sent_tokenize(text)
        if not sentences:
            return np.array([])

        embeddings = self.sentence_transformer.encode(sentences)
        return embeddings

    def _get_document_embedding(self, text: str) -> np.ndarray:
        """Extract document-level embedding."""
        return self.sentence_transformer.encode([text])[0]

    def _get_keyword_embeddings(self, text: str) -> np.ndarray:
        """Extract embeddings for key terms."""
        keywords = self._extract_keywords(text)
        if not keywords:
            return np.array([])

        return self.sentence_transformer.encode(keywords)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()

            # Get top keywords
            scores = tfidf_matrix.toarray()[0]
            keyword_indices = np.argsort(scores)[-10:]  # Top 10
            return [feature_names[i] for i in keyword_indices if scores[i] > 0]
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using transformer model."""
        try:
            result = self.classifier(text[:512])  # Limit length
            # Map sentiment to score
            sentiment_map = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}
            return sentiment_map.get(result[0]['label'], 0.0)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0

    def _analyze_complexity(self, text: str) -> float:
        """Analyze text complexity using various metrics."""
        try:
            doc = self.nlp(text)

            # Average word length
            avg_word_length = np.mean([len(token.text) for token in doc if not token.is_space])

            # Average sentence length
            sentences = list(doc.sents)
            avg_sentence_length = np.mean([len([t for t in sent if not t.is_space]) for sent in sentences])

            # Type-token ratio
            words = [token.text.lower() for token in doc if token.is_alpha]
            type_token_ratio = len(set(words)) / len(words) if words else 0

            # Complexity score (0-1, higher = more complex)
            complexity = (avg_word_length * 0.3 + avg_sentence_length * 0.4 + type_token_ratio * 0.3) / 10
            return min(complexity, 1.0)

        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
            return 0.5

    def _analyze_readability(self, text: str) -> float:
        """Calculate readability score using Flesch Reading Ease."""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())

            # Count syllables (approximate)
            syllables = sum(self._count_syllables(word) for word in words)

            if len(sentences) == 0 or len(words) == 0:
                return 0.0

            # Flesch Reading Ease formula
            flesch_score = 206.835 - (1.015 * len(words) / len(sentences)) - (84.6 * syllables / len(words))
            return max(0.0, min(100.0, flesch_score)) / 100.0  # Normalize to 0-1

        except Exception as e:
            logger.warning(f"Readability analysis failed: {e}")
            return 0.5

    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel

        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        return count

    def _extract_topics(self, text: str) -> Tuple[List[str], np.ndarray]:
        """Extract topics using LDA."""
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform([text])

            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            topic_weights = lda.fit_transform(tfidf_matrix)

            feature_names = vectorizer.get_feature_names_out()
            topics = []

            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-5:]]
                topics.append(" ".join(top_words))

            return topics, topic_weights[0]

        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return [], np.array([])

    def _extract_genes(self, text: str) -> List[str]:
        """Extract gene names using NER and pattern matching."""
        try:
            # Use NER pipeline
            entities = self.ner_pipeline(text)
            genes = [ent['word'] for ent in entities if ent['entity'] in ['B-GENE', 'I-GENE']]

            # Additional pattern matching for gene names
            doc = self.nlp(text)
            gene_patterns = [
                r'\b[A-Z][A-Z0-9]*\b',  # All caps
                r'\b[A-Z][a-z]+[0-9]*\b',  # Pascal case
            ]

            import re
            for pattern in gene_patterns:
                matches = re.findall(pattern, text)
                genes.extend(matches)

            return list(set(genes))

        except Exception as e:
            logger.warning(f"Gene extraction failed: {e}")
            return []

    def _extract_proteins(self, text: str) -> List[str]:
        """Extract protein names."""
        try:
            doc = self.nlp(text)
            proteins = []

            # Look for protein-related terms
            protein_keywords = ['protein', 'enzyme', 'receptor', 'antibody', 'peptide']

            for token in doc:
                if any(keyword in token.text.lower() for keyword in protein_keywords):
                    # Extract the full noun phrase
                    chunk = token.doc[token.i:token.i+3]  # Look ahead
                    proteins.append(chunk.text)

            return list(set(proteins))

        except Exception as e:
            logger.warning(f"Protein extraction failed: {e}")
            return []

    def _extract_organisms(self, text: str) -> List[str]:
        """Extract organism names."""
        try:
            doc = self.nlp(text)
            organisms = []

            # Look for organism-related entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'GPE']:
                    organisms.append(ent.text)

            # Additional pattern matching
            organism_patterns = [
                r'\b[A-Z][a-z]+ [a-z]+\b',  # Genus species
                r'\b[A-Z][a-z]+ sp\.\b',  # Genus sp.
            ]

            import re
            for pattern in organism_patterns:
                matches = re.findall(pattern, text)
                organisms.extend(matches)

            return list(set(organisms))

        except Exception as e:
            logger.warning(f"Organism extraction failed: {e}")
            return []

    def _extract_techniques(self, text: str) -> List[str]:
        """Extract experimental techniques."""
        try:
            technique_keywords = [
                'CRISPR', 'PCR', 'Western blot', 'ELISA', 'sequencing',
                'microscopy', 'flow cytometry', 'mass spectrometry',
                'gene editing', 'base editing', 'prime editing',
                'transfection', 'transduction', 'electroporation'
            ]

            techniques = []
            text_lower = text.lower()

            for technique in technique_keywords:
                if technique.lower() in text_lower:
                    techniques.append(technique)

            return techniques

        except Exception as e:
            logger.warning(f"Technique extraction failed: {e}")
            return []

    def _analyze_linguistics(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features."""
        try:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            words = [token.text for token in doc if not token.is_space]

            return {
                'sentence_count': len(sentences),
                'word_count': len(words),
                'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
                'unique_words': len(set(words)),
                'vocabulary_diversity': len(set(words)) / len(words) if words else 0
            }

        except Exception as e:
            logger.warning(f"Linguistic analysis failed: {e}")
            return {
                'sentence_count': 0,
                'word_count': 0,
                'avg_sentence_length': 0,
                'unique_words': 0,
                'vocabulary_diversity': 0
            }

    def _extract_methodology_indicators(self, text: str) -> List[str]:
        """Extract methodology-related indicators."""
        methodology_terms = [
            'method', 'methodology', 'procedure', 'protocol',
            'experiment', 'study', 'analysis', 'technique',
            'approach', 'strategy', 'design'
        ]

        indicators = []
        text_lower = text.lower()

        for term in methodology_terms:
            if term in text_lower:
                indicators.append(term)

        return indicators

    def _extract_result_indicators(self, text: str) -> List[str]:
        """Extract result-related indicators."""
        result_terms = [
            'result', 'finding', 'outcome', 'conclusion',
            'data', 'evidence', 'observation', 'measurement',
            'statistical', 'significant', 'p-value'
        ]

        indicators = []
        text_lower = text.lower()

        for term in result_terms:
            if term in text_lower:
                indicators.append(term)

        return indicators

    def _extract_conclusion_indicators(self, text: str) -> List[str]:
        """Extract conclusion-related indicators."""
        conclusion_terms = [
            'conclusion', 'summary', 'discussion', 'implication',
            'therefore', 'thus', 'consequently', 'in conclusion',
            'overall', 'finally', 'in summary'
        ]

        indicators = []
        text_lower = text.lower()

        for term in conclusion_terms:
            if term in text_lower:
                indicators.append(term)

        return indicators
