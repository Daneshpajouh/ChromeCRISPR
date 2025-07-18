#!/usr/bin/env python3
"""
AI-Powered Knowledge Extraction System
Uses ML/DL/NLP to extract knowledge from 75 years of gene editing literature
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

# ML/DL/NLP Libraries
try:
    from transformers import (
        AutoTokenizer, AutoModel, pipeline,
        T5Tokenizer, T5ForConditionalGeneration,
        BertTokenizer, BertModel
    )
    from sentence_transformers import SentenceTransformer, util
    import torch
    import torch.nn.functional as F
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    import spacy
    from spacy.matcher import Matcher
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import gensim
    from gensim.models import Word2Vec, Doc2Vec
    from gensim.models.doc2vec import TaggedDocument

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

except ImportError as e:
    print(f"Warning: Some ML libraries not available: {e}")
    print("Using simplified extraction methods")

@dataclass
class ExtractedKnowledge:
    """Structured knowledge extracted from scientific literature"""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    journal: str
    doi: Optional[str] = None

    # AI-extracted knowledge
    key_findings: List[str] = field(default_factory=list)
    methodology: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    conclusions: List[str] = field(default_factory=list)

    # Gene editing specific
    editing_technique: Optional[str] = None
    target_genes: List[str] = field(default_factory=list)
    cell_types: List[str] = field(default_factory=list)
    organisms: List[str] = field(default_factory=list)
    efficiency_data: Dict[str, float] = field(default_factory=dict)
    safety_data: Dict[str, float] = field(default_factory=dict)

    # Knowledge graph entities
    entities: List[str] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)  # (entity1, relation, entity2)

    # Confidence scores
    extraction_confidence: float = 0.0
    relevance_score: float = 0.0

    # Metadata
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "v1.0"

class AIKnowledgeExtractor:
    """AI-powered knowledge extraction from scientific literature"""

    def __init__(self, model_config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)

        # Model configuration
        self.config = model_config or {
            'sentence_transformer': 'all-MiniLM-L6-v2',
            'bert_model': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
            't5_model': 't5-base',
            'spacy_model': 'en_core_web_sm',
            'max_length': 512,
            'batch_size': 8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # Initialize AI models
        self._initialize_models()

        # Knowledge extraction pipelines
        self._setup_extraction_pipelines()

        # Domain-specific knowledge
        self._load_domain_knowledge()

    def _initialize_models(self):
        """Initialize all AI models"""
        self.logger.info("Initializing AI models...")

        try:
            # Sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer(self.config['sentence_transformer'])

            # BERT for biomedical text understanding
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.config['bert_model'])
            self.bert_model = AutoModel.from_pretrained(self.config['bert_model'])

            # T5 for text generation and summarization
            self.t5_tokenizer = T5Tokenizer.from_pretrained(self.config['t5_model'])
            self.t5_model = T5ForConditionalGeneration.from_pretrained(self.config['t5_model'])

            # SpaCy for NLP tasks
            try:
                self.nlp = spacy.load(self.config['spacy_model'])
            except OSError:
                self.logger.warning(f"SpaCy model {self.config['spacy_model']} not found. Installing...")
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', self.config['spacy_model']])
                self.nlp = spacy.load(self.config['spacy_model'])

            # Move models to device
            self.bert_model.to(self.config['device'])
            self.t5_model.to(self.config['device'])

            self.logger.info("AI models initialized successfully")

        except Exception as e:
            self.logger.warning(f"Error initializing AI models: {e}")
            self.logger.info("Falling back to simplified extraction methods")
            self._initialize_simplified_models()

    def _initialize_simplified_models(self):
        """Initialize simplified models for basic extraction"""
        self.sentence_model = None
        self.bert_model = None
        self.t5_model = None
        self.nlp = None
        self.gene_ner_pipeline = None
        self.qa_pipeline = None
        self.classifier = None
        self.summarizer = None

    def _setup_extraction_pipelines(self):
        """Setup specialized extraction pipelines"""
        try:
            # Named Entity Recognition for gene editing domain
            self.gene_ner_pipeline = pipeline(
                "ner",
                model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                aggregation_strategy="simple"
            )

            # Question-Answering for specific information extraction
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2"
            )

            # Text classification for paper categorization
            self.classifier = pipeline(
                "text-classification",
                model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            )

            # Summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
        except Exception as e:
            self.logger.warning(f"Error setting up extraction pipelines: {e}")
            self.gene_ner_pipeline = None
            self.qa_pipeline = None
            self.classifier = None
            self.summarizer = None

    def _load_domain_knowledge(self):
        """Load domain-specific knowledge for gene editing"""
        # Gene editing techniques and terminology
        self.gene_editing_terms = {
            'techniques': [
                'CRISPR', 'Cas9', 'Prime Editing', 'Base Editing', 'TALEN', 'ZFN',
                'Homology Directed Repair', 'Non-Homologous End Joining', 'NHEJ',
                'HDR', 'PAM', 'gRNA', 'sgRNA', 'pegRNA', 'nickase', 'dead Cas9'
            ],
            'targets': [
                'gene knockout', 'gene knockin', 'gene activation', 'gene repression',
                'point mutation', 'insertion', 'deletion', 'substitution'
            ],
            'metrics': [
                'editing efficiency', 'off-target effects', 'specificity',
                'cytotoxicity', 'cell viability', 'mutation rate'
            ]
        }

        # Biological entities
        self.biological_entities = {
            'cell_types': ['HEK293T', 'HeLa', 'K562', 'Jurkat', 'iPSC', 'primary cells'],
            'organisms': ['human', 'mouse', 'rat', 'zebrafish', 'drosophila', 'yeast'],
            'diseases': ['cancer', 'sickle cell anemia', 'cystic fibrosis', 'Duchenne muscular dystrophy']
        }

    def extract_knowledge_from_paper(self, paper_text: str, metadata: Dict[str, Any]) -> ExtractedKnowledge:
        """Extract comprehensive knowledge from a scientific paper"""
        self.logger.info(f"Extracting knowledge from paper: {metadata.get('title', 'Unknown')}")

        # Initialize knowledge object
        knowledge = ExtractedKnowledge(
            paper_id=metadata.get('id', 'unknown'),
            title=metadata.get('title', ''),
            authors=metadata.get('authors', []),
            year=metadata.get('year', 2024),
            journal=metadata.get('journal', ''),
            doi=metadata.get('doi')
        )

        # Extract key findings using AI
        knowledge.key_findings = self._extract_key_findings(paper_text)

        # Extract methodology
        knowledge.methodology = self._extract_methodology(paper_text)

        # Extract results and data
        knowledge.results = self._extract_results(paper_text)

        # Extract conclusions
        knowledge.conclusions = self._extract_conclusions(paper_text)

        # Extract gene editing specific information
        knowledge.editing_technique = self._identify_editing_technique(paper_text)
        knowledge.target_genes = self._extract_target_genes(paper_text)
        knowledge.cell_types = self._extract_cell_types(paper_text)
        knowledge.organisms = self._extract_organisms(paper_text)

        # Extract quantitative data
        knowledge.efficiency_data = self._extract_efficiency_data(paper_text)
        knowledge.safety_data = self._extract_safety_data(paper_text)

        # Extract entities and relationships
        knowledge.entities = self._extract_entities(paper_text)
        knowledge.relationships = self._extract_relationships(paper_text)

        # Calculate confidence scores
        knowledge.extraction_confidence = self._calculate_extraction_confidence(paper_text, knowledge)
        knowledge.relevance_score = self._calculate_relevance_score(paper_text, knowledge)

        return knowledge

    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings using AI"""
        if self.sentence_model is None:
            return self._extract_key_findings_simple(text)

        # Use T5 for summarization and key point extraction
        sentences = sent_tokenize(text)

        # Find sentences with high information content
        key_sentences = []
        for sentence in sentences[:50]:  # Limit to first 50 sentences for efficiency
            if len(sentence.split()) > 10:  # Only meaningful sentences
                # Use BERT embeddings to find important sentences
                embedding = self.sentence_model.encode(sentence)
                key_sentences.append((sentence, embedding))

        # Cluster similar sentences and select representatives
        if key_sentences:
            embeddings = [emb for _, emb in key_sentences]
            sentences = [sent for sent, _ in key_sentences]

            # Use DBSCAN to cluster similar sentences
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)

            # Select representative sentences from each cluster
            unique_labels = set(clustering.labels_)
            findings = []

            for label in unique_labels:
                if label != -1:  # Not noise
                    cluster_sentences = [sent for i, sent in enumerate(sentences) if clustering.labels_[i] == label]
                    # Select the most representative sentence (highest similarity to cluster center)
                    cluster_embeddings = [emb for i, emb in enumerate(embeddings) if clustering.labels_[i] == label]
                    center = np.mean(cluster_embeddings, axis=0)
                    similarities = [util.pytorch_cos_sim(torch.tensor(emb), torch.tensor(center)).item()
                                  for emb in cluster_embeddings]
                    best_idx = np.argmax(similarities)
                    findings.append(cluster_sentences[best_idx])

            return findings[:5]  # Return top 5 findings

        return []

    def _extract_key_findings_simple(self, text: str) -> List[str]:
        """Simple key findings extraction without AI models"""
        sentences = sent_tokenize(text)
        findings = []

        # Look for sentences with key phrases
        key_phrases = ['found', 'discovered', 'showed', 'demonstrated', 'revealed', 'identified']

        for sentence in sentences[:20]:
            sentence_lower = sentence.lower()
            if any(phrase in sentence_lower for phrase in key_phrases):
                if len(sentence.split()) > 8:  # Meaningful length
                    findings.append(sentence)

        return findings[:3]

    def _extract_methodology(self, text: str) -> List[str]:
        """Extract methodology using AI"""
        if self.qa_pipeline is None:
            return self._extract_methodology_simple(text)

        # Use question-answering to extract methodology
        methodology_questions = [
            "What methods were used?",
            "How was the experiment conducted?",
            "What experimental procedures were followed?",
            "What techniques were employed?",
            "What was the experimental design?"
        ]

        methodology = []
        for question in methodology_questions:
            try:
                answer = self.qa_pipeline(question=question, context=text[:1000])
                if answer['score'] > 0.5:  # Confidence threshold
                    methodology.append(answer['answer'])
            except Exception as e:
                self.logger.warning(f"Error extracting methodology: {e}")

        return list(set(methodology))  # Remove duplicates

    def _extract_methodology_simple(self, text: str) -> List[str]:
        """Simple methodology extraction without AI models"""
        methodology = []

        # Look for methodology sections
        method_keywords = ['method', 'procedure', 'protocol', 'technique', 'experiment']

        sentences = sent_tokenize(text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in method_keywords):
                if len(sentence.split()) > 5:
                    methodology.append(sentence)

        return methodology[:3]

    def _extract_results(self, text: str) -> Dict[str, Any]:
        """Extract quantitative results using AI"""
        results = {}

        # Extract numerical data using regex and AI
        # Look for patterns like "efficiency was X%" or "editing rate of Y"
        numerical_patterns = [
            r'(\d+(?:\.\d+)?)\s*%?\s*(?:efficiency|rate|frequency)',
            r'(?:editing|mutation|knockout)\s*(?:efficiency|rate)\s*(?:of|was)\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:fold|times)\s*(?:increase|decrease)',
            r'(?:IC50|EC50|LD50)\s*(?:of|was)\s*(\d+(?:\.\d+)?)'
        ]

        for pattern in numerical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results[f'pattern_{len(results)}'] = [float(m) for m in matches]

        # Use NER to extract specific measurements
        if self.gene_ner_pipeline:
            try:
                entities = self.gene_ner_pipeline(text[:1000])
                for entity in entities:
                    if entity['score'] > 0.7:  # High confidence
                        results[f"entity_{entity['entity_group']}"] = entity['word']
            except Exception as e:
                self.logger.warning(f"Error in NER extraction: {e}")

        return results

    def _extract_conclusions(self, text: str) -> List[str]:
        """Extract conclusions using AI"""
        # Look for conclusion sections
        conclusion_keywords = ['conclusion', 'summary', 'therefore', 'thus', 'in conclusion']

        sentences = sent_tokenize(text)
        conclusions = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in conclusion_keywords):
                # Use BERT to verify if this is actually a conclusion
                if self.sentence_model:
                    embedding = self.sentence_model.encode(sentence)
                conclusions.append(sentence)

        # If no explicit conclusions found, use summarization
        if not conclusions and len(text) > 100 and self.summarizer:
            try:
                summary = self.summarizer(text[:1000], max_length=150, min_length=50)
                conclusions.append(summary[0]['summary_text'])
            except Exception as e:
                self.logger.warning(f"Error in summarization: {e}")

        return conclusions[:3]  # Return top 3 conclusions

    def _identify_editing_technique(self, text: str) -> Optional[str]:
        """Identify the gene editing technique used"""
        techniques = self.gene_editing_terms['techniques']

        # Use sentence similarity to find the most relevant technique
        if self.sentence_model:
            text_embedding = self.sentence_model.encode(text[:1000])

            best_technique = None
            best_score = 0.0

            for technique in techniques:
                technique_embedding = self.sentence_model.encode(technique)
                similarity = util.pytorch_cos_sim(
                    torch.tensor(text_embedding),
                    torch.tensor(technique_embedding)
                ).item()

                if similarity > best_score and similarity > 0.3:  # Threshold
                    best_score = similarity
                    best_technique = technique

            return best_technique
        else:
            # Simple keyword matching
            text_lower = text.lower()
            for technique in techniques:
                if technique.lower() in text_lower:
                    return technique
            return None

    def _extract_target_genes(self, text: str) -> List[str]:
        """Extract target genes using AI"""
        # Use NER to identify gene names
        if self.gene_ner_pipeline:
            try:
                entities = self.gene_ner_pipeline(text[:2000])
                genes = []

                for entity in entities:
                    if entity['entity_group'] in ['GENE', 'PROTEIN'] and entity['score'] > 0.7:
                        gene_name = entity['word']
                        # Clean up gene name
                        gene_name = re.sub(r'[^\w\-]', '', gene_name)
                        if len(gene_name) > 2:  # Filter out very short names
                            genes.append(gene_name)

                return list(set(genes))  # Remove duplicates
            except Exception as e:
                self.logger.warning(f"Error extracting genes: {e}")

        # Fallback to simple extraction
        return self._extract_genes_simple(text)

    def _extract_genes_simple(self, text: str) -> List[str]:
        """Simple gene extraction without AI models"""
        genes = []

        # Look for common gene patterns
        gene_patterns = [
            r'\b[A-Z]{3,5}\d*\b',  # Short gene codes like BRCA1, TP53
            r'\b[A-Z][a-z]+\d*\b',  # Gene names like Cas9, Cas12
        ]

        for pattern in gene_patterns:
            matches = re.findall(pattern, text)
            genes.extend(matches)

        return list(set(genes))

    def _extract_cell_types(self, text: str) -> List[str]:
        """Extract cell types mentioned"""
        cell_types = []

        for cell_type in self.biological_entities['cell_types']:
            if cell_type.lower() in text.lower():
                cell_types.append(cell_type)

        # Use NER for additional cell types
        if self.gene_ner_pipeline:
            try:
                entities = self.gene_ner_pipeline(text[:1000])
                for entity in entities:
                    if 'cell' in entity['word'].lower() and entity['score'] > 0.6:
                        cell_types.append(entity['word'])
            except Exception as e:
                self.logger.warning(f"Error extracting cell types: {e}")

        return list(set(cell_types))

    def _extract_organisms(self, text: str) -> List[str]:
        """Extract organisms mentioned"""
        organisms = []

        for organism in self.biological_entities['organisms']:
            if organism.lower() in text.lower():
                organisms.append(organism)

        return list(set(organisms))

    def _extract_efficiency_data(self, text: str) -> Dict[str, float]:
        """Extract efficiency-related data"""
        efficiency_data = {}

        # Look for efficiency measurements
        efficiency_patterns = [
            (r'(\d+(?:\.\d+)?)\s*%?\s*editing\s*efficiency', 'editing_efficiency'),
            (r'(\d+(?:\.\d+)?)\s*%?\s*knockout\s*efficiency', 'knockout_efficiency'),
            (r'(\d+(?:\.\d+)?)\s*%?\s*knockin\s*efficiency', 'knockin_efficiency'),
            (r'(\d+(?:\.\d+)?)\s*%?\s*mutation\s*rate', 'mutation_rate'),
            (r'(\d+(?:\.\d+)?)\s*%?\s*success\s*rate', 'success_rate')
        ]

        for pattern, key in efficiency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the highest value if multiple matches
                values = [float(m) for m in matches]
                efficiency_data[key] = max(values)

        return efficiency_data

    def _extract_safety_data(self, text: str) -> Dict[str, float]:
        """Extract safety-related data"""
        safety_data = {}

        # Look for safety measurements
        safety_patterns = [
            (r'(\d+(?:\.\d+)?)\s*%?\s*cytotoxicity', 'cytotoxicity'),
            (r'(\d+(?:\.\d+)?)\s*%?\s*cell\s*viability', 'cell_viability'),
            (r'(\d+(?:\.\d+)?)\s*%?\s*apoptosis', 'apoptosis_rate'),
            (r'(\d+(?:\.\d+)?)\s*off\s*target', 'off_target_rate')
        ]

        for pattern, key in safety_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                values = [float(m) for m in matches]
                safety_data[key] = max(values)

        return safety_data

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities using AI"""
        entities = []

        try:
            # Use SpaCy for NER
            if self.nlp:
                doc = self.nlp(text[:2000])
                for ent in doc.ents:
                    if ent.label_ in ['GENE', 'PROTEIN', 'DISEASE', 'CHEMICAL', 'ORGANISM']:
                        entities.append(ent.text)

            # Use BERT NER for additional entities
            if self.gene_ner_pipeline:
                bert_entities = self.gene_ner_pipeline(text[:1000])
                for entity in bert_entities:
                    if entity['score'] > 0.6:
                        entities.append(entity['word'])
        except Exception as e:
            self.logger.warning(f"Error extracting entities: {e}")

        return list(set(entities))

    def _extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities using AI"""
        relationships = []

        # Use dependency parsing to find relationships
        if self.nlp:
            try:
                doc = self.nlp(text[:1000])

                for token in doc:
                    if token.dep_ in ['nsubj', 'dobj'] and token.head.pos_ == 'VERB':
                        subject = token.text
                        verb = token.head.text
                        # Find object
                        for child in token.head.children:
                            if child.dep_ == 'dobj':
                                obj = child.text
                                relationships.append((subject, verb, obj))
            except Exception as e:
                self.logger.warning(f"Error extracting relationships: {e}")

        return relationships

    def _calculate_extraction_confidence(self, text: str, knowledge: ExtractedKnowledge) -> float:
        """Calculate confidence in the extraction"""
        # Base confidence on various factors
        confidence_factors = []

        # Text length factor
        if len(text) > 1000:
            confidence_factors.append(0.8)
        elif len(text) > 500:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)

        # Entity extraction factor
        if knowledge.entities:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)

        # Results extraction factor
        if knowledge.results:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)

        # Key findings factor
        if knowledge.key_findings:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.2)

        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _calculate_relevance_score(self, text: str, knowledge: ExtractedKnowledge) -> float:
        """Calculate relevance to gene editing domain"""
        # Calculate similarity to gene editing domain
        gene_editing_text = " ".join(self.gene_editing_terms['techniques'] +
                                   self.gene_editing_terms['targets'] +
                                   self.gene_editing_terms['metrics'])

        if self.sentence_model:
            text_embedding = self.sentence_model.encode(text[:1000])
            domain_embedding = self.sentence_model.encode(gene_editing_text)

            similarity = util.pytorch_cos_sim(
                torch.tensor(text_embedding),
                torch.tensor(domain_embedding)
            ).item()

            return max(0.0, similarity)  # Ensure non-negative
        else:
            # Simple keyword-based relevance
            text_lower = text.lower()
            domain_keywords = self.gene_editing_terms['techniques'] + self.gene_editing_terms['targets']
            matches = sum(1 for keyword in domain_keywords if keyword.lower() in text_lower)
            return min(1.0, matches / len(domain_keywords))

    def build_knowledge_graph(self, knowledge_list: List[ExtractedKnowledge]) -> Dict[str, Any]:
        """Build a knowledge graph from extracted knowledge"""
        self.logger.info("Building knowledge graph...")

        # Collect all entities and relationships
        all_entities = set()
        all_relationships = []

        for knowledge in knowledge_list:
            all_entities.update(knowledge.entities)
            all_relationships.extend(knowledge.relationships)

        # Create entity embeddings
        entity_embeddings = {}
        if self.sentence_model:
            for entity in all_entities:
                try:
                    embedding = self.sentence_model.encode(entity)
                    entity_embeddings[entity] = embedding
                except Exception as e:
                    self.logger.warning(f"Error encoding entity {entity}: {e}")

        # Create knowledge graph structure
        knowledge_graph = {
            'entities': list(all_entities),
            'relationships': all_relationships,
            'entity_embeddings': entity_embeddings,
            'papers': [k.paper_id for k in knowledge_list],
            'total_papers': len(knowledge_list),
            'extraction_date': datetime.now().isoformat()
        }

        return knowledge_graph

    def save_knowledge_base(self, knowledge_list: List[ExtractedKnowledge],
                          output_path: str):
        """Save the knowledge base to file"""
        self.logger.info(f"Saving knowledge base to {output_path}")

        # Convert to serializable format
        knowledge_data = []
        for knowledge in knowledge_list:
            knowledge_dict = {
                'paper_id': knowledge.paper_id,
                'title': knowledge.title,
                'authors': knowledge.authors,
                'year': knowledge.year,
                'journal': knowledge.journal,
                'doi': knowledge.doi,
                'key_findings': knowledge.key_findings,
                'methodology': knowledge.methodology,
                'results': knowledge.results,
                'conclusions': knowledge.conclusions,
                'editing_technique': knowledge.editing_technique,
                'target_genes': knowledge.target_genes,
                'cell_types': knowledge.cell_types,
                'organisms': knowledge.organisms,
                'efficiency_data': knowledge.efficiency_data,
                'safety_data': knowledge.safety_data,
                'entities': knowledge.entities,
                'relationships': knowledge.relationships,
                'extraction_confidence': knowledge.extraction_confidence,
                'relevance_score': knowledge.relevance_score,
                'extraction_timestamp': knowledge.extraction_timestamp.isoformat(),
                'model_version': knowledge.model_version
            }
            knowledge_data.append(knowledge_dict)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(knowledge_data, f, indent=2)

        self.logger.info(f"Knowledge base saved with {len(knowledge_data)} papers")
