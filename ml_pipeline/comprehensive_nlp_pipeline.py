import os
import json
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# NLP and ML imports
import spacy
from spacy.tokens import Doc, Span
import scispacy
from scispacy.linking import EntityLinker
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# BioNLP specific imports
from .knowledge_extractor import AIKnowledgeExtractor
from .nlp_processor import NLPProcessor

# Knowledge Graph imports
import networkx as nx
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

# Configuration
import yaml
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class NLPConfig:
    """Configuration for NLP pipeline."""
    # Model configurations
    spacy_model: str = "en_core_sci_md"
    bert_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    # Processing parameters
    max_text_length: int = 512
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Entity extraction
    confidence_threshold: float = 0.7
    max_entities_per_doc: int = 100

    # Knowledge graph
    kg_namespace: str = "http://genex.org/ontology/"
    kg_output_format: str = "turtle"

    # Output paths
    silver_layer_path: str = "data/silver"
    gold_layer_path: str = "data/gold"
    kg_output_path: str = "data/knowledge_graph"


class BaseNLPProcessor(ABC):
    """Base class for NLP processors."""

    def __init__(self, config: NLPConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the processor."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abstractmethod
    async def process(self, data: List[Dict]) -> List[Dict]:
        """Process the input data and return processed results."""
        pass


class TextPreprocessor(BaseNLPProcessor):
    """Text preprocessing pipeline for scientific literature."""

    def __init__(self, config: NLPConfig):
        super().__init__(config)
        self.nlp = spacy.load(config.spacy_model)

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

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    async def process(self, data: List[Dict]) -> List[Dict]:
        """Preprocess text data from mined sources."""
        self.logger.info(f"Preprocessing {len(data)} documents...")

        processed_data = []

        for item in data:
            try:
                # Extract text content based on source
                text_content = self._extract_text_content(item)

                if text_content:
                    # Clean and preprocess text
                    cleaned_text = self._clean_text(text_content)

                    # Tokenize and lemmatize
                    tokens = self._tokenize_and_lemmatize(cleaned_text)

                    # Extract sentences
                    sentences = self._extract_sentences(cleaned_text)

                    # Create processed document
                    processed_doc = {
                        "original_id": item.get("id", ""),
                        "source": item.get("source", ""),
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "full_text": cleaned_text,
                        "tokens": tokens,
                        "sentences": sentences,
                        "processed_at": datetime.now().isoformat(),
                        "metadata": {
                            "text_length": len(cleaned_text),
                            "num_sentences": len(sentences),
                            "num_tokens": len(tokens)
                        }
                    }

                    processed_data.append(processed_doc)

            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                continue

        self.logger.info(f"Preprocessing completed: {len(processed_data)} documents processed")
        return processed_data

    def _extract_text_content(self, item: Dict) -> str:
        """Extract text content from different data sources."""
        text_parts = []

        # Extract from different fields based on source
        if item.get("source") == "PubMed":
            if item.get("title"):
                text_parts.append(item["title"])
            if item.get("abstract"):
                text_parts.append(item["abstract"])

        elif item.get("source") == "SemanticScholar":
            if item.get("title"):
                text_parts.append(item["title"])
            if item.get("abstract"):
                text_parts.append(item["abstract"])

        elif item.get("source") == "ENCODE":
            # Extract from ENCODE metadata
            data = item.get("data", {})
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 10:
                        text_parts.append(f"{key}: {value}")

        else:
            # Generic text extraction
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 10:
                    text_parts.append(value)

        return " ".join(text_parts)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special characters but keep scientific notation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\+\=\*\/\^]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove extra spaces
        text = text.strip()

        return text

    def _tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text."""
        # Tokenize
        tokens = word_tokenize(text.lower())

        # Remove stop words and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)

        return processed_tokens

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        return sent_tokenize(text)


class EntityExtractor(BaseNLPProcessor):
    """Entity extraction pipeline for biological and scientific entities."""

    def __init__(self, config: NLPConfig):
        super().__init__(config)
        self.nlp = spacy.load(config.spacy_model)

        # Add entity linker for biomedical entities
        if "scispacy_linker" not in self.nlp.pipe_names:
            self.nlp.add_pipe("scispacy_linker", config={
                "resolve_abbreviations": True,
                "linker_name": "umls"
            })

        # Initialize BERT-based entity extraction
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        self.model = AutoModel.from_pretrained(config.bert_model)
        self.model.to(config.device)

        # Entity types to extract
        self.entity_types = [
            "GENE", "PROTEIN", "DISEASE", "CHEMICAL", "ORGANISM",
            "CELL_LINE", "CELL_TYPE", "ANATOMICAL_STRUCTURE",
            "BIOLOGICAL_PROCESS", "MOLECULAR_FUNCTION"
        ]

    async def process(self, data: List[Dict]) -> List[Dict]:
        """Extract entities from processed documents."""
        self.logger.info(f"Extracting entities from {len(data)} documents...")

        processed_data = []

        for doc in data:
            try:
                # Extract entities using spaCy
                spacy_entities = self._extract_spacy_entities(doc["full_text"])

                # Extract entities using BERT
                bert_entities = await self._extract_bert_entities(doc["full_text"])

                # Combine and deduplicate entities
                all_entities = self._combine_entities(spacy_entities, bert_entities)

                # Add entities to document
                doc["entities"] = all_entities
                doc["entity_count"] = len(all_entities)

                processed_data.append(doc)

            except Exception as e:
                self.logger.error(f"Error extracting entities: {e}")
                doc["entities"] = []
                doc["entity_count"] = 0
                processed_data.append(doc)

        self.logger.info(f"Entity extraction completed: {len(processed_data)} documents processed")
        return processed_data

    def _extract_spacy_entities(self, text: str) -> List[Dict]:
        """Extract entities using spaCy and scispaCy."""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity_info = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.8,  # Default confidence for spaCy
                    "source": "spacy"
                }

                # Add linking information if available
                if hasattr(ent, '_.umls_ents'):
                    entity_info["umls_links"] = [
                        {
                            "cui": link[0],
                            "score": link[1]
                        } for link in ent._.umls_ents
                    ]

                entities.append(entity_info)

        return entities

    async def _extract_bert_entities(self, text: str) -> List[Dict]:
        """Extract entities using BERT model."""
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.max_text_length,
                truncation=True,
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Process outputs (simplified - in practice would use NER head)
            # This is a placeholder for actual NER processing
            entities = []

            return entities

        except Exception as e:
            self.logger.error(f"Error in BERT entity extraction: {e}")
            return []

    def _combine_entities(self, spacy_entities: List[Dict], bert_entities: List[Dict]) -> List[Dict]:
        """Combine and deduplicate entities from different sources."""
        all_entities = spacy_entities + bert_entities

        # Simple deduplication based on text and label
        seen = set()
        unique_entities = []

        for entity in all_entities:
            key = (entity["text"].lower(), entity["label"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities


class RelationshipExtractor(BaseNLPProcessor):
    """Extract relationships between entities in scientific text."""

    def __init__(self, config: NLPConfig):
        super().__init__(config)
        self.nlp = spacy.load(config.spacy_model)

        # Relationship patterns
        self.relationship_patterns = {
            "interacts_with": [
                "interacts with", "binds to", "associates with", "complexes with"
            ],
            "regulates": [
                "regulates", "activates", "inhibits", "upregulates", "downregulates"
            ],
            "expresses": [
                "expresses", "encodes", "produces", "synthesizes"
            ],
            "causes": [
                "causes", "leads to", "results in", "induces", "triggers"
            ],
            "treats": [
                "treats", "targets", "therapies", "drug for"
            ]
        }

    async def process(self, data: List[Dict]) -> List[Dict]:
        """Extract relationships between entities."""
        self.logger.info(f"Extracting relationships from {len(data)} documents...")

        processed_data = []

        for doc in data:
            try:
                # Extract relationships
                relationships = self._extract_relationships(doc)

                # Add relationships to document
                doc["relationships"] = relationships
                doc["relationship_count"] = len(relationships)

                processed_data.append(doc)

            except Exception as e:
                self.logger.error(f"Error extracting relationships: {e}")
                doc["relationships"] = []
                doc["relationship_count"] = 0
                processed_data.append(doc)

        self.logger.info(f"Relationship extraction completed: {len(processed_data)} documents processed")
        return processed_data

    def _extract_relationships(self, doc: Dict) -> List[Dict]:
        """Extract relationships between entities in a document."""
        relationships = []
        entities = doc.get("entities", [])

        if len(entities) < 2:
            return relationships

        # Process each sentence
        for sentence in doc.get("sentences", []):
            sentence_doc = self.nlp(sentence)

            # Find entities in this sentence
            sentence_entities = []
            for entity in entities:
                if entity["text"] in sentence:
                    sentence_entities.append(entity)

            # Extract relationships between entities in the same sentence
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    relationship = self._find_relationship(
                        sentence_doc, entity1, entity2
                    )
                    if relationship:
                        relationships.append(relationship)

        return relationships

    def _find_relationship(self, sentence_doc: Doc, entity1: Dict, entity2: Dict) -> Optional[Dict]:
        """Find relationship between two entities in a sentence."""
        # Check for relationship patterns
        sentence_text = sentence_doc.text.lower()

        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                if pattern in sentence_text:
                    # Check if both entities are mentioned
                    if (entity1["text"].lower() in sentence_text and
                        entity2["text"].lower() in sentence_text):

                        return {
                            "entity1": entity1,
                            "entity2": entity2,
                            "relationship_type": rel_type,
                            "pattern": pattern,
                            "sentence": sentence_doc.text,
                            "confidence": 0.7
                        }

        return None


class SemanticAnalyzer(BaseNLPProcessor):
    """Semantic analysis and embedding generation."""

    def __init__(self, config: NLPConfig):
        super().__init__(config)
        self.sentence_transformer = SentenceTransformer(config.sentence_transformer_model)
        self.sentence_transformer.to(config.device)

        # Sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if config.device == "cuda" else -1
        )

    async def process(self, data: List[Dict]) -> List[Dict]:
        """Perform semantic analysis on documents."""
        self.logger.info(f"Performing semantic analysis on {len(data)} documents...")

        processed_data = []

        for doc in data:
            try:
                # Generate embeddings
                embeddings = await self._generate_embeddings(doc)

                # Analyze sentiment
                sentiment = self._analyze_sentiment(doc["full_text"])

                # Add semantic analysis results
                doc["embeddings"] = embeddings
                doc["sentiment"] = sentiment

                processed_data.append(doc)

            except Exception as e:
                self.logger.error(f"Error in semantic analysis: {e}")
                doc["embeddings"] = {}
                doc["sentiment"] = {"label": "neutral", "score": 0.5}
                processed_data.append(doc)

        self.logger.info(f"Semantic analysis completed: {len(processed_data)} documents processed")
        return processed_data

    async def _generate_embeddings(self, doc: Dict) -> Dict:
        """Generate embeddings for different parts of the document."""
        embeddings = {}

        try:
            # Document-level embedding
            if doc["full_text"]:
                doc_embedding = self.sentence_transformer.encode(
                    doc["full_text"][:512],  # Limit length
                    convert_to_tensor=True
                )
                embeddings["document"] = doc_embedding.cpu().numpy().tolist()

            # Sentence-level embeddings
            sentence_embeddings = []
            for sentence in doc.get("sentences", [])[:10]:  # Limit number of sentences
                embedding = self.sentence_transformer.encode(
                    sentence,
                    convert_to_tensor=True
                )
                sentence_embeddings.append(embedding.cpu().numpy().tolist())

            embeddings["sentences"] = sentence_embeddings

            # Entity embeddings
            entity_embeddings = {}
            for entity in doc.get("entities", [])[:20]:  # Limit number of entities
                embedding = self.sentence_transformer.encode(
                    entity["text"],
                    convert_to_tensor=True
                )
                entity_embeddings[entity["text"]] = embedding.cpu().numpy().tolist()

            embeddings["entities"] = entity_embeddings

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")

        return embeddings

    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the text."""
        try:
            # Use first 500 characters for sentiment analysis
            text_sample = text[:500]
            result = self.sentiment_analyzer(text_sample)[0]

            return {
                "label": result["label"],
                "score": result["score"]
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {"label": "neutral", "score": 0.5}


class KnowledgeGraphBuilder(BaseNLPProcessor):
    """Build knowledge graph from processed documents."""

    def __init__(self, config: NLPConfig):
        super().__init__(config)
        self.kg = Graph()
        self.ns = Namespace(config.kg_namespace)
        self.kg.bind("genex", self.ns)

        # Initialize NetworkX graph for analysis
        self.nx_graph = nx.MultiDiGraph()

    async def process(self, data: List[Dict]) -> List[Dict]:
        """Build knowledge graph from processed documents."""
        self.logger.info(f"Building knowledge graph from {len(data)} documents...")

        # Build RDF graph
        await self._build_rdf_graph(data)

        # Build NetworkX graph
        self._build_networkx_graph(data)

        # Save graphs
        await self._save_graphs()

        # Add graph statistics to documents
        processed_data = []
        for doc in data:
            doc["kg_statistics"] = {
                "total_nodes": len(self.nx_graph.nodes()),
                "total_edges": len(self.nx_graph.edges()),
                "graph_density": nx.density(self.nx_graph)
            }
            processed_data.append(doc)

        self.logger.info(f"Knowledge graph building completed")
        return processed_data

    async def _build_rdf_graph(self, data: List[Dict]):
        """Build RDF knowledge graph."""
        for doc in data:
            # Add document node
            doc_uri = URIRef(f"{self.ns}document_{doc['original_id']}")
            self.kg.add((doc_uri, RDF.type, self.ns.Document))
            self.kg.add((doc_uri, self.ns.title, Literal(doc.get("title", ""))))
            self.kg.add((doc_uri, self.ns.source, Literal(doc.get("source", ""))))

            # Add entity nodes
            for entity in doc.get("entities", []):
                entity_uri = URIRef(f"{self.ns}entity_{entity['text'].replace(' ', '_')}")
                self.kg.add((entity_uri, RDF.type, URIRef(f"{self.ns}{entity['label']}")))
                self.kg.add((entity_uri, self.ns.name, Literal(entity["text"])))

                # Link entity to document
                self.kg.add((doc_uri, self.ns.containsEntity, entity_uri))

            # Add relationship nodes
            for rel in doc.get("relationships", []):
                rel_uri = URIRef(f"{self.ns}relationship_{len(self.kg)}")
                entity1_uri = URIRef(f"{self.ns}entity_{rel['entity1']['text'].replace(' ', '_')}")
                entity2_uri = URIRef(f"{self.ns}entity_{rel['entity2']['text'].replace(' ', '_')}")

                self.kg.add((rel_uri, RDF.type, URIRef(f"{self.ns}{rel['relationship_type']}")))
                self.kg.add((rel_uri, self.ns.subject, entity1_uri))
                self.kg.add((rel_uri, self.ns.object, entity2_uri))
                self.kg.add((rel_uri, self.ns.confidence, Literal(rel["confidence"])))

    def _build_networkx_graph(self, data: List[Dict]):
        """Build NetworkX graph for analysis."""
        for doc in data:
            # Add document node
            doc_id = f"doc_{doc['original_id']}"
            self.nx_graph.add_node(doc_id, type="document", title=doc.get("title", ""))

            # Add entity nodes and edges
            for entity in doc.get("entities", []):
                entity_id = f"entity_{entity['text']}"
                self.nx_graph.add_node(
                    entity_id,
                    type="entity",
                    label=entity["label"],
                    text=entity["text"]
                )

                # Add edge from document to entity
                self.nx_graph.add_edge(
                    doc_id,
                    entity_id,
                    relationship="contains"
                )

            # Add relationship edges
            for rel in doc.get("relationships", []):
                entity1_id = f"entity_{rel['entity1']['text']}"
                entity2_id = f"entity_{rel['entity2']['text']}"

                self.nx_graph.add_edge(
                    entity1_id,
                    entity2_id,
                    relationship=rel["relationship_type"],
                    confidence=rel["confidence"]
                )

    async def _save_graphs(self):
        """Save knowledge graphs to files."""
        os.makedirs(self.config.kg_output_path, exist_ok=True)

        # Save RDF graph
        rdf_path = os.path.join(self.config.kg_output_path, "knowledge_graph.ttl")
        self.kg.serialize(destination=rdf_path, format=self.config.kg_output_format)

        # Save NetworkX graph
        nx_path = os.path.join(self.config.kg_output_path, "knowledge_graph.pkl")
        nx.write_gpickle(self.nx_graph, nx_path)

        # Save graph statistics
        stats = {
            "rdf_triples": len(self.kg),
            "nx_nodes": len(self.nx_graph.nodes()),
            "nx_edges": len(self.nx_graph.edges()),
            "graph_density": nx.density(self.nx_graph),
            "connected_components": nx.number_connected_components(self.nx_graph.to_undirected())
        }

        stats_path = os.path.join(self.config.kg_output_path, "graph_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


class ComprehensiveNLPipeline:
    """Comprehensive NLP pipeline orchestrating all processing stages."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.nlp_config = NLPConfig()
        self.logger = self._setup_logging()

        # Initialize processors
        self.preprocessor = TextPreprocessor(self.nlp_config)
        self.entity_extractor = EntityExtractor(self.nlp_config)
        self.relationship_extractor = RelationshipExtractor(self.nlp_config)
        self.semantic_analyzer = SemanticAnalyzer(self.nlp_config)
        self.kg_builder = KnowledgeGraphBuilder(self.nlp_config)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("ComprehensiveNLPipeline")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def process_bronze_to_silver(self, bronze_data: List[Dict]) -> List[Dict]:
        """Process Bronze layer data to Silver layer."""
        self.logger.info("Processing Bronze to Silver layer...")

        # Stage 1: Text preprocessing
        self.logger.info("Stage 1: Text preprocessing")
        preprocessed_data = await self.preprocessor.process(bronze_data)

        # Stage 2: Entity extraction
        self.logger.info("Stage 2: Entity extraction")
        entity_data = await self.entity_extractor.process(preprocessed_data)

        # Stage 3: Relationship extraction
        self.logger.info("Stage 3: Relationship extraction")
        relationship_data = await self.relationship_extractor.process(entity_data)

        # Save to Silver layer
        await self._save_to_silver_layer(relationship_data)

        return relationship_data

    async def process_silver_to_gold(self, silver_data: List[Dict]) -> List[Dict]:
        """Process Silver layer data to Gold layer."""
        self.logger.info("Processing Silver to Gold layer...")

        # Stage 4: Semantic analysis
        self.logger.info("Stage 4: Semantic analysis")
        semantic_data = await self.semantic_analyzer.process(silver_data)

        # Stage 5: Knowledge graph construction
        self.logger.info("Stage 5: Knowledge graph construction")
        final_data = await self.kg_builder.process(semantic_data)

        # Save to Gold layer
        await self._save_to_gold_layer(final_data)

        return final_data

    async def _save_to_silver_layer(self, data: List[Dict]):
        """Save processed data to Silver layer."""
        os.makedirs(self.nlp_config.silver_layer_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.nlp_config.silver_layer_path}/silver_processed_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Saved {len(data)} documents to Silver layer: {filename}")

    async def _save_to_gold_layer(self, data: List[Dict]):
        """Save processed data to Gold layer."""
        os.makedirs(self.nlp_config.gold_layer_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.nlp_config.gold_layer_path}/gold_processed_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Saved {len(data)} documents to Gold layer: {filename}")

    async def run_complete_pipeline(self, bronze_data: List[Dict]) -> Dict:
        """Run the complete NLP pipeline from Bronze to Gold layer."""
        try:
            self.logger.info("Starting comprehensive NLP pipeline...")

            # Process Bronze to Silver
            silver_data = await self.process_bronze_to_silver(bronze_data)

            # Process Silver to Gold
            gold_data = await self.process_silver_to_gold(silver_data)

            # Generate pipeline report
            report = self._generate_pipeline_report(bronze_data, silver_data, gold_data)

            self.logger.info("Comprehensive NLP pipeline completed successfully")
            return report

        except Exception as e:
            self.logger.error(f"Error in NLP pipeline: {e}")
            raise

    def _generate_pipeline_report(self, bronze_data: List[Dict], silver_data: List[Dict], gold_data: List[Dict]) -> Dict:
        """Generate comprehensive pipeline report."""
        report = {
            "pipeline_execution": {
                "start_time": datetime.now().isoformat(),
                "bronze_documents": len(bronze_data),
                "silver_documents": len(silver_data),
                "gold_documents": len(gold_data)
            },
            "processing_statistics": {
                "total_entities": sum(doc.get("entity_count", 0) for doc in gold_data),
                "total_relationships": sum(doc.get("relationship_count", 0) for doc in gold_data),
                "avg_entities_per_doc": np.mean([doc.get("entity_count", 0) for doc in gold_data]),
                "avg_relationships_per_doc": np.mean([doc.get("relationship_count", 0) for doc in gold_data])
            },
            "knowledge_graph": {
                "total_nodes": gold_data[0].get("kg_statistics", {}).get("total_nodes", 0) if gold_data else 0,
                "total_edges": gold_data[0].get("kg_statistics", {}).get("total_edges", 0) if gold_data else 0,
                "graph_density": gold_data[0].get("kg_statistics", {}).get("graph_density", 0) if gold_data else 0
            }
        }

        return report


# Main execution function
async def main():
    """Main function to run the comprehensive NLP pipeline."""
    # Load bronze data
    bronze_files = glob.glob("data/bronze/*.json")
    bronze_data = []

    for file in bronze_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                bronze_data.extend(data)
            else:
                bronze_data.append(data)

    # Run NLP pipeline
    pipeline = ComprehensiveNLPipeline()
    report = await pipeline.run_complete_pipeline(bronze_data)

    # Save report
    with open("logs/nlp_pipeline_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print("NLP pipeline completed successfully!")


if __name__ == "__main__":
    import glob
    asyncio.run(main())
