#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder
Based on Report 2 research recommendations for comprehensive knowledge graph construction
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from pathlib import Path

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraphNode:
    """Knowledge graph node with normalized entity information"""
    entity_id: str
    entity_name: str
    entity_type: str
    normalized_id: Optional[str] = None  # Gene symbol, UniProt ID, etc.
    synonyms: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    source_papers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraphEdge:
    """Knowledge graph edge representing relationships"""
    source_id: str
    target_id: str
    relationship_type: str
    confidence_score: float = 1.0
    source_papers: List[str] = field(default_factory=list)
    evidence_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedKnowledgeGraphBuilder:
    """
    Enhanced knowledge graph builder with advanced relationship extraction
    Based on Report 2 research recommendations
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes = {}
        self.edges = []
        self.entity_normalizer = {}

        # Initialize models for relationship extraction
        self._initialize_models()

        # Load entity normalization mappings
        self._load_entity_normalizations()

    def _initialize_models(self):
        """Initialize models for relationship extraction"""
        try:
            # Primary model for relation extraction
            self.relation_model = pipeline(
                "text-classification",
                model=self.config['nlp_models']['relation_extraction']['primary'],
                device=0 if torch.cuda.is_available() else -1
            )

            # Secondary model for clinical text
            self.clinical_model = pipeline(
                "text-classification",
                model=self.config['nlp_models']['ner_models']['clinical'],
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info("Knowledge graph models initialized successfully")

        except Exception as e:
            logger.warning(f"Error initializing knowledge graph models: {e}")
            self.relation_model = None
            self.clinical_model = None

    def _load_entity_normalizations(self):
        """Load entity normalization mappings"""
        # Gene name normalization
        self.gene_normalizations = {
            'EMX1': 'ENSG00000104327',
            'BRCA1': 'ENSG00000012048',
            'PCSK9': 'ENSG00000069122',
            'TP53': 'ENSG00000141510',
            'CFTR': 'ENSG00000001626'
        }

        # Protein normalization
        self.protein_normalizations = {
            'Cas9': 'Q99ZW2',
            'Cas12a': 'A0A0H3J1X4',
            'p53': 'P04637'
        }

    def build_knowledge_graph(self, knowledge_list: List[Any]) -> Dict[str, Any]:
        """
        Build comprehensive knowledge graph from extracted knowledge
        Based on Report 2 relationship extraction recommendations
        """
        logger.info("Building enhanced knowledge graph...")

        # Process each knowledge item
        for knowledge in knowledge_list:
            self._process_knowledge_item(knowledge)

        # Extract relationships using advanced models
        self._extract_relationships(knowledge_list)

        # Normalize entities
        self._normalize_entities()

        # Quality assurance
        self._quality_assurance()

        # Build final graph structure
        graph = self._build_graph_structure()

        logger.info(f"Knowledge graph built with {len(self.nodes)} nodes and {len(self.edges)} edges")
        return graph

    def _process_knowledge_item(self, knowledge: Any):
        """Process individual knowledge item and extract entities"""
        # Extract entities from knowledge item
        entities = self._extract_entities_from_knowledge(knowledge)

        # Add entities to graph
        for entity in entities:
            self._add_entity_to_graph(entity, knowledge.paper_id)

    def _extract_entities_from_knowledge(self, knowledge: Any) -> List[Dict[str, Any]]:
        """Extract entities from knowledge item"""
        entities = []

        # Gene and protein entities
        for gene in knowledge.target_genes:
            entities.append({
                'name': gene,
                'type': 'gene',
                'confidence': knowledge.extraction_confidence
            })

        # Editing technique entities
        if knowledge.editing_technique:
            entities.append({
                'name': knowledge.editing_technique,
                'type': 'editing_technique',
                'confidence': knowledge.extraction_confidence
            })

        # Cell type entities
        for cell_type in knowledge.cell_types:
            entities.append({
                'name': cell_type,
                'type': 'cell_type',
                'confidence': knowledge.extraction_confidence
            })

        # Organism entities
        for organism in knowledge.organisms:
            entities.append({
                'name': organism,
                'type': 'organism',
                'confidence': knowledge.extraction_confidence
            })

        return entities

    def _add_entity_to_graph(self, entity: Dict[str, Any], paper_id: str):
        """Add entity to knowledge graph"""
        entity_id = f"{entity['type']}_{entity['name']}"

        if entity_id not in self.nodes:
            # Create new node
            self.nodes[entity_id] = KnowledgeGraphNode(
                entity_id=entity_id,
                entity_name=entity['name'],
                entity_type=entity['type'],
                confidence_score=entity['confidence'],
                source_papers=[paper_id]
            )
        else:
            # Update existing node
            self.nodes[entity_id].source_papers.append(paper_id)
            self.nodes[entity_id].confidence_score = max(
                self.nodes[entity_id].confidence_score,
                entity['confidence']
            )

    def _extract_relationships(self, knowledge_list: List[Any]):
        """Extract relationships using advanced models"""
        logger.info("Extracting relationships using advanced models...")

        for knowledge in knowledge_list:
            # Extract relationships from key findings
            for finding in knowledge.key_findings:
                relationships = self._extract_relationships_from_text(finding, knowledge.paper_id)
                self.edges.extend(relationships)

            # Extract relationships from results
            for result_key, result_value in knowledge.results.items():
                if isinstance(result_value, str):
                    relationships = self._extract_relationships_from_text(result_value, knowledge.paper_id)
                    self.edges.extend(relationships)

    def _extract_relationships_from_text(self, text: str, paper_id: str) -> List[KnowledgeGraphEdge]:
        """Extract relationships from text using NLP models"""
        relationships = []

        # Use rule-based extraction for common patterns
        rule_based_relations = self._extract_rule_based_relationships(text, paper_id)
        relationships.extend(rule_based_relations)

        # Use ML models for complex relationships
        if self.relation_model:
            ml_relations = self._extract_ml_relationships(text, paper_id)
            relationships.extend(ml_relations)

        return relationships

    def _extract_rule_based_relationships(self, text: str, paper_id: str) -> List[KnowledgeGraphEdge]:
        """Extract relationships using rule-based patterns"""
        relationships = []

        # Pattern: "X edits Y"
        edit_patterns = [
            r'(\w+)\s+(?:edits?|targets?|modifies?)\s+(\w+)',
            r'(\w+)\s+(?:was\s+)?(?:edited|targeted|modified)\s+by\s+(\w+)',
            r'(\w+)\s+(?:achieved|resulted\s+in)\s+(\d+%?\s+\w+)'
        ]

        for pattern in edit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entity1, entity2 = match

                # Find corresponding nodes
                node1 = self._find_entity_node(entity1)
                node2 = self._find_entity_node(entity2)

                if node1 and node2:
                    edge = KnowledgeGraphEdge(
                        source_id=node1.entity_id,
                        target_id=node2.entity_id,
                        relationship_type='edits',
                        confidence_score=0.8,
                        source_papers=[paper_id],
                        evidence_text=text
                    )
                    relationships.append(edge)

        return relationships

    def _extract_ml_relationships(self, text: str, paper_id: str) -> List[KnowledgeGraphEdge]:
        """Extract relationships using ML models"""
        relationships = []

        try:
            # Use relation extraction model
            if self.relation_model:
                # This would require more sophisticated implementation
                # For now, return empty list
                pass

        except Exception as e:
            logger.warning(f"Error in ML relationship extraction: {e}")

        return relationships

    def _find_entity_node(self, entity_name: str) -> Optional[KnowledgeGraphNode]:
        """Find entity node by name"""
        for node in self.nodes.values():
            if (node.entity_name.lower() == entity_name.lower() or
                entity_name.lower() in [syn.lower() for syn in node.synonyms]):
                return node
        return None

    def _normalize_entities(self):
        """Normalize entity names and IDs"""
        logger.info("Normalizing entities...")

        for node in self.nodes.values():
            # Gene normalization
            if node.entity_type == 'gene':
                normalized_id = self.gene_normalizations.get(node.entity_name)
                if normalized_id:
                    node.normalized_id = normalized_id

            # Protein normalization
            elif node.entity_type == 'editing_technique':
                normalized_id = self.protein_normalizations.get(node.entity_name)
                if normalized_id:
                    node.normalized_id = normalized_id

    def _quality_assurance(self):
        """Perform quality assurance on knowledge graph"""
        logger.info("Performing quality assurance...")

        # Remove low-confidence edges
        self.edges = [edge for edge in self.edges if edge.confidence_score > 0.5]

        # Remove isolated nodes
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.source_id)
            connected_nodes.add(edge.target_id)

        isolated_nodes = [node_id for node_id in self.nodes.keys()
                         if node_id not in connected_nodes]

        for node_id in isolated_nodes:
            del self.nodes[node_id]

        logger.info(f"Removed {len(isolated_nodes)} isolated nodes")

    def _build_graph_structure(self) -> Dict[str, Any]:
        """Build final graph structure"""
        return {
            'nodes': [self._node_to_dict(node) for node in self.nodes.values()],
            'edges': [self._edge_to_dict(edge) for edge in self.edges],
            'metadata': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'entity_types': list(set(node.entity_type for node in self.nodes.values())),
                'relationship_types': list(set(edge.relationship_type for edge in self.edges)),
                'created_at': datetime.now().isoformat(),
                'version': '2.0'
            }
        }

    def _node_to_dict(self, node: KnowledgeGraphNode) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            'id': node.entity_id,
            'name': node.entity_name,
            'type': node.entity_type,
            'normalized_id': node.normalized_id,
            'synonyms': node.synonyms,
            'confidence_score': node.confidence_score,
            'source_papers': node.source_papers,
            'metadata': node.metadata
        }

    def _edge_to_dict(self, edge: KnowledgeGraphEdge) -> Dict[str, Any]:
        """Convert edge to dictionary"""
        return {
            'source': edge.source_id,
            'target': edge.target_id,
            'relationship_type': edge.relationship_type,
            'confidence_score': edge.confidence_score,
            'source_papers': edge.source_papers,
            'evidence_text': edge.evidence_text,
            'metadata': edge.metadata
        }

    def save_knowledge_graph(self, graph: Dict[str, Any], output_path: str):
        """Save knowledge graph to file"""
        output_file = Path(output_path) / 'enhanced_knowledge_graph.json'

        with open(output_file, 'w') as f:
            json.dump(graph, f, indent=2)

        logger.info(f"Knowledge graph saved to {output_file}")

    def generate_graph_statistics(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive graph statistics"""
        nodes = graph['nodes']
        edges = graph['edges']

        # Entity type distribution
        entity_types = {}
        for node in nodes:
            entity_type = node['type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        # Relationship type distribution
        relationship_types = {}
        for edge in edges:
            rel_type = edge['relationship_type']
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        # Confidence score distribution
        confidence_scores = [edge['confidence_score'] for edge in edges]

        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'entity_type_distribution': entity_types,
            'relationship_type_distribution': relationship_types,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_distribution': {
                'high': len([s for s in confidence_scores if s > 0.8]),
                'medium': len([s for s in confidence_scores if 0.5 < s <= 0.8]),
                'low': len([s for s in confidence_scores if s <= 0.5])
            }
        }
