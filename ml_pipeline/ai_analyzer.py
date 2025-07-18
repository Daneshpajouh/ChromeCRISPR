"""
AI Analyzer for GeneX Project

Advanced AI techniques for scientific paper analysis, knowledge discovery,
and intelligent insights extraction using state-of-the-art AI models.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)
from sentence_transformers import SentenceTransformer
import openai
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter
import re

logger = logging.getLogger(__name__)

@dataclass
class AIInsights:
    """AI-generated insights from paper analysis"""
    # Key findings
    key_findings: List[str]
    research_gaps: List[str]
    future_directions: List[str]

    # Impact analysis
    potential_impact: str
    clinical_relevance: str
    commercial_potential: str

    # Technical insights
    methodology_insights: List[str]
    experimental_design: str
    statistical_analysis: str

    # Comparative analysis
    related_techniques: List[str]
    competing_approaches: List[str]
    advantages_disadvantages: Dict[str, List[str]]

    # Knowledge graph
    knowledge_graph: Dict[str, Any]
    entity_relationships: List[Dict[str, Any]]

class AIAnalyzer:
    """
    Advanced AI analyzer using multiple AI techniques for comprehensive
    scientific paper analysis and knowledge discovery.
    """

    def __init__(self,
                 openai_api_key: str = None,
                 model_name: str = "gpt-3.5-turbo",
                 device: str = "auto"):
        """
        Initialize AI analyzer with various AI models and techniques.

        Args:
            openai_api_key: OpenAI API key for GPT models
            model_name: GPT model to use
            device: Computing device
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model_name = model_name

        # Setup OpenAI if available
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_openai = True
        else:
            self.use_openai = False
            logger.warning("OpenAI API key not provided, using local models only")

        # Initialize local models
        self._setup_local_models()

        logger.info(f"AI analyzer initialized on device: {self.device}")

    def _setup_local_models(self):
        """Setup local AI models for analysis."""
        try:
            # Summarization model
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )

            # Question answering model
            self.qa_model = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if torch.cuda.is_available() else -1
            )

            # Text generation model
            self.generator = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )

            # Sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

        except Exception as e:
            logger.warning(f"Some local models failed to load: {e}")

    def analyze_paper(self, text: str, title: str = "") -> AIInsights:
        """
        Perform comprehensive AI analysis of a scientific paper.

        Args:
            text: Paper text content
            title: Paper title

        Returns:
            AIInsights object with comprehensive analysis
        """
        logger.info("Starting comprehensive AI analysis")

        # Combine title and text
        full_text = f"{title}. {text}" if title else text

        # Extract key findings
        key_findings = self._extract_key_findings(full_text)

        # Identify research gaps
        research_gaps = self._identify_research_gaps(full_text)

        # Predict future directions
        future_directions = self._predict_future_directions(full_text)

        # Analyze potential impact
        potential_impact = self._analyze_potential_impact(full_text)
        clinical_relevance = self._analyze_clinical_relevance(full_text)
        commercial_potential = self._analyze_commercial_potential(full_text)

        # Extract methodology insights
        methodology_insights = self._extract_methodology_insights(full_text)
        experimental_design = self._analyze_experimental_design(full_text)
        statistical_analysis = self._analyze_statistical_analysis(full_text)

        # Perform comparative analysis
        related_techniques = self._identify_related_techniques(full_text)
        competing_approaches = self._identify_competing_approaches(full_text)
        advantages_disadvantages = self._analyze_advantages_disadvantages(full_text)

        # Build knowledge graph
        knowledge_graph = self._build_knowledge_graph(full_text)
        entity_relationships = self._extract_entity_relationships(full_text)

        return AIInsights(
            key_findings=key_findings,
            research_gaps=research_gaps,
            future_directions=future_directions,
            potential_impact=potential_impact,
            clinical_relevance=clinical_relevance,
            commercial_potential=commercial_potential,
            methodology_insights=methodology_insights,
            experimental_design=experimental_design,
            statistical_analysis=statistical_analysis,
            related_techniques=related_techniques,
            competing_approaches=competing_approaches,
            advantages_disadvantages=advantages_disadvantages,
            knowledge_graph=knowledge_graph,
            entity_relationships=entity_relationships
        )

    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings using AI."""
        try:
            if self.use_openai:
                return self._extract_key_findings_openai(text)
            else:
                return self._extract_key_findings_local(text)
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return []

    def _extract_key_findings_openai(self, text: str) -> List[str]:
        """Extract key findings using OpenAI GPT."""
        prompt = f"""
        Analyze the following scientific paper and extract the 5 most important key findings:

        Paper: {text[:3000]}

        Please provide the key findings as a numbered list, focusing on:
        1. Novel discoveries or breakthroughs
        2. Significant experimental results
        3. Important methodological advances
        4. Clinical or practical implications
        5. Theoretical contributions

        Key Findings:
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a scientific research analyst specializing in gene editing and biotechnology."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )

            findings_text = response.choices[0].message.content
            findings = [f.strip() for f in findings_text.split('\n') if f.strip() and not f.startswith('Key Findings:')]
            return findings[:5]  # Return top 5 findings

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return []

    def _extract_key_findings_local(self, text: str) -> List[str]:
        """Extract key findings using local models."""
        try:
            # Use summarization to extract key points
            summary = self.summarizer(text[:1024], max_length=150, min_length=50)[0]['summary_text']

            # Extract sentences that contain key terms
            key_terms = ['discovered', 'found', 'demonstrated', 'showed', 'revealed', 'identified', 'developed']
            sentences = text.split('.')

            findings = []
            for sentence in sentences:
                if any(term in sentence.lower() for term in key_terms):
                    findings.append(sentence.strip())
                    if len(findings) >= 5:
                        break

            return findings

        except Exception as e:
            logger.error(f"Local model error: {e}")
            return []

    def _identify_research_gaps(self, text: str) -> List[str]:
        """Identify research gaps using AI."""
        try:
            if self.use_openai:
                prompt = f"""
                Analyze the following scientific paper and identify 3-5 research gaps or limitations:

                Paper: {text[:3000]}

                Focus on:
                - What questions remain unanswered
                - Limitations of the current study
                - Areas that need further investigation
                - Potential improvements or extensions

                Research Gaps:
                """

                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a research analyst identifying gaps in scientific literature."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.4
                )

                gaps_text = response.choices[0].message.content
                gaps = [g.strip() for g in gaps_text.split('\n') if g.strip() and not g.startswith('Research Gaps:')]
                return gaps[:5]
            else:
                # Use local analysis
                gap_indicators = ['limitation', 'future work', 'further study', 'remains unclear', 'not addressed']
                sentences = text.split('.')
                gaps = []

                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in gap_indicators):
                        gaps.append(sentence.strip())
                        if len(gaps) >= 5:
                            break

                return gaps

        except Exception as e:
            logger.error(f"Error identifying research gaps: {e}")
            return []

    def _predict_future_directions(self, text: str) -> List[str]:
        """Predict future research directions using AI."""
        try:
            if self.use_openai:
                prompt = f"""
                Based on the following scientific paper, predict 3-5 future research directions:

                Paper: {text[:3000]}

                Consider:
                - Potential applications and extensions
                - Next logical steps in this research area
                - Emerging opportunities and challenges
                - Clinical or commercial translation possibilities

                Future Directions:
                """

                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a research strategist predicting future directions in biotechnology."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.5
                )

                directions_text = response.choices[0].message.content
                directions = [d.strip() for d in directions_text.split('\n') if d.strip() and not d.startswith('Future Directions:')]
                return directions[:5]
            else:
                # Use local analysis
                future_indicators = ['future', 'next', 'potential', 'could', 'might', 'will']
                sentences = text.split('.')
                directions = []

                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in future_indicators):
                        directions.append(sentence.strip())
                        if len(directions) >= 5:
                            break

                return directions

        except Exception as e:
            logger.error(f"Error predicting future directions: {e}")
            return []

    def _analyze_potential_impact(self, text: str) -> str:
        """Analyze potential scientific and societal impact."""
        try:
            if self.use_openai:
                prompt = f"""
                Analyze the potential impact of this scientific research:

                Paper: {text[:3000]}

                Consider:
                - Scientific impact (advances in knowledge, methodology)
                - Societal impact (health, environment, economy)
                - Technological impact (new tools, applications)
                - Educational impact (training, awareness)

                Provide a concise analysis (2-3 sentences):
                """

                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an impact assessment expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )

                return response.choices[0].message.content
            else:
                # Local analysis
                impact_terms = ['significant', 'important', 'breakthrough', 'advance', 'improve', 'benefit']
                impact_score = sum(1 for term in impact_terms if term in text.lower())

                if impact_score > 3:
                    return "High potential impact with significant scientific and practical implications."
                elif impact_score > 1:
                    return "Moderate potential impact with notable contributions to the field."
                else:
                    return "Limited potential impact, primarily incremental contributions."

        except Exception as e:
            logger.error(f"Error analyzing potential impact: {e}")
            return "Impact analysis unavailable."

    def _analyze_clinical_relevance(self, text: str) -> str:
        """Analyze clinical relevance and medical applications."""
        try:
            clinical_terms = ['clinical', 'patient', 'therapy', 'treatment', 'disease', 'medical', 'health']
            clinical_count = sum(1 for term in clinical_terms if term in text.lower())

            if clinical_count > 5:
                return "High clinical relevance with direct medical applications."
            elif clinical_count > 2:
                return "Moderate clinical relevance with potential medical applications."
            else:
                return "Limited clinical relevance, primarily basic research."

        except Exception as e:
            logger.error(f"Error analyzing clinical relevance: {e}")
            return "Clinical relevance analysis unavailable."

    def _analyze_commercial_potential(self, text: str) -> str:
        """Analyze commercial potential and market applications."""
        try:
            commercial_terms = ['commercial', 'market', 'industry', 'product', 'application', 'business', 'economic']
            commercial_count = sum(1 for term in commercial_terms if term in text.lower())

            if commercial_count > 3:
                return "High commercial potential with market applications."
            elif commercial_count > 1:
                return "Moderate commercial potential with industry interest."
            else:
                return "Limited commercial potential, primarily academic research."

        except Exception as e:
            logger.error(f"Error analyzing commercial potential: {e}")
            return "Commercial potential analysis unavailable."

    def _extract_methodology_insights(self, text: str) -> List[str]:
        """Extract insights about methodology and experimental approach."""
        try:
            methodology_terms = ['method', 'technique', 'protocol', 'procedure', 'approach', 'strategy']
            sentences = text.split('.')
            insights = []

            for sentence in sentences:
                if any(term in sentence.lower() for term in methodology_terms):
                    insights.append(sentence.strip())
                    if len(insights) >= 3:
                        break

            return insights

        except Exception as e:
            logger.error(f"Error extracting methodology insights: {e}")
            return []

    def _analyze_experimental_design(self, text: str) -> str:
        """Analyze experimental design and methodology."""
        try:
            design_terms = ['randomized', 'controlled', 'blinded', 'cohort', 'case-control', 'longitudinal']
            design_count = sum(1 for term in design_terms if term in text.lower())

            if design_count > 2:
                return "Well-designed experimental study with robust methodology."
            elif design_count > 0:
                return "Moderate experimental design with some methodological considerations."
            else:
                return "Basic experimental design, primarily descriptive or exploratory."

        except Exception as e:
            logger.error(f"Error analyzing experimental design: {e}")
            return "Experimental design analysis unavailable."

    def _analyze_statistical_analysis(self, text: str) -> str:
        """Analyze statistical analysis and data processing."""
        try:
            stats_terms = ['statistical', 'p-value', 'significance', 'correlation', 'regression', 'analysis']
            stats_count = sum(1 for term in stats_terms if term in text.lower())

            if stats_count > 3:
                return "Comprehensive statistical analysis with rigorous data processing."
            elif stats_count > 1:
                return "Moderate statistical analysis with basic data processing."
            else:
                return "Limited statistical analysis, primarily descriptive statistics."

        except Exception as e:
            logger.error(f"Error analyzing statistical analysis: {e}")
            return "Statistical analysis unavailable."

    def _identify_related_techniques(self, text: str) -> List[str]:
        """Identify related techniques and methods."""
        try:
            techniques = [
                'CRISPR', 'TALEN', 'ZFN', 'base editing', 'prime editing',
                'gene therapy', 'genome editing', 'synthetic biology',
                'PCR', 'sequencing', 'microscopy', 'flow cytometry'
            ]

            found_techniques = []
            for technique in techniques:
                if technique.lower() in text.lower():
                    found_techniques.append(technique)

            return found_techniques

        except Exception as e:
            logger.error(f"Error identifying related techniques: {e}")
            return []

    def _identify_competing_approaches(self, text: str) -> List[str]:
        """Identify competing or alternative approaches."""
        try:
            competing_indicators = ['alternative', 'competing', 'different', 'other', 'instead', 'compared to']
            sentences = text.split('.')
            approaches = []

            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in competing_indicators):
                    approaches.append(sentence.strip())
                    if len(approaches) >= 3:
                        break

            return approaches

        except Exception as e:
            logger.error(f"Error identifying competing approaches: {e}")
            return []

    def _analyze_advantages_disadvantages(self, text: str) -> Dict[str, List[str]]:
        """Analyze advantages and disadvantages of the approach."""
        try:
            advantages = []
            disadvantages = []

            # Look for advantage/disadvantage indicators
            advantage_indicators = ['advantage', 'benefit', 'improve', 'better', 'superior', 'effective']
            disadvantage_indicators = ['disadvantage', 'limitation', 'problem', 'issue', 'challenge', 'difficulty']

            sentences = text.split('.')

            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in advantage_indicators):
                    advantages.append(sentence.strip())
                elif any(indicator in sentence.lower() for indicator in disadvantage_indicators):
                    disadvantages.append(sentence.strip())

            return {
                'advantages': advantages[:3],
                'disadvantages': disadvantages[:3]
            }

        except Exception as e:
            logger.error(f"Error analyzing advantages/disadvantages: {e}")
            return {'advantages': [], 'disadvantages': []}

    def _build_knowledge_graph(self, text: str) -> Dict[str, Any]:
        """Build a knowledge graph from the paper content."""
        try:
            # Extract entities and relationships
            entities = self._extract_entities(text)
            relationships = self._extract_relationships(text, entities)

            # Create graph structure
            graph = {
                'nodes': entities,
                'edges': relationships,
                'metadata': {
                    'total_entities': len(entities),
                    'total_relationships': len(relationships),
                    'graph_type': 'scientific_knowledge'
                }
            }

            return graph

        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return {'nodes': [], 'edges': [], 'metadata': {}}

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        try:
            # Simple entity extraction (in practice, use NER models)
            entities = []

            # Extract gene names (simple pattern)
            gene_pattern = r'\b[A-Z][A-Z0-9]*\b'
            genes = re.findall(gene_pattern, text)

            for gene in set(genes):
                entities.append({
                    'id': gene,
                    'type': 'gene',
                    'name': gene,
                    'frequency': genes.count(gene)
                })

            # Extract techniques
            techniques = ['CRISPR', 'PCR', 'sequencing', 'microscopy']
            for technique in techniques:
                if technique.lower() in text.lower():
                    entities.append({
                        'id': technique,
                        'type': 'technique',
                        'name': technique,
                        'frequency': text.lower().count(technique.lower())
                    })

            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    def _extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        try:
            relationships = []

            # Simple relationship extraction
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    # Check if entities appear in same sentence
                    sentences = text.split('.')
                    for sentence in sentences:
                        if (entity1['name'] in sentence and entity2['name'] in sentence):
                            relationships.append({
                                'source': entity1['id'],
                                'target': entity2['id'],
                                'type': 'co_occurrence',
                                'context': sentence.strip()
                            })
                            break

            return relationships

        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []

    def _extract_entity_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract detailed entity relationships."""
        try:
            relationships = []

            # Look for specific relationship patterns
            relationship_patterns = [
                r'(\w+)\s+(regulates|activates|inhibits|binds to|interacts with)\s+(\w+)',
                r'(\w+)\s+(leads to|results in|causes)\s+(\w+)',
                r'(\w+)\s+(is regulated by|is activated by|is inhibited by)\s+(\w+)'
            ]

            for pattern in relationship_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    relationships.append({
                        'source': match[0],
                        'target': match[2],
                        'relationship': match[1],
                        'type': 'functional'
                    })

            return relationships

        except Exception as e:
            logger.error(f"Error extracting entity relationships: {e}")
            return []
