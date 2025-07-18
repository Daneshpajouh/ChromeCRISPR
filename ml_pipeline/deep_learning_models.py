"""
Deep Learning Models for GeneX Project

Advanced deep learning models for scientific paper classification,
prediction, and analysis using state-of-the-art architectures.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForQuestionAnswering
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for deep learning models"""
    model_name: str = "allenai/scibert_scivocab_uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    num_classes: int = 11  # 11 GeneX projects
    dropout_rate: float = 0.3
    device: str = "auto"

class ScientificPaperDataset(Dataset):
    """Custom dataset for scientific papers"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class GeneXClassifier(nn.Module):
    """
    Deep learning classifier for GeneX project classification
    using transformer-based architecture.
    """

    def __init__(self, config: ModelConfig):
        super(GeneXClassifier, self).__init__()
        self.config = config

        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(config.model_name)

        # Classification head
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, config.num_classes)

        # Additional layers for enhanced performance
        self.hidden_layer = nn.Linear(self.transformer.config.hidden_size, 512)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(512)

    def forward(self, input_ids, attention_mask, labels=None):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        pooled_output = outputs.pooler_output

        # Enhanced classification head
        hidden = self.hidden_layer(pooled_output)
        hidden = self.relu(hidden)
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)

        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))

        return loss, logits

class QualityPredictor(nn.Module):
    """
    Deep learning model for predicting paper quality scores
    using multi-task learning approach.
    """

    def __init__(self, config: ModelConfig):
        super(QualityPredictor, self).__init__()
        self.config = config

        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(config.model_name)

        # Multi-task heads
        self.quality_head = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 1)  # Quality score (0-1)
        )

        self.completeness_head = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 1)  # Completeness score (0-1)
        )

        self.relevance_head = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 1)  # Relevance score (0-1)
        )

    def forward(self, input_ids, attention_mask, quality_labels=None,
                completeness_labels=None, relevance_labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output

        quality_score = self.quality_head(pooled_output)
        completeness_score = self.completeness_head(pooled_output)
        relevance_score = self.relevance_head(pooled_output)

        loss = 0
        if quality_labels is not None:
            quality_loss = F.mse_loss(quality_score.squeeze(), quality_labels)
            loss += quality_loss

        if completeness_labels is not None:
            completeness_loss = F.mse_loss(completeness_score.squeeze(), completeness_labels)
            loss += completeness_loss

        if relevance_labels is not None:
            relevance_loss = F.mse_loss(relevance_score.squeeze(), relevance_labels)
            loss += relevance_loss

        return loss, quality_score, completeness_score, relevance_score

class KnowledgeExtractor(nn.Module):
    """
    Deep learning model for extracting knowledge from scientific papers
    using sequence labeling and question answering.
    """

    def __init__(self, config: ModelConfig):
        super(KnowledgeExtractor, self).__init__()
        self.config = config

        # Load pre-trained models
        self.ner_model = BertForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=len(self._get_entity_labels())
        )

        self.qa_model = BertForQuestionAnswering.from_pretrained(
            config.model_name
        )

        # Entity classification head
        self.entity_classifier = nn.Sequential(
            nn.Linear(self.ner_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, len(self._get_entity_labels()))
        )

    def _get_entity_labels(self):
        """Get entity labels for NER"""
        return [
            'O', 'B-GENE', 'I-GENE', 'B-PROTEIN', 'I-PROTEIN',
            'B-ORGANISM', 'I-ORGANISM', 'B-TECHNIQUE', 'I-TECHNIQUE',
            'B-DISEASE', 'I-DISEASE', 'B-CHEMICAL', 'I-CHEMICAL'
        ]

    def forward(self, input_ids, attention_mask, labels=None):
        # NER prediction
        ner_outputs = self.ner_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return ner_outputs

class DeepLearningModels:
    """
    Manager class for deep learning models in the GeneX project.
    Handles training, evaluation, and inference for various models.
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")

        # Initialize models
        self.classifier = None
        self.quality_predictor = None
        self.knowledge_extractor = None

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        logger.info(f"Deep learning models initialized on device: {self.device}")

    def train_classifier(self, texts: List[str], labels: List[int],
                        validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the GeneX classifier model.

        Args:
            texts: List of paper texts
            labels: List of project labels (0-10 for 11 projects)
            validation_split: Fraction of data for validation

        Returns:
            Training results and metrics
        """
        logger.info("Training GeneX classifier")

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = ScientificPaperDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        val_dataset = ScientificPaperDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Initialize model
        self.classifier = GeneXClassifier(self.config).to(self.device)
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.config.learning_rate)

        # Training loop
        best_val_acc = 0
        training_history = []

        for epoch in range(self.config.num_epochs):
            # Training
            self.classifier.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                loss, logits = self.classifier(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation
            self.classifier.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    loss, logits = self.classifier(input_ids, attention_mask, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_acc': train_acc,
                'val_loss': val_loss / len(val_loader),
                'val_acc': val_acc
            })

            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                       f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model(self.classifier, 'classifier')

        return {
            'training_history': training_history,
            'best_val_accuracy': best_val_acc,
            'final_train_accuracy': train_acc
        }

    def predict_project(self, text: str) -> Dict[str, Any]:
        """
        Predict which GeneX project a paper belongs to.

        Args:
            text: Paper text

        Returns:
            Prediction results with confidence scores
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier first.")

        self.classifier.eval()

        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Make prediction
        with torch.no_grad():
            _, logits = self.classifier(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Project mapping
        projects = [
            "CRISPR Gene Editing", "Base Editing", "Prime Editing",
            "Gene Therapy", "Genome Engineering", "Synthetic Biology",
            "Bioinformatics", "Computational Biology", "Systems Biology",
            "Precision Medicine", "Personalized Genomics"
        ]

        return {
            'predicted_project': projects[predicted_class],
            'project_id': predicted_class,
            'confidence': confidence,
            'all_probabilities': probabilities[0].cpu().numpy().tolist()
        }

    def train_quality_predictor(self, texts: List[str], quality_scores: List[float],
                               completeness_scores: List[float], relevance_scores: List[float],
                               validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the quality predictor model.

        Args:
            texts: List of paper texts
            quality_scores: List of quality scores (0-1)
            completeness_scores: List of completeness scores (0-1)
            relevance_scores: List of relevance scores (0-1)
            validation_split: Fraction of data for validation

        Returns:
            Training results
        """
        logger.info("Training quality predictor")

        # Split data
        train_texts, val_texts, train_quality, val_quality, train_complete, val_complete, train_relevant, val_relevant = train_test_split(
            texts, quality_scores, completeness_scores, relevance_scores,
            test_size=validation_split, random_state=42
        )

        # Create datasets (using dummy labels for dataset compatibility)
        train_labels = [0] * len(train_texts)
        val_labels = [0] * len(val_texts)

        train_dataset = ScientificPaperDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        val_dataset = ScientificPaperDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Initialize model
        self.quality_predictor = QualityPredictor(self.config).to(self.device)
        optimizer = torch.optim.AdamW(self.quality_predictor.parameters(), lr=self.config.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        training_history = []

        for epoch in range(self.config.num_epochs):
            # Training
            self.quality_predictor.train()
            train_loss = 0

            for i, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Get corresponding scores
                start_idx = i * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(train_quality))
                batch_quality = torch.tensor(train_quality[start_idx:end_idx], dtype=torch.float).to(self.device)
                batch_complete = torch.tensor(train_complete[start_idx:end_idx], dtype=torch.float).to(self.device)
                batch_relevant = torch.tensor(train_relevant[start_idx:end_idx], dtype=torch.float).to(self.device)

                optimizer.zero_grad()
                loss, _, _, _ = self.quality_predictor(
                    input_ids, attention_mask, batch_quality, batch_complete, batch_relevant
                )
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.quality_predictor.eval()
            val_loss = 0

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    start_idx = i * self.config.batch_size
                    end_idx = min(start_idx + self.config.batch_size, len(val_quality))
                    batch_quality = torch.tensor(val_quality[start_idx:end_idx], dtype=torch.float).to(self.device)
                    batch_complete = torch.tensor(val_complete[start_idx:end_idx], dtype=torch.float).to(self.device)
                    batch_relevant = torch.tensor(val_relevant[start_idx:end_idx], dtype=torch.float).to(self.device)

                    loss, _, _, _ = self.quality_predictor(
                        input_ids, attention_mask, batch_quality, batch_complete, batch_relevant
                    )
                    val_loss += loss.item()

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader)
            })

            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Val Loss: {val_loss/len(val_loader):.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(self.quality_predictor, 'quality_predictor')

        return {
            'training_history': training_history,
            'best_val_loss': best_val_loss
        }

    def predict_quality(self, text: str) -> Dict[str, float]:
        """
        Predict quality metrics for a paper.

        Args:
            text: Paper text

        Returns:
            Quality predictions
        """
        if self.quality_predictor is None:
            raise ValueError("Quality predictor not trained. Call train_quality_predictor first.")

        self.quality_predictor.eval()

        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Make prediction
        with torch.no_grad():
            _, quality_score, completeness_score, relevance_score = self.quality_predictor(
                input_ids, attention_mask
            )

        return {
            'quality_score': quality_score.item(),
            'completeness_score': completeness_score.item(),
            'relevance_score': relevance_score.item()
        }

    def _save_model(self, model: nn.Module, model_name: str):
        """Save model to disk"""
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/{model_name}.pth')
        logger.info(f"Model saved: models/{model_name}.pth")

    def load_model(self, model_name: str):
        """Load model from disk"""
        model_path = f'models/{model_name}.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_name == 'classifier':
            self.classifier = GeneXClassifier(self.config).to(self.device)
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
        elif model_name == 'quality_predictor':
            self.quality_predictor = QualityPredictor(self.config).to(self.device)
            self.quality_predictor.load_state_dict(torch.load(model_path, map_location=self.device))

        logger.info(f"Model loaded: {model_path}")
