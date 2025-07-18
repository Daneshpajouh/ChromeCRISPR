import os
import json
import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Deep Learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

# Graph Neural Networks
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool
from torch_geometric.loader import DataLoader as GraphDataLoader

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Configuration
import yaml
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ModelConfig:
    """Configuration for advanced ML models."""
    # Model parameters
    model_type: str = "transformer"  # transformer, gnn, hybrid
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    dropout: float = 0.1

    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Data parameters
    max_length: int = 512
    num_classes: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output paths
    model_save_path: str = "models"
    results_path: str = "results"


class GeneEditingDataset(Dataset):
    """Custom dataset for gene editing research data."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract text content
        text = item.get("full_text", "")
        if not text:
            text = f"{item.get('title', '')} {item.get('abstract', '')}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Extract labels (simplified - in practice would have proper labels)
        labels = torch.tensor([0])  # Placeholder

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels,
            "metadata": {
                "id": item.get("original_id", ""),
                "source": item.get("source", ""),
                "entities": item.get("entities", []),
                "relationships": item.get("relationships", [])
            }
        }


class GraphDataset(Dataset):
    """Dataset for graph-based learning on knowledge graphs."""

    def __init__(self, data: List[Dict]):
        self.data = data
        self.graphs = self._build_graphs()

    def _build_graphs(self) -> List[Data]:
        """Build PyTorch Geometric graphs from data."""
        graphs = []

        for item in self.data:
            # Extract entities as nodes
            entities = item.get("entities", [])
            relationships = item.get("relationships", [])

            if not entities:
                continue

            # Create node features
            node_features = []
            node_mapping = {}

            for i, entity in enumerate(entities):
                # Simple feature vector (in practice would be embeddings)
                feature = torch.tensor([i, len(entity["text"]), hash(entity["label"]) % 1000])
                node_features.append(feature)
                node_mapping[entity["text"]] = i

            node_features = torch.stack(node_features)

            # Create edge indices
            edge_indices = []
            edge_attrs = []

            for rel in relationships:
                entity1 = rel["entity1"]["text"]
                entity2 = rel["entity2"]["text"]

                if entity1 in node_mapping and entity2 in node_mapping:
                    idx1 = node_mapping[entity1]
                    idx2 = node_mapping[entity2]

                    edge_indices.append([idx1, idx2])
                    edge_indices.append([idx2, idx1])  # Bidirectional

                    # Edge attributes (relationship type)
                    rel_type = hash(rel["relationship_type"]) % 100
                    edge_attrs.extend([rel_type, rel_type])

            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0,), dtype=torch.long)

            # Create graph
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([0])  # Placeholder label
            )

            graphs.append(graph)

        return graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class TransformerModel(nn.Module):
    """Advanced transformer model for gene editing research."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Load pre-trained model
        self.transformer = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.transformer.config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )

        # Entity-aware attention
        self.entity_attention = nn.MultiheadAttention(
            config.hidden_size, config.num_heads, dropout=config.dropout
        )

        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask, entity_positions=None):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state

        # Apply entity-aware attention if entity positions provided
        if entity_positions is not None:
            entity_embeddings = self._extract_entity_embeddings(
                sequence_output, entity_positions
            )
            attended_output, _ = self.entity_attention(
                sequence_output, entity_embeddings, entity_embeddings
            )
            sequence_output = attended_output

        # Global average pooling
        pooled_output = torch.mean(sequence_output, dim=1)

        # Classification
        logits = self.classifier(pooled_output)

        return logits

    def _extract_entity_embeddings(self, sequence_output, entity_positions):
        """Extract embeddings for entity positions."""
        batch_size, seq_len, hidden_size = sequence_output.shape
        entity_embeddings = []

        for i in range(batch_size):
            positions = entity_positions[i]
            if len(positions) > 0:
                entity_emb = sequence_output[i, positions].mean(dim=0)
            else:
                entity_emb = sequence_output[i].mean(dim=0)
            entity_embeddings.append(entity_emb)

        return torch.stack(entity_embeddings).unsqueeze(1)


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for knowledge graph analysis."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Node feature dimension
        input_dim = 3  # From GraphDataset

        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, config.hidden_size)
        self.conv2 = GCNConv(config.hidden_size, config.hidden_size)
        self.conv3 = GCNConv(config.hidden_size, config.hidden_size)

        # Graph attention layers
        self.gat1 = GATConv(config.hidden_size, config.hidden_size // config.num_heads, heads=config.num_heads)
        self.gat2 = GATConv(config.hidden_size, config.hidden_size // config.num_heads, heads=config.num_heads)

        # Global pooling and classification
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Graph attention
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        # Global pooling
        x = self.pool(x, batch)

        # Classification
        logits = self.classifier(x)

        return logits


class HybridModel(nn.Module):
    """Hybrid model combining transformer and GNN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Text encoder (transformer)
        self.text_encoder = TransformerModel(config)

        # Graph encoder (GNN)
        self.graph_encoder = GraphNeuralNetwork(config)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, text_data, graph_data):
        # Encode text
        text_features = self.text_encoder(
            text_data["input_ids"],
            text_data["attention_mask"]
        )

        # Encode graph
        graph_features = self.graph_encoder(graph_data)

        # Fuse features
        combined_features = torch.cat([text_features, graph_features], dim=1)
        logits = self.fusion(combined_features)

        return logits


class AdvancedMLTrainer:
    """Advanced ML trainer for gene editing models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.device = torch.device(config.device)

        # Initialize models
        self.models = {
            "transformer": TransformerModel(config),
            "gnn": GraphNeuralNetwork(config),
            "hybrid": HybridModel(config)
        }

        # Move models to device
        for model in self.models.values():
            model.to(self.device)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("AdvancedMLTrainer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def train_transformer_model(self, train_data: List[Dict], val_data: List[Dict]):
        """Train transformer model on gene editing data."""
        self.logger.info("Training transformer model...")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        )

        # Create datasets
        train_dataset = GeneEditingDataset(train_data, tokenizer, self.config.max_length)
        val_dataset = GeneEditingDataset(val_data, tokenizer, self.config.max_length)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # Initialize model and optimizer
        model = self.models["transformer"]
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float('inf')
        training_history = []

        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            # Update learning rate
            scheduler.step()

            # Log progress
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            training_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })

            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}"
            )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_model(model, "transformer_best")

        # Save training history
        self._save_training_history(training_history, "transformer")

        return model, training_history

    async def train_gnn_model(self, train_data: List[Dict], val_data: List[Dict]):
        """Train Graph Neural Network model."""
        self.logger.info("Training GNN model...")

        # Create datasets
        train_dataset = GraphDataset(train_data)
        val_dataset = GraphDataset(val_data)

        # Create data loaders
        train_loader = GraphDataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = GraphDataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # Initialize model and optimizer
        model = self.models["gnn"]
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float('inf')
        training_history = []

        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    outputs = model(batch)
                    loss = criterion(outputs, batch.y)
                    val_loss += loss.item()

            # Log progress
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            training_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}"
            )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_model(model, "gnn_best")

        # Save training history
        self._save_training_history(training_history, "gnn")

        return model, training_history

    def _save_model(self, model: nn.Module, name: str):
        """Save trained model."""
        os.makedirs(self.config.model_save_path, exist_ok=True)
        path = os.path.join(self.config.model_save_path, f"{name}.pth")
        torch.save(model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def _save_training_history(self, history: List[Dict], model_name: str):
        """Save training history."""
        os.makedirs(self.config.results_path, exist_ok=True)
        path = os.path.join(self.config.results_path, f"{model_name}_training_history.json")

        with open(path, 'w') as f:
            json.dump(history, f, indent=2)

        self.logger.info(f"Training history saved to {path}")


class ModelEvaluator:
    """Advanced model evaluator for gene editing models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("ModelEvaluator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def evaluate_model(self, model: nn.Module, test_data: List[Dict], model_type: str):
        """Evaluate trained model on test data."""
        self.logger.info(f"Evaluating {model_type} model...")

        # Prepare test data
        if model_type == "transformer":
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            )
            test_dataset = GeneEditingDataset(test_data, tokenizer, self.config.max_length)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        elif model_type == "gnn":
            test_dataset = GraphDataset(test_data)
            test_loader = GraphDataLoader(test_dataset, batch_size=self.config.batch_size)

        # Evaluation
        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                if model_type == "transformer":
                    input_ids = batch["input_ids"].to(self.config.device)
                    attention_mask = batch["attention_mask"].to(self.config.device)
                    labels = batch["labels"]
                    outputs = model(input_ids, attention_mask)
                else:
                    batch = batch.to(self.config.device)
                    outputs = model(batch)
                    labels = batch.y

                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, true_labels)

        # Save results
        self._save_evaluation_results(metrics, model_type)

        return metrics

    def _calculate_metrics(self, predictions: List[int], true_labels: List[int]) -> Dict:
        """Calculate evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "predictions": predictions,
            "true_labels": true_labels
        }

    def _save_evaluation_results(self, metrics: Dict, model_name: str):
        """Save evaluation results."""
        os.makedirs(self.config.results_path, exist_ok=True)
        path = os.path.join(self.config.results_path, f"{model_name}_evaluation.json")

        # Remove predictions and true_labels for file size
        save_metrics = {k: v for k, v in metrics.items() if k not in ["predictions", "true_labels"]}

        with open(path, 'w') as f:
            json.dump(save_metrics, f, indent=2)

        self.logger.info(f"Evaluation results saved to {path}")


class AdvancedMLPipeline:
    """Advanced ML pipeline orchestrating model training and evaluation."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.model_config = ModelConfig()
        self.logger = self._setup_logging()

        # Initialize components
        self.trainer = AdvancedMLTrainer(self.model_config)
        self.evaluator = ModelEvaluator(self.model_config)

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
        logger = logging.getLogger("AdvancedMLPipeline")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def run_complete_ml_pipeline(self, gold_data: List[Dict]) -> Dict:
        """Run complete ML pipeline on gold layer data."""
        try:
            self.logger.info("Starting advanced ML pipeline...")

            # Split data
            train_data, temp_data = train_test_split(gold_data, test_size=0.3, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

            self.logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

            # Train transformer model
            transformer_model, transformer_history = await self.trainer.train_transformer_model(
                train_data, val_data
            )

            # Train GNN model
            gnn_model, gnn_history = await self.trainer.train_gnn_model(
                train_data, val_data
            )

            # Evaluate models
            transformer_metrics = await self.evaluator.evaluate_model(
                transformer_model, test_data, "transformer"
            )

            gnn_metrics = await self.evaluator.evaluate_model(
                gnn_model, test_data, "gnn"
            )

            # Generate comprehensive report
            report = self._generate_ml_report(
                transformer_history, gnn_history,
                transformer_metrics, gnn_metrics
            )

            self.logger.info("Advanced ML pipeline completed successfully")
            return report

        except Exception as e:
            self.logger.error(f"Error in ML pipeline: {e}")
            raise

    def _generate_ml_report(self, transformer_history: List[Dict], gnn_history: List[Dict],
                           transformer_metrics: Dict, gnn_metrics: Dict) -> Dict:
        """Generate comprehensive ML pipeline report."""
        report = {
            "pipeline_execution": {
                "start_time": datetime.now().isoformat(),
                "models_trained": ["transformer", "gnn"],
                "total_training_time": "calculated_from_history"
            },
            "model_performance": {
                "transformer": {
                    "accuracy": transformer_metrics["accuracy"],
                    "precision": transformer_metrics["precision"],
                    "recall": transformer_metrics["recall"],
                    "f1_score": transformer_metrics["f1_score"]
                },
                "gnn": {
                    "accuracy": gnn_metrics["accuracy"],
                    "precision": gnn_metrics["precision"],
                    "recall": gnn_metrics["recall"],
                    "f1_score": gnn_metrics["f1_score"]
                }
            },
            "training_history": {
                "transformer": transformer_history,
                "gnn": gnn_history
            },
            "model_comparison": {
                "best_model": "transformer" if transformer_metrics["f1_score"] > gnn_metrics["f1_score"] else "gnn",
                "performance_difference": abs(transformer_metrics["f1_score"] - gnn_metrics["f1_score"])
            }
        }

        return report


# Main execution function
async def main():
    """Main function to run the advanced ML pipeline."""
    # Load gold layer data
    gold_files = glob.glob("data/gold/*.json")
    gold_data = []

    for file in gold_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                gold_data.extend(data)
            else:
                gold_data.append(data)

    # Run ML pipeline
    pipeline = AdvancedMLPipeline()
    report = await pipeline.run_complete_ml_pipeline(gold_data)

    # Save report
    with open("logs/advanced_ml_pipeline_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print("Advanced ML pipeline completed successfully!")


if __name__ == "__main__":
    import glob
    asyncio.run(main())
