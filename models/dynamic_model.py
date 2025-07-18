import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import math

class DynamicModel(nn.Module):
    """
    Dynamic model supporting multiple architectures for CRISPR/Cas9 activity prediction.
    Focuses on CNN-BiLSTM+GC hybrid achieving 0.8768 Spearman correlation.
    """

    def __init__(self, config: Dict[str, Any]):
        super(DynamicModel, self).__init__()
        self.config = config

        # Model dimensions
        self.sequence_length = config.get('sequence_length', 21)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.hidden_size = config.get('hidden_size', 256)
        self.num_classes = config.get('num_classes', 1)

        # Biofeatures
        self.biofeature_dim = config.get('biofeature_dim', 4)
        self.use_biofeatures = config.get('use_biofeatures', True)

        # Architecture components
        self.use_cnn = config.get('use_cnn', True)
        self.use_rnn = config.get('use_rnn', True)
        self.rnn_type = config.get('rnn_type', 'bilstm')
        self.use_mlp_mixer = config.get('use_mlp_mixer', True)

        # Build model components
        self._build_embedding()
        self._build_cnn_layers()
        self._build_rnn_layers()
        self._build_mlp_mixer()
        self._build_final_layers()

    def _build_embedding(self):
        """Build embedding layer for DNA sequences."""
        self.embedding = nn.Embedding(4, self.embedding_dim)  # A, T, G, C

    def _build_cnn_layers(self):
        """Build CNN layers for feature extraction."""
        if not self.use_cnn:
            self.cnn_layers = None
            return

        num_cnn_layers = self.config.get('num_cnn_layers', 2)
        kernel_size = self.config.get('cnn_kernel_size', 5)
        filters = self.config.get('cnn_filters', 64)

        self.cnn_layers = nn.ModuleList()
        in_channels = self.embedding_dim

        for i in range(num_cnn_layers):
            self.cnn_layers.append(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2)
            )
            in_channels = filters

        self.cnn_batch_norm = nn.BatchNorm1d(filters)
        self.cnn_dropout = nn.Dropout(self.config.get('dropout_rate', 0.1))

    def _build_rnn_layers(self):
        """Build RNN layers for sequential modeling."""
        if not self.use_rnn:
            self.rnn_layers = None
            return

        rnn_hidden_size = self.config.get('rnn_hidden_size', 128)
        num_rnn_layers = self.config.get('num_rnn_layers', 2)
        dropout = self.config.get('dropout_rate', 0.1)

        if self.rnn_type == 'bilstm':
            self.rnn_layers = nn.LSTM(
                input_size=self.hidden_size if self.use_cnn else self.embedding_dim,
                hidden_size=rnn_hidden_size,
                num_layers=num_rnn_layers,
                dropout=dropout if num_rnn_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        elif self.rnn_type == 'lstm':
            self.rnn_layers = nn.LSTM(
                input_size=self.hidden_size if self.use_cnn else self.embedding_dim,
                hidden_size=rnn_hidden_size,
                num_layers=num_rnn_layers,
                dropout=dropout if num_rnn_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        elif self.rnn_type == 'gru':
            self.rnn_layers = nn.GRU(
                input_size=self.hidden_size if self.use_cnn else self.embedding_dim,
                hidden_size=rnn_hidden_size,
                num_layers=num_rnn_layers,
                dropout=dropout if num_rnn_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )

        self.rnn_dropout = nn.Dropout(dropout)

    def _build_mlp_mixer(self):
        """Build MLP Mixer layers for token and channel mixing."""
        if not self.use_mlp_mixer:
            self.mlp_mixer = None
            return

        depth = self.config.get('mlp_mixer_depth', 1)
        token_dim = self.config.get('mlp_mixer_token_dim', 172)
        channel_dim = self.config.get('mlp_mixer_channel_dim', 197)

        self.mlp_mixer = MLPMixer(
            sequence_length=self.sequence_length,
            token_dim=token_dim,
            channel_dim=channel_dim,
            depth=depth
        )

    def _build_final_layers(self):
        """Build final fully connected layers."""
        # Calculate input size for final layers
        input_size = 0

        if self.use_rnn:
            rnn_hidden_size = self.config.get('rnn_hidden_size', 128)
            if self.rnn_type == 'bilstm':
                input_size += rnn_hidden_size * 2  # Bidirectional
            else:
                input_size += rnn_hidden_size
        elif self.use_cnn:
            input_size += self.config.get('cnn_filters', 64)
        else:
            input_size += self.embedding_dim

        if self.use_biofeatures:
            input_size += self.biofeature_dim

        # Final layers
        num_fc_layers = self.config.get('num_fc_layers', 3)
        fc_sizes = [256, 128, self.num_classes]

        self.fc_layers = nn.ModuleList()
        current_size = input_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(current_size, fc_sizes[i]))
            current_size = fc_sizes[i]

        self.fc_dropout = nn.Dropout(self.config.get('dropout_rate', 0.1))
        self.activation = nn.ReLU()

    def forward(self, sequence: torch.Tensor, biofeatures: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            sequence: Input DNA sequence tensor [batch_size, sequence_length]
            biofeatures: Optional biofeatures tensor [batch_size, biofeature_dim]

        Returns:
            Predicted CRISPR activity scores [batch_size, 1]
        """
        batch_size = sequence.size(0)

        # Embedding
        x = self.embedding(sequence)  # [batch_size, seq_len, embedding_dim]

        # CNN processing
        if self.use_cnn and self.cnn_layers is not None:
            x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]

            for conv in self.cnn_layers:
                x = conv(x)
                x = self.activation(x)
                x = self.cnn_batch_norm(x)
                x = self.cnn_dropout(x)

            x = x.transpose(1, 2)  # [batch_size, seq_len, filters]

        # RNN processing
        if self.use_rnn and self.rnn_layers is not None:
            rnn_out, _ = self.rnn_layers(x)
            if self.rnn_type == 'bilstm':
                # Concatenate forward and backward outputs
                x = rnn_out[:, -1, :]  # Take last time step
            else:
                x = rnn_out[:, -1, :]  # Take last time step
            x = self.rnn_dropout(x)

        # MLP Mixer processing
        if self.use_mlp_mixer and self.mlp_mixer is not None:
            x = self.mlp_mixer(x)

        # Biofeatures integration
        if self.use_biofeatures and biofeatures is not None:
            x = torch.cat([x, biofeatures], dim=1)

        # Final layers
        for i, fc in enumerate(self.fc_layers[:-1]):
            x = fc(x)
            x = self.activation(x)
            x = self.fc_dropout(x)

        # Output layer
        x = self.fc_layers[-1](x)
        x = torch.sigmoid(x)  # Output between 0 and 1

        return x

class MLPMixer(nn.Module):
    """MLP Mixer for token and channel mixing."""

    def __init__(self, sequence_length: int, token_dim: int, channel_dim: int, depth: int):
        super(MLPMixer, self).__init__()
        self.sequence_length = sequence_length
        self.token_dim = token_dim
        self.channel_dim = channel_dim
        self.depth = depth

        self.token_mixers = nn.ModuleList([
            nn.Linear(sequence_length, sequence_length) for _ in range(depth)
        ])

        self.channel_mixers = nn.ModuleList([
            nn.Linear(token_dim, token_dim) for _ in range(depth)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(token_dim) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP Mixer."""
        # x: [batch_size, seq_len, token_dim]

        for i in range(self.depth):
            # Token mixing
            x_transposed = x.transpose(1, 2)  # [batch_size, token_dim, seq_len]
            x_transposed = self.token_mixers[i](x_transposed)
            x = x_transposed.transpose(1, 2)  # [batch_size, seq_len, token_dim]
            x = self.layer_norms[i](x)

            # Channel mixing
            x = self.channel_mixers[i](x)
            x = self.layer_norms[i](x)

        # Global average pooling
        x = torch.mean(x, dim=1)  # [batch_size, token_dim]

        return x

def create_model(config: Dict[str, Any]) -> DynamicModel:
    """
    Factory function to create a model with the specified configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Configured DynamicModel instance
    """
    return DynamicModel(config)

def get_best_model_config() -> Dict[str, Any]:
    """
    Get configuration for the best performing model (0.8768 Spearman correlation).

    Returns:
        Configuration dictionary for CNN-BiLSTM+GC hybrid model
    """
    return {
                    'sequence_length': 21,
        'embedding_dim': 128,
        'hidden_size': 256,
        'num_classes': 1,
        'biofeature_dim': 4,
        'use_biofeatures': True,
        'use_cnn': True,
        'use_rnn': True,
        'rnn_type': 'bilstm',
        'use_mlp_mixer': True,
        'num_cnn_layers': 2,
        'cnn_kernel_size': 5,
        'cnn_filters': 64,
        'rnn_hidden_size': 128,
        'num_rnn_layers': 2,
        'mlp_mixer_depth': 1,
        'mlp_mixer_token_dim': 172,
        'mlp_mixer_channel_dim': 197,
        'num_fc_layers': 3,
        'dropout_rate': 0.18458170507780688,
        'learning_rate': 0.0011062631134035848,
        'batch_size': 64,
        'weight_decay': 1.614789571500063e-05,
        'num_epochs': 113
    }
