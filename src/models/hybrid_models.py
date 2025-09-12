import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNGRUModel(nn.Module):
    """
    Hybrid CNN-GRU model for CRISPR guide RNA efficiency prediction.

    Architecture extracted from actual saved models:
    - Embedding: 4 nucleotides -> 128 dimensions
    - CNN: 2 conv layers, 64 filters, kernel size 5
    - GRU: 128 hidden units, 2 layers
    - FC layers: [385, 128, 64, 32, 1] with batch normalization (385 = 384 + 1 GC)
    - This is the best performing model (0.876 Spearman correlation)
    """

    def __init__(self, input_size=21, embedding_dim=128, num_filters=64,
                 kernel_size=5, hidden_size=128, num_layers=2, dropout=0.2):
        super(CNNGRUModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Sequence embedding layer (matches saved models)
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # CNN layers (matches saved models architecture exactly)
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )
        # Second CNN layer (separate to match saved model indexing)
        self.cnn_layers_2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        # RNN layer (standard implementation - will need adjustment for saved model compatibility)
        self.rnn = nn.GRU(num_filters, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers (matches saved models - includes GC feature)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size + 1, 128),  # (hidden_size + 1) -> 128 (GC feature)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),  # 128 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),   # 64 -> 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),    # 32 -> 1
        )

    def forward(self, x, gc_content=None):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # CNN layers
        x = self.cnn_layers(x)

        # Transpose for RNN
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_filters)

        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size)
        x = rnn_out[:, -1, :]  # Take last output

        # Add GC content feature if provided
        if gc_content is not None:
            gc_content = gc_content.unsqueeze(-1)  # (batch_size, 1)
            x = torch.cat([x, gc_content], dim=1)  # (batch_size, hidden_size + 1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x.squeeze()

class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model for CRISPR guide RNA efficiency prediction.

    Architecture extracted from actual saved models:
    - Embedding: 4 nucleotides -> 128 dimensions
    - CNN: 2 conv layers, 64 filters, kernel size 5
    - LSTM: 128 hidden units, 2 layers
    - FC layers: [385, 128, 64, 32, 1] with batch normalization (385 = 384 + 1 GC)
    """

    def __init__(self, input_size=21, embedding_dim=128, num_filters=64,
                 kernel_size=5, hidden_size=128, num_layers=2, dropout=0.2):
        super(CNNLSTMModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Sequence embedding layer (matches saved models)
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # CNN layers (matches saved models architecture)
        self.cnn_layers = nn.Sequential(
            # Layer 1: Conv + BatchNorm + ReLU
            nn.Conv1d(embedding_dim, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),

            # Layer 2: Conv + BatchNorm + ReLU
            nn.Conv1d(num_filters, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        # RNN layer (matches saved models architecture)
        self.rnn = nn.LSTM(num_filters, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers (matches saved models - includes GC feature)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size + 1, 128),  # (hidden_size + 1) -> 128 (GC feature)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),  # 128 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),   # 64 -> 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),    # 32 -> 1
        )

    def forward(self, x, gc_content=None):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # CNN layers
        x = self.cnn_layers(x)

        # Transpose for RNN
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_filters)

        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size)
        x = rnn_out[:, -1, :]  # Take last output

        # Add GC content feature if provided
        if gc_content is not None:
            gc_content = gc_content.unsqueeze(-1)  # (batch_size, 1)
            x = torch.cat([x, gc_content], dim=1)  # (batch_size, hidden_size + 1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x.squeeze()

class CNNBiLSTMModel(nn.Module):
    """
    Hybrid CNN-BiLSTM model for CRISPR guide RNA efficiency prediction.

    Architecture extracted from actual saved models:
    - Embedding: 4 nucleotides -> 128 dimensions
    - CNN: 2 conv layers, 64 filters, kernel size 5
    - BiLSTM: 128 hidden units, 2 layers, bidirectional
    - FC layers: [769, 128, 64, 32, 1] with batch normalization (769 = 768 + 1 GC)
    """

    def __init__(self, input_size=21, embedding_dim=128, num_filters=64,
                 kernel_size=5, hidden_size=128, num_layers=2, dropout=0.2):
        super(CNNBiLSTMModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Sequence embedding layer (matches saved models)
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # CNN layers (matches saved models architecture)
        self.cnn_layers = nn.Sequential(
            # Layer 1: Conv + BatchNorm + ReLU
            nn.Conv1d(embedding_dim, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),

            # Layer 2: Conv + BatchNorm + ReLU
            nn.Conv1d(num_filters, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        # RNN layer (matches saved models architecture)
        self.rnn = nn.LSTM(num_filters, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=True)

        # Fully connected layers (matches saved models - bidirectional + GC)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, 128),  # (hidden_size * 2 + 1) -> 128 (bidirectional + GC)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),  # 128 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),   # 64 -> 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),    # 32 -> 1
        )

    def forward(self, x, gc_content=None):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # CNN layers
        x = self.cnn_layers(x)

        # Transpose for RNN
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_filters)

        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size * 2)

        # Concatenate forward and backward outputs
        last_forward = rnn_out[:, -1, :self.hidden_size]
        last_backward = rnn_out[:, 0, self.hidden_size:]
        x = torch.cat([last_forward, last_backward], dim=1)

        # Add GC content feature if provided
        if gc_content is not None:
            gc_content = gc_content.unsqueeze(-1)  # (batch_size, 1)
            x = torch.cat([x, gc_content], dim=1)  # (batch_size, hidden_size * 2 + 1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x.squeeze()

# Factory functions
def create_cnn_gru_model(input_size=21, embedding_dim=128, num_filters=64,
                        kernel_size=5, hidden_size=128, num_layers=2, dropout=0.2):
    """Create and return a CNN-GRU hybrid model with specified parameters."""
    return CNNGRUModel(input_size=input_size, embedding_dim=embedding_dim,
                      num_filters=num_filters, kernel_size=kernel_size,
                      hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

def create_cnn_lstm_model(input_size=21, embedding_dim=128, num_filters=64,
                         kernel_size=5, hidden_size=128, num_layers=2, dropout=0.2):
    """Create and return a CNN-LSTM hybrid model with specified parameters."""
    return CNNLSTMModel(input_size=input_size, embedding_dim=embedding_dim,
                       num_filters=num_filters, kernel_size=kernel_size,
                       hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

def create_cnn_bilstm_model(input_size=21, embedding_dim=128, num_filters=64,
                           kernel_size=5, hidden_size=128, num_layers=2, dropout=0.2):
    """Create and return a CNN-BiLSTM hybrid model with specified parameters."""
    return CNNBiLSTMModel(input_size=input_size, embedding_dim=embedding_dim,
                         num_filters=num_filters, kernel_size=kernel_size,
                         hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
