import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModel(nn.Module):
    """
    Gated Recurrent Unit model for CRISPR guide RNA efficiency prediction.

    Architecture extracted from actual saved models:
    - Embedding: 4 nucleotides -> 128 dimensions
    - GRU: 128 hidden units, 2 layers (base) or 4 layers (deep)
    - FC layers: [128, 64, 32, 1] with batch normalization
    """

    def __init__(self, input_size=21, embedding_dim=128, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Sequence embedding layer (matches saved models)
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # RNN layer (matches saved models architecture)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers (matches saved models)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),  # hidden_size -> 128
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

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size)
        x = rnn_out[:, -1, :]  # Take last output

        # Fully connected layers
        x = self.fc_layers(x)

        return x.squeeze()

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory model for CRISPR guide RNA efficiency prediction.

    Architecture extracted from actual saved models:
    - Embedding: 4 nucleotides -> 128 dimensions
    - LSTM: 128 hidden units, 2 layers (base) or 4 layers (deep)
    - FC layers: [128, 64, 32, 1] with batch normalization
    """

    def __init__(self, input_size=21, embedding_dim=128, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Sequence embedding layer (matches saved models)
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # RNN layer (matches saved models architecture)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers (matches saved models)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),  # hidden_size -> 128
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

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size)
        x = rnn_out[:, -1, :]  # Take last output

        # Fully connected layers
        x = self.fc_layers(x)

        return x.squeeze()

class BiLSTMModel(nn.Module):
    """
    Bidirectional Long Short-Term Memory model for CRISPR guide RNA efficiency prediction.

    Architecture extracted from actual saved models:
    - Embedding: 4 nucleotides -> 128 dimensions
    - BiLSTM: 128 hidden units, 2 layers, bidirectional
    - FC layers: [256, 128, 64, 32, 1] with batch normalization (256 = 128×2 for bidirectional)
    """

    def __init__(self, input_size=21, embedding_dim=128, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Sequence embedding layer (matches saved models)
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # RNN layer (matches saved models architecture)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=True)

        # Fully connected layers (matches saved models - bidirectional doubles input size)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # (hidden_size * 2) -> 128
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

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size * 2)

        # Concatenate forward and backward outputs
        last_forward = rnn_out[:, -1, :self.hidden_size]
        last_backward = rnn_out[:, 0, self.hidden_size:]
        x = torch.cat([last_forward, last_backward], dim=1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x.squeeze()

# Factory functions
def create_gru_model(input_size=21, embedding_dim=128, hidden_size=128,
                    num_layers=2, dropout=0.2):
    """Create and return a GRU model with specified parameters."""
    return GRUModel(input_size=input_size, embedding_dim=embedding_dim,
                   hidden_size=hidden_size, num_layers=num_layers,
                   dropout=dropout)

def create_lstm_model(input_size=21, embedding_dim=128, hidden_size=128,
                     num_layers=2, dropout=0.2):
    """Create and return an LSTM model with specified parameters."""
    return LSTMModel(input_size=input_size, embedding_dim=embedding_dim,
                    hidden_size=hidden_size, num_layers=num_layers,
                    dropout=dropout)

def create_bilstm_model(input_size=21, embedding_dim=128, hidden_size=128,
                       num_layers=2, dropout=0.2):
    """Create and return a BiLSTM model with specified parameters."""
    return BiLSTMModel(input_size=input_size, embedding_dim=embedding_dim,
                      hidden_size=hidden_size, num_layers=num_layers,
                      dropout=dropout)
