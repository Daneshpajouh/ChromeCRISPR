import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModel(nn.Module):
    """
    Gated Recurrent Unit model for CRISPR guide RNA efficiency prediction.

    Architecture optimized for sequence processing with biological features.
    """

    def __init__(self, input_size=21, embedding_dim=128, hidden_size=128,
                 num_layers=2, dropout=0.2, bidirectional=False):
        super(GRUModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Sequence embedding layer
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         bidirectional=bidirectional)

        # Calculate output size
        output_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected layers
        self.fc1 = nn.Linear(output_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)

        # Use the last output
        if self.bidirectional:
            # Concatenate forward and backward outputs
            last_forward = gru_out[:, -1, :self.hidden_size]
            last_backward = gru_out[:, 0, self.hidden_size:]
            x = torch.cat([last_forward, last_backward], dim=1)
        else:
            x = gru_out[:, -1, :]  # Take last output

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory model for CRISPR guide RNA efficiency prediction.

    Architecture optimized for capturing long-term dependencies in sequence data.
    """

    def __init__(self, input_size=21, embedding_dim=128, hidden_size=128,
                 num_layers=2, dropout=0.2, bidirectional=False):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Sequence embedding layer
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=bidirectional)

        # Calculate output size
        output_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected layers
        self.fc1 = nn.Linear(output_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)

        # Use the last output
        if self.bidirectional:
            # Concatenate forward and backward outputs
            last_forward = lstm_out[:, -1, :self.hidden_size]
            last_backward = lstm_out[:, 0, self.hidden_size:]
            x = torch.cat([last_forward, last_backward], dim=1)
        else:
            x = lstm_out[:, -1, :]  # Take last output

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()

class BiLSTMModel(nn.Module):
    """
    Bidirectional Long Short-Term Memory model for CRISPR guide RNA efficiency prediction.

    Architecture that processes data in both forward and reverse directions.
    """

    def __init__(self, input_size=21, embedding_dim=128, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__()

        # Use LSTM with bidirectional=True
        self.lstm_model = LSTMModel(input_size=input_size, embedding_dim=embedding_dim,
                                   hidden_size=hidden_size, num_layers=num_layers,
                                   dropout=dropout, bidirectional=True)

    def forward(self, x):
        return self.lstm_model(x)

# Factory functions
def create_gru_model(input_size=21, embedding_dim=128, hidden_size=128,
                    num_layers=2, dropout=0.2, bidirectional=False):
    """Create and return a GRU model with specified parameters."""
    return GRUModel(input_size=input_size, embedding_dim=embedding_dim,
                   hidden_size=hidden_size, num_layers=num_layers,
                   dropout=dropout, bidirectional=bidirectional)

def create_lstm_model(input_size=21, embedding_dim=128, hidden_size=128,
                     num_layers=2, dropout=0.2, bidirectional=False):
    """Create and return an LSTM model with specified parameters."""
    return LSTMModel(input_size=input_size, embedding_dim=embedding_dim,
                    hidden_size=hidden_size, num_layers=num_layers,
                    dropout=dropout, bidirectional=bidirectional)

def create_bilstm_model(input_size=21, embedding_dim=128, hidden_size=128,
                       num_layers=2, dropout=0.2):
    """Create and return a BiLSTM model with specified parameters."""
    return BiLSTMModel(input_size=input_size, embedding_dim=embedding_dim,
                      hidden_size=hidden_size, num_layers=num_layers,
                      dropout=dropout)
