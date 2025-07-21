import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNGRUModel(nn.Module):
    """
    Hybrid CNN-GRU model for CRISPR guide RNA efficiency prediction.

    Architecture combines CNN for local feature extraction and GRU for sequence modeling.
    This is the best performing model (0.876 Spearman correlation).
    """

    def __init__(self, input_size=21, embedding_dim=128, num_filters=128,
                 kernel_size=3, hidden_size=128, num_layers=2, dropout=0.2):
        super(CNNGRUModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Sequence embedding layer
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size,
                              stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size,
                              stride=1, padding=1)

        # Batch normalization for CNN
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)

        # GRU layer for sequence modeling
        self.gru = nn.GRU(num_filters, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

        # Batch normalization for FC layers
        self.fc_bn1 = nn.BatchNorm1d(hidden_size // 2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Transpose back for RNN
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_filters)

        # GRU sequence modeling
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)
        x = gru_out[:, -1, :]  # Take last output

        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()

class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model for CRISPR guide RNA efficiency prediction.

    Architecture combines CNN for local feature extraction and LSTM for sequence modeling.
    """

    def __init__(self, input_size=21, embedding_dim=128, num_filters=128,
                 kernel_size=3, hidden_size=128, num_layers=2, dropout=0.2):
        super(CNNLSTMModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Sequence embedding layer
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size,
                              stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size,
                              stride=1, padding=1)

        # Batch normalization for CNN
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)

        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

        # Batch normalization for FC layers
        self.fc_bn1 = nn.BatchNorm1d(hidden_size // 2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Transpose back for RNN
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_filters)

        # LSTM sequence modeling
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        x = lstm_out[:, -1, :]  # Take last output

        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()

class CNNBiLSTMModel(nn.Module):
    """
    Hybrid CNN-BiLSTM model for CRISPR guide RNA efficiency prediction.

    Architecture combines CNN for local feature extraction and BiLSTM for bidirectional sequence modeling.
    """

    def __init__(self, input_size=21, embedding_dim=128, num_filters=128,
                 kernel_size=3, hidden_size=128, num_layers=2, dropout=0.2):
        super(CNNBiLSTMModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Sequence embedding layer
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size,
                              stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size,
                              stride=1, padding=1)

        # Batch normalization for CNN
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)

        # BiLSTM layer for sequence modeling
        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.fc2 = nn.Linear(hidden_size, 1)

        # Batch normalization for FC layers
        self.fc_bn1 = nn.BatchNorm1d(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Transpose back for RNN
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_filters)

        # BiLSTM sequence modeling
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)

        # Concatenate forward and backward outputs
        last_forward = lstm_out[:, -1, :self.hidden_size]
        last_backward = lstm_out[:, 0, self.hidden_size:]
        x = torch.cat([last_forward, last_backward], dim=1)

        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()

# Factory functions
def create_cnn_gru_model(input_size=21, embedding_dim=128, num_filters=128,
                        kernel_size=3, hidden_size=128, num_layers=2, dropout=0.2):
    """Create and return a CNN-GRU hybrid model with specified parameters."""
    return CNNGRUModel(input_size=input_size, embedding_dim=embedding_dim,
                      num_filters=num_filters, kernel_size=kernel_size,
                      hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

def create_cnn_lstm_model(input_size=21, embedding_dim=128, num_filters=128,
                         kernel_size=3, hidden_size=128, num_layers=2, dropout=0.2):
    """Create and return a CNN-LSTM hybrid model with specified parameters."""
    return CNNLSTMModel(input_size=input_size, embedding_dim=embedding_dim,
                       num_filters=num_filters, kernel_size=kernel_size,
                       hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

def create_cnn_bilstm_model(input_size=21, embedding_dim=128, num_filters=128,
                           kernel_size=3, hidden_size=128, num_layers=2, dropout=0.2):
    """Create and return a CNN-BiLSTM hybrid model with specified parameters."""
    return CNNBiLSTMModel(input_size=input_size, embedding_dim=embedding_dim,
                         num_filters=num_filters, kernel_size=kernel_size,
                         hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
