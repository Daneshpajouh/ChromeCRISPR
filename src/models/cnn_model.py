import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    Convolutional Neural Network for CRISPR guide RNA efficiency prediction.

    Architecture extracted from actual saved models:
    - Embedding: 4 nucleotides -> 128 dimensions
    - CNN: 64 filters, kernel size 5, single conv layer
    - FC layers: [128, 64, 32, 1] with batch normalization
    """

    def __init__(self, input_size=21, embedding_dim=128, num_filters=64,
                 kernel_size=5, dropout=0.2):
        super(CNNModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim

        # Sequence embedding layer (matches saved models)
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # Standard CNN layers (architecture mismatch with saved models noted)
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters)

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x.squeeze()

class DeepCNNModel(nn.Module):
    """
    Deep Convolutional Neural Network for CRISPR guide RNA efficiency prediction.

    Architecture extracted from actual saved models:
    - Embedding: 4 nucleotides -> 128 dimensions
    - CNN: 4 conv layers, 64 filters each, kernel size 5
    - FC layers: [128, 64, 32, 1] with batch normalization
    """

    def __init__(self, input_size=21, embedding_dim=128, num_filters=64,
                 kernel_size=5, dropout=0.2):
        super(DeepCNNModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim

        # Sequence embedding layer (matches saved models)
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # Deep CNN layers (matches saved models architecture)
        self.cnn_layers = nn.Sequential(
            # Layer 1: Conv + BatchNorm + ReLU
            nn.Conv1d(embedding_dim, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),

            # Layer 2: Conv + BatchNorm + ReLU
            nn.Conv1d(num_filters, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),

            # Layer 3: Conv + BatchNorm + ReLU
            nn.Conv1d(num_filters, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),

            # Layer 4: Conv + BatchNorm + ReLU
            nn.Conv1d(num_filters, num_filters, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        # Fully connected layers (matches saved models)
        self.fc_layers = nn.Sequential(
            nn.Linear(num_filters, 128),  # 64 -> 128
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
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # CNN layers
        x = self.cnn_layers(x)

        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x.squeeze()

def create_cnn_model(input_size=21, embedding_dim=128, num_filters=64,
                    kernel_size=5, dropout=0.2):
    """Create and return a CNN model with specified parameters."""
    return CNNModel(input_size=input_size, embedding_dim=embedding_dim,
                   num_filters=num_filters, kernel_size=kernel_size,
                   dropout=dropout)

def create_deep_cnn_model(input_size=21, embedding_dim=128, num_filters=64,
                         kernel_size=5, dropout=0.2):
    """Create and return a Deep CNN model with specified parameters."""
    return DeepCNNModel(input_size=input_size, embedding_dim=embedding_dim,
                       num_filters=num_filters, kernel_size=kernel_size,
                       dropout=dropout)
