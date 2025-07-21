import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    Convolutional Neural Network for CRISPR guide RNA efficiency prediction.

    Architecture as described in the manuscript:
    - Two convolutional layers, each with 128 filters, kernel size 3, stride 1, padding 1
    - ReLU activation after each convolutional layer
    - Flattened output concatenated
    - Two fully connected layers with batch normalization
    - First FC layer: 64 units, Second FC layer: 1 output unit
    - Sequence embedding to 1-dimensional tensor of size 128
    - Batch size: 64
    """

    def __init__(self, input_size=21, embedding_dim=128, num_filters=128,
                 kernel_size=3, fc_units=64, dropout=0.2):
        super(CNNModel, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim

        # Sequence embedding layer
        self.embedding = nn.Embedding(4, embedding_dim)  # 4 nucleotides: A, C, G, T

        # Convolutional layers
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size,
                              stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size,
                              stride=1, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters * input_size, fc_units)
        self.fc2 = nn.Linear(fc_units, 1)

        # Batch normalization for FC layers
        self.fc_bn1 = nn.BatchNorm1d(fc_units)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # First convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))

        # Second convolutional layer
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten
        x = x.view(batch_size, -1)

        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()

def create_cnn_model(input_size=21, embedding_dim=128, num_filters=128,
                    kernel_size=3, fc_units=64, dropout=0.2):
    """Create and return a CNN model with specified parameters."""
    return CNNModel(input_size=input_size, embedding_dim=embedding_dim,
                   num_filters=num_filters, kernel_size=kernel_size,
                   fc_units=fc_units, dropout=dropout)
