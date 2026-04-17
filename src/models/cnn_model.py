import torch
import torch.nn as nn
import torch.nn.functional as F


def _gc_from_sequence_indices(x: torch.Tensor) -> torch.Tensor:
    gc_mask = ((x == 1) | (x == 2)).float()
    return gc_mask.mean(dim=1, keepdim=True)


class CNNModel(nn.Module):
    """CNN family used by the published ChromeCRISPR checkpoints."""

    def __init__(
        self,
        input_size=21,
        embedding_dim=128,
        num_filters=64,
        kernel_size=5,
        dropout=0.2,
        num_conv_layers=2,
        use_gc_content=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.num_conv_layers = num_conv_layers
        self.use_gc_content = use_gc_content

        self.embedding = nn.Embedding(4, embedding_dim)

        conv_layers = []
        in_channels = embedding_dim
        for _ in range(num_conv_layers):
            conv_layers.extend(
                [
                    nn.Conv1d(in_channels, num_filters, kernel_size, stride=1, padding=kernel_size // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = num_filters
        self.cnn_layers = nn.Sequential(*conv_layers)

        dense_in = num_filters + (1 if use_gc_content else 0)
        self.fc_layers = nn.Sequential(
            nn.Linear(dense_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, gc_content=None):
        sequence_indices = x.long()
        embedded = self.embedding(sequence_indices).transpose(1, 2)
        features = self.cnn_layers(embedded)
        features = F.adaptive_max_pool1d(features, 1).squeeze(-1)

        if self.use_gc_content:
            if gc_content is None:
                gc_content = _gc_from_sequence_indices(sequence_indices)
            features = torch.cat([features, gc_content.float()], dim=1)

        output = self.fc_layers(features)
        return output.squeeze(-1)


class DeepCNNModel(CNNModel):
    """4-layer CNN variant used by the deep published checkpoints."""

    def __init__(
        self,
        input_size=21,
        embedding_dim=128,
        num_filters=64,
        kernel_size=5,
        dropout=0.2,
        use_gc_content=False,
    ):
        super().__init__(
            input_size=input_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
            num_conv_layers=4,
            use_gc_content=use_gc_content,
        )


def create_cnn_model(
    input_size=21,
    embedding_dim=128,
    num_filters=64,
    kernel_size=5,
    dropout=0.2,
    use_gc_content=False,
):
    return CNNModel(
        input_size=input_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout=dropout,
        num_conv_layers=2,
        use_gc_content=use_gc_content,
    )


def create_deep_cnn_model(
    input_size=21,
    embedding_dim=128,
    num_filters=64,
    kernel_size=5,
    dropout=0.2,
    use_gc_content=False,
):
    return DeepCNNModel(
        input_size=input_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout=dropout,
        use_gc_content=use_gc_content,
    )
