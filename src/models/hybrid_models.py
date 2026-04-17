import torch
import torch.nn as nn
import torch.nn.functional as F


def _gc_from_sequence_indices(x: torch.Tensor) -> torch.Tensor:
    gc_mask = ((x == 1) | (x == 2)).float()
    return gc_mask.mean(dim=1, keepdim=True)


class _BaseCNNRNNModel(nn.Module):
    def __init__(
        self,
        rnn_type,
        input_size=21,
        embedding_dim=128,
        num_filters=64,
        kernel_size=5,
        hidden_size=384,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
        use_gc_content=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_gc_content = use_gc_content

        self.embedding = nn.Embedding(4, embedding_dim)
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_filters, num_filters, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rnn = rnn_type(
            num_filters,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        directions = 2 if bidirectional else 1
        dense_in = hidden_size * directions + (1 if use_gc_content else 0)
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

    def _final_rnn_features(self, rnn_out: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            last_forward = rnn_out[:, -1, : self.hidden_size]
            last_backward = rnn_out[:, 0, self.hidden_size :]
            return torch.cat([last_forward, last_backward], dim=1)
        return rnn_out[:, -1, :]

    def forward(self, x, gc_content=None):
        sequence_indices = x.long()
        embedded = self.embedding(sequence_indices).transpose(1, 2)
        cnn_features = self.cnn_layers(embedded).transpose(1, 2)
        rnn_out, _ = self.rnn(cnn_features)
        features = self._final_rnn_features(rnn_out)

        if self.use_gc_content:
            if gc_content is None:
                gc_content = _gc_from_sequence_indices(sequence_indices)
            features = torch.cat([features, gc_content.float()], dim=1)

        output = self.fc_layers(features)
        return output.squeeze(-1)


class CNNGRUModel(_BaseCNNRNNModel):
    def __init__(self, **kwargs):
        super().__init__(nn.GRU, bidirectional=False, **kwargs)


class CNNLSTMModel(_BaseCNNRNNModel):
    def __init__(self, **kwargs):
        super().__init__(nn.LSTM, bidirectional=False, **kwargs)


class CNNBiLSTMModel(_BaseCNNRNNModel):
    def __init__(self, **kwargs):
        super().__init__(nn.LSTM, bidirectional=True, **kwargs)


def create_cnn_gru_model(
    input_size=21,
    embedding_dim=128,
    num_filters=64,
    kernel_size=5,
    hidden_size=384,
    num_layers=2,
    dropout=0.2,
    use_gc_content=True,
):
    return CNNGRUModel(
        input_size=input_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_size=kernel_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_gc_content=use_gc_content,
    )


def create_cnn_lstm_model(
    input_size=21,
    embedding_dim=128,
    num_filters=64,
    kernel_size=5,
    hidden_size=384,
    num_layers=2,
    dropout=0.2,
    use_gc_content=True,
):
    return CNNLSTMModel(
        input_size=input_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_size=kernel_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_gc_content=use_gc_content,
    )


def create_cnn_bilstm_model(
    input_size=21,
    embedding_dim=128,
    num_filters=64,
    kernel_size=5,
    hidden_size=384,
    num_layers=2,
    dropout=0.2,
    use_gc_content=True,
):
    return CNNBiLSTMModel(
        input_size=input_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_size=kernel_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_gc_content=use_gc_content,
    )
