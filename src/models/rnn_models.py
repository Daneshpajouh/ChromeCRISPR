import torch
import torch.nn as nn


def _gc_from_sequence_indices(x: torch.Tensor) -> torch.Tensor:
    gc_mask = ((x == 1) | (x == 2)).float()
    return gc_mask.mean(dim=1, keepdim=True)


class _BaseRNNModel(nn.Module):
    def __init__(
        self,
        rnn_type,
        input_size=21,
        embedding_dim=128,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
        use_gc_content=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_gc_content = use_gc_content

        self.embedding = nn.Embedding(4, embedding_dim)
        self.rnn = rnn_type(
            embedding_dim,
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
        embedded = self.embedding(sequence_indices)
        rnn_out, _ = self.rnn(embedded)
        features = self._final_rnn_features(rnn_out)

        if self.use_gc_content:
            if gc_content is None:
                gc_content = _gc_from_sequence_indices(sequence_indices)
            features = torch.cat([features, gc_content.float()], dim=1)

        output = self.fc_layers(features)
        return output.squeeze(-1)


class GRUModel(_BaseRNNModel):
    def __init__(self, **kwargs):
        super().__init__(nn.GRU, bidirectional=False, **kwargs)


class LSTMModel(_BaseRNNModel):
    def __init__(self, **kwargs):
        super().__init__(nn.LSTM, bidirectional=False, **kwargs)


class BiLSTMModel(_BaseRNNModel):
    def __init__(self, **kwargs):
        super().__init__(nn.LSTM, bidirectional=True, **kwargs)


def create_gru_model(
    input_size=21,
    embedding_dim=128,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    use_gc_content=False,
):
    return GRUModel(
        input_size=input_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_gc_content=use_gc_content,
    )


def create_lstm_model(
    input_size=21,
    embedding_dim=128,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    use_gc_content=False,
):
    return LSTMModel(
        input_size=input_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_gc_content=use_gc_content,
    )


def create_bilstm_model(
    input_size=21,
    embedding_dim=128,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    use_gc_content=False,
):
    return BiLSTMModel(
        input_size=input_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_gc_content=use_gc_content,
    )
