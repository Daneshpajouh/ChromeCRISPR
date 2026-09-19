"""Model architectures for CRISPR/Cas9 on-target activity prediction.

ENCODING
    Each nucleotide of the 21-mer is one-hot encoded, giving a 21 x 4 matrix, then embedded
    per position to 128 dimensions.

GC CONTENT
    The proportion of G and C in the sequence, appended as a single input in the last layer of
    the models whose names end in +GC.

BASE MODELS
    CNN     two convolutional layers of 128 filters, kernel 3, stride 1, padding 1, ReLU; the
            output is flattened and passed through two dense layers with batch normalisation,
            64 units then one output unit.
    GRU     two GRU layers of 128 hidden units, then the same head.
    LSTM    as GRU, with LSTM layers.
    BiLSTM  two bidirectional LSTM layers of 128 hidden units, then the same head.

DEEP MODELS
    Three specialized layers, convolutional or recurrent, of 128 filters or hidden units,
    followed by three dense layers with batch normalisation of 128, 64 and 32 units, then the
    output layer.

CHROMECRISPR HYBRIDS
    A convolutional branch of three layers of 128 filters feeding a recurrent branch of three
    layers of 128 hidden units. Both branches emit 128 dimensions, which are concatenated to
    256, combined with GC content to 257, and passed through dense layers of 128, 64 and 32
    before the output layer.

TRANSFORMER
    Multi-head self-attention with 8 heads and position-wise feed-forward networks, positional
    encoding, layer normalisation, three layers of 128 hidden units, then dense layers with
    batch normalisation.

CONFIGURABLE
    The reduction from the convolutional branch's 128 x 21 feature map to its 128-dimensional
    output is selectable through `cnn_readout`:

        "flatten_proj"  flatten to 128*21 then a learned projection to 128   [default]
        "max"           global max over the 21 positions
        "mean"          global average over the 21 positions

    The default matches the base CNN, whose convolutional output is flattened before the dense
    layers. It costs 344,192 parameters; either pooling costs none.

    Also configurable, with the defaults given as arguments: the dropout rate, whether batch
    normalisation is applied to the convolutional layers, and the recurrent branch's reduction
    over timesteps, which defaults to the last hidden state.
"""

import torch
import torch.nn as nn

SEQ_LEN = 21
N_BASES = 4
EMBED_DIM = 128
FILTERS = 128
KERNEL = 3
HIDDEN = 128


def _dense_stack(in_features, widths, dropout):
    """Fully connected layers with batch normalization, then a single output unit."""
    layers, prev = [], in_features
    for w in widths:
        layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = w
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class _Embedding(nn.Module):
    """One-hot 21 x 4 to a 128-dimensional representation per position."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Conv1d(N_BASES, embed_dim, kernel_size=1)

    def forward(self, x):                       # x: (B, 21, 4) one-hot, or (B, 21) indices
        if x.dim() == 2:
            x = nn.functional.one_hot(x.long(), N_BASES).float()
        return self.proj(x.transpose(1, 2))     # (B, embed_dim, 21)


class CNNModel(nn.Module):
    """: two conv layers of 128 filters, kernel 3, stride 1, padding 1, ReLU;
    output flattened; two dense layers with batch normalization, 64 units then one output."""

    def __init__(self, n_conv=2, dense_widths=(64,), use_gc_content=False,
                 dropout=0.0, conv_batch_norm=False):
        super().__init__()
        self.use_gc_content = use_gc_content
        self.embedding = _Embedding()
        layers, in_ch = [], EMBED_DIM
        for _ in range(n_conv):
            layers.append(nn.Conv1d(in_ch, FILTERS, KERNEL, stride=1, padding=1))
            if conv_batch_norm:
                layers.append(nn.BatchNorm1d(FILTERS))
            layers.append(nn.ReLU())
            in_ch = FILTERS
        self.conv = nn.Sequential(*layers)
        self.head = _dense_stack(FILTERS * SEQ_LEN + (1 if use_gc_content else 0),
                                 dense_widths, dropout)

    def forward(self, x, gc_content=None):
        h = self.conv(self.embedding(x)).flatten(1)     # "flattened and concatenated"
        if self.use_gc_content:
            h = torch.cat([h, gc_content.view(-1, 1)], dim=1)
        return self.head(h)


class DeepCNNModel(CNNModel):
    """: three convolutional layers of 128 filters; dense 128, 64, 32."""

    def __init__(self, **kw):
        kw.setdefault("n_conv", 3)
        kw.setdefault("dense_widths", (128, 64, 32))
        super().__init__(**kw)


class RNNModel(nn.Module):
    """Sections 2.5.3 to 2.5.5: two recurrent layers of 128 hidden units, then two dense
    layers with batch normalization, 64 units then one output."""

    def __init__(self, rnn_type=nn.GRU, num_layers=2, bidirectional=False,
                 dense_widths=(64,), use_gc_content=False, dropout=0.0):
        super().__init__()
        self.use_gc_content = use_gc_content
        self.bidirectional = bidirectional
        self.embedding = _Embedding()
        self.rnn = rnn_type(EMBED_DIM, HIDDEN, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        out = HIDDEN * (2 if bidirectional else 1)
        self.head = _dense_stack(out + (1 if use_gc_content else 0), dense_widths, dropout)

    def _reduce(self, out):
        if self.bidirectional:
            return torch.cat([out[:, -1, :HIDDEN], out[:, 0, HIDDEN:]], dim=1)
        return out[:, -1, :]

    def forward(self, x, gc_content=None):
        out, _ = self.rnn(self.embedding(x).transpose(1, 2))
        h = self._reduce(out)
        if self.use_gc_content:
            h = torch.cat([h, gc_content.view(-1, 1)], dim=1)
        return self.head(h)


class DeepRNNModel(RNNModel):
    """: three recurrent layers of 128 hidden units; dense 128, 64, 32."""

    def __init__(self, **kw):
        kw.setdefault("num_layers", 3)
        kw.setdefault("dense_widths", (128, 64, 32))
        super().__init__(**kw)


class ChromeCRISPR(nn.Module):
    """: CNN followed by RNN, both branches emitting 128 dimensions, concatenated
    to 256, combined with GC content to 257, then dense layers of 128, 64 and 32."""

    def __init__(self, rnn_type=nn.GRU, bidirectional=False, cnn_readout="flatten_proj",
                 use_gc_content=True, dropout=0.0, conv_batch_norm=False):
        super().__init__()
        self.use_gc_content = use_gc_content
        self.cnn_readout = cnn_readout
        self.embedding = _Embedding()
        layers, in_ch = [], EMBED_DIM
        for _ in range(3):                                   # "three convolutional layers"
            layers.append(nn.Conv1d(in_ch, FILTERS, KERNEL, stride=1, padding=1))
            if conv_batch_norm:
                layers.append(nn.BatchNorm1d(FILTERS))
            layers.append(nn.ReLU())
            in_ch = FILTERS
        self.conv = nn.Sequential(*layers)
        if cnn_readout == "flatten_proj":
            self.cnn_out = nn.Linear(FILTERS * SEQ_LEN, 128)
        elif cnn_readout not in ("max", "mean"):
            raise ValueError("cnn_readout must be flatten_proj, max or mean")
        # "CNN followed by RNN": the recurrent branch reads the convolutional feature map.
        self.rnn = rnn_type(FILTERS, HIDDEN, num_layers=3, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if dropout else 0.0)
        rnn_out = HIDDEN * (2 if bidirectional else 1)
        if bidirectional:
            self.rnn_proj = nn.Linear(rnn_out, 128)          # branch still emits 128
        self.bidirectional = bidirectional
        self.head = _dense_stack(256 + (1 if use_gc_content else 0), (128, 64, 32), dropout)

    def forward(self, x, gc_content=None):
        feat = self.conv(self.embedding(x))                  # (B, 128, 21)
        if self.cnn_readout == "flatten_proj":
            c = self.cnn_out(feat.flatten(1))
        elif self.cnn_readout == "max":
            c = feat.max(dim=2).values
        else:
            c = feat.mean(dim=2)
        out, _ = self.rnn(feat.transpose(1, 2))
        r = torch.cat([out[:, -1, :HIDDEN], out[:, 0, HIDDEN:]], dim=1) if self.bidirectional \
            else out[:, -1, :]
        if self.bidirectional:
            r = self.rnn_proj(r)
        h = torch.cat([c, r], dim=1)                         # 256 features
        if self.use_gc_content:
            h = torch.cat([h, gc_content.view(-1, 1)], dim=1)   # 257 features
        return self.head(h)


class TransformerModel(nn.Module):
    """: a multi-head self-attention model with 8 heads, positional encoding,
    layer normalisation, three transformer layers of 128 hidden units, then dense layers with
    batch normalisation.
    """

    def __init__(self, n_heads=8, n_layers=3, dense_widths=(128, 64, 32),
                 use_gc_content=False, dropout=0.0):
        super().__init__()
        self.use_gc_content = use_gc_content
        self.embedding = _Embedding()
        self.positional = nn.Parameter(torch.zeros(1, SEQ_LEN, EMBED_DIM))
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=n_heads, dim_feedforward=EMBED_DIM * 4,
            dropout=dropout, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = _dense_stack(EMBED_DIM + (1 if use_gc_content else 0),
                                 dense_widths, dropout)

    def forward(self, x, gc_content=None):
        h = self.embedding(x).transpose(1, 2) + self.positional
        h = self.norm(self.encoder(h)).mean(dim=1)
        if self.use_gc_content:
            h = torch.cat([h, gc_content.view(-1, 1)], dim=1)
        return self.head(h)


def create_random_forest(random_state=0):
    """Random forest regressor with 100 estimators."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, random_state=random_state)


MODELS = {
    "CNN":            lambda **k: CNNModel(**k),
    "GRU":            lambda **k: RNNModel(rnn_type=nn.GRU, **k),
    "LSTM":           lambda **k: RNNModel(rnn_type=nn.LSTM, **k),
    "BiLSTM":         lambda **k: RNNModel(rnn_type=nn.LSTM, bidirectional=True, **k),
    "deepCNN":        lambda **k: DeepCNNModel(**k),
    "deepGRU":        lambda **k: DeepRNNModel(rnn_type=nn.GRU, **k),
    "deepLSTM":       lambda **k: DeepRNNModel(rnn_type=nn.LSTM, **k),
    "deepBiLSTM":     lambda **k: DeepRNNModel(rnn_type=nn.LSTM, bidirectional=True, **k),
    "CNN+GC":         lambda **k: CNNModel(use_gc_content=True, **k),
    "GRU+GC":         lambda **k: RNNModel(rnn_type=nn.GRU, use_gc_content=True, **k),
    "LSTM+GC":        lambda **k: RNNModel(rnn_type=nn.LSTM, use_gc_content=True, **k),
    "BiLSTM+GC":      lambda **k: RNNModel(rnn_type=nn.LSTM, bidirectional=True,
                                           use_gc_content=True, **k),
    "deepCNN+GC":     lambda **k: DeepCNNModel(use_gc_content=True, **k),
    "deepGRU+GC":     lambda **k: DeepRNNModel(rnn_type=nn.GRU, use_gc_content=True, **k),
    "deepLSTM+GC":    lambda **k: DeepRNNModel(rnn_type=nn.LSTM, use_gc_content=True, **k),
    "deepBiLSTM+GC":  lambda **k: DeepRNNModel(rnn_type=nn.LSTM, bidirectional=True,
                                               use_gc_content=True, **k),
    "CNN_GRU+GC":     lambda **k: ChromeCRISPR(rnn_type=nn.GRU, **k),
    "CNN_LSTM+GC":    lambda **k: ChromeCRISPR(rnn_type=nn.LSTM, **k),
    "CNN_BiLSTM+GC":  lambda **k: ChromeCRISPR(rnn_type=nn.LSTM, bidirectional=True, **k),
    "Transformer":    lambda **k: TransformerModel(**k),
}
