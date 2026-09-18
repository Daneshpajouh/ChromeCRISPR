"""The architecture family behind the published results, as defined in Train_GC.py.

Conv stack -> RNN -> flatten the whole sequence -> concatenate GC -> dense head -> sigmoid.
The flattened sequence output is what separates this family from a pooled head.
"""
import torch, torch.nn as nn

ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}


class PublishedModel(nn.Module):
    def __init__(self, family, hidden=256, n_cnn=1, n_rnn=1, kernel=3, n_fc=3,
                 dropout=0.3388134505129095, act="elu", batchnorm=False,
                 use_gc=True, seq_len=21):
        super().__init__()
        self.use_gc = use_gc
        A = ACT[act]
        layers, ch = [], 4
        for _ in range(n_cnn if "cnn" in family else 0):
            layers += [nn.Conv1d(ch, hidden, kernel, padding=kernel // 2)]
            if batchnorm:
                layers += [nn.BatchNorm1d(hidden)]
            layers += [A(alpha=1.0) if act == "elu" else A()]
            ch = hidden
        self.cnn_layers = nn.Sequential(*layers)
        self.rnn = None
        if "bilstm" in family:
            self.rnn = nn.LSTM(ch, hidden, n_rnn, batch_first=True, bidirectional=True)
        elif "lstm" in family:
            self.rnn = nn.LSTM(ch, hidden, n_rnn, batch_first=True)
        elif "gru" in family:
            self.rnn = nn.GRU(ch, hidden, n_rnn, batch_first=True)
        out = hidden * (2 if "bilstm" in family else 1) if self.rnn else ch
        feat = out * seq_len + (1 if use_gc else 0)     # the flattened sequence, plus GC
        fc, w = [nn.Linear(feat, hidden)], hidden
        for _ in range(n_fc - 1):
            fc += [A(alpha=1.0) if act == "elu" else A(), nn.Dropout(dropout),
                   nn.Linear(w, w // 2)]
            w //= 2
        fc += [nn.Linear(w, 1)]
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, x, gc=None):
        if len(self.cnn_layers):
            x = x.permute(0, 2, 1)
            x = self.cnn_layers(x)
            x = x.permute(0, 2, 1)
        if self.rnn is not None:
            x, _ = self.rnn(x)
        x = x.reshape(x.size(0), -1)
        if self.use_gc:
            x = torch.cat((x, gc.unsqueeze(1)), dim=1)
        return torch.sigmoid(self.fc_layers(x))
