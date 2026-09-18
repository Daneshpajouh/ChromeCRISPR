"""ChromeCRISPR hybrid models as described in  of this work.

The implementation lives in `architectures`; these names are kept so existing call sites continue
to work.
"""

import torch.nn as nn

from .architectures import ChromeCRISPR


class CNNGRUModel(ChromeCRISPR):
    def __init__(self, **kwargs):
        super().__init__(rnn_type=nn.GRU, **kwargs)


class CNNLSTMModel(ChromeCRISPR):
    def __init__(self, **kwargs):
        super().__init__(rnn_type=nn.LSTM, **kwargs)


class CNNBiLSTMModel(ChromeCRISPR):
    def __init__(self, **kwargs):
        kwargs.setdefault("bidirectional", True)
        super().__init__(rnn_type=nn.LSTM, **kwargs)


def create_cnn_gru_model(**kwargs):
    return CNNGRUModel(**kwargs)


def create_cnn_lstm_model(**kwargs):
    return CNNLSTMModel(**kwargs)


def create_cnn_bilstm_model(**kwargs):
    return CNNBiLSTMModel(**kwargs)
