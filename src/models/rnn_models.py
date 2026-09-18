"""Recurrent models as .

Sections 2.5.3 to 2.5.5 for the base models,  for the deep variants. The
implementation lives in `architectures`; these names are kept so existing call sites continue to
work.
"""

import torch.nn as nn

from .architectures import DeepRNNModel, RNNModel


class GRUModel(RNNModel):
    def __init__(self, **kwargs):
        super().__init__(rnn_type=nn.GRU, **kwargs)


class LSTMModel(RNNModel):
    def __init__(self, **kwargs):
        super().__init__(rnn_type=nn.LSTM, **kwargs)


class BiLSTMModel(RNNModel):
    def __init__(self, **kwargs):
        kwargs.setdefault("bidirectional", True)
        super().__init__(rnn_type=nn.LSTM, **kwargs)


class DeepGRUModel(DeepRNNModel):
    def __init__(self, **kwargs):
        super().__init__(rnn_type=nn.GRU, **kwargs)


class DeepLSTMModel(DeepRNNModel):
    def __init__(self, **kwargs):
        super().__init__(rnn_type=nn.LSTM, **kwargs)


class DeepBiLSTMModel(DeepRNNModel):
    def __init__(self, **kwargs):
        kwargs.setdefault("bidirectional", True)
        super().__init__(rnn_type=nn.LSTM, **kwargs)


def create_gru_model(**kwargs):
    return GRUModel(**kwargs)


def create_lstm_model(**kwargs):
    return LSTMModel(**kwargs)


def create_bilstm_model(**kwargs):
    return BiLSTMModel(**kwargs)
