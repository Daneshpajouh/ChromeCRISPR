"""Convolutional models as .

 for the base CNN,  for the deep variant. The implementation lives in
`architectures`; these names are kept so existing call sites continue to work.
"""

from .architectures import CNNModel as CNNModel, DeepCNNModel as DeepCNNModel


def create_cnn_model(**kwargs):
    return CNNModel(**kwargs)


def create_deep_cnn_model(**kwargs):
    return DeepCNNModel(**kwargs)
