"""Every architectural statement the article makes, checked against the code.

One assertion per claim, so a change that silently departs from the published description
fails here rather than being discovered by a reader rebuilding the model.
"""
import pytest
import torch.nn as nn

from src.models.architectures import MODELS, create_random_forest

BASE_RNN = [("GRU", False), ("LSTM", False), ("BiLSTM", True)]
DEEP = ["deepCNN", "deepGRU", "deepLSTM", "deepBiLSTM"]
HYBRIDS = ["CNN_GRU+GC", "CNN_LSTM+GC", "CNN_BiLSTM+GC"]


def convs(m):
    return [x for x in m.modules() if isinstance(x, nn.Conv1d) and x.kernel_size != (1,)]


def rnns(m):
    return [x for x in m.modules() if isinstance(x, (nn.GRU, nn.LSTM, nn.RNN))]


def dense(m):
    return [x for x in m.modules() if isinstance(x, nn.Linear)]


def test_cnn_convolutions():
    """Two convolutional layers, 128 filters, kernel 3, stride 1, padding 1."""
    c = convs(MODELS["CNN"]())
    assert len(c) == 2
    assert [x.out_channels for x in c] == [128, 128]
    assert all(x.kernel_size == (3,) and x.stride == (1,) and x.padding == (1,) for x in c)


def test_cnn_activation_and_head():
    """ReLU, then two dense layers with batch normalisation, 64 units then one output."""
    m = MODELS["CNN"]()
    assert any(isinstance(x, nn.ReLU) for x in m.modules())
    assert [l.out_features for l in dense(m)][-2:] == [64, 1]
    assert any(isinstance(x, nn.BatchNorm1d) for x in m.modules())


@pytest.mark.parametrize("name,bidirectional", BASE_RNN)
def test_base_recurrent(name, bidirectional):
    """Two recurrent layers of 128 hidden units; bidirectional only for BiLSTM."""
    m = MODELS[name]()
    r = rnns(m)[0]
    assert (r.num_layers, r.hidden_size, r.bidirectional) == (2, 128, bidirectional)
    assert [l.out_features for l in dense(m)][-2:] == [64, 1]


@pytest.mark.parametrize("name", DEEP)
def test_deep_models(name):
    """Three specialized layers of 128, then dense layers of 128, 64 and 32."""
    m = MODELS[name]()
    if name == "deepCNN":
        assert len(convs(m)) == 3
        assert [x.out_channels for x in convs(m)] == [128, 128, 128]
    else:
        assert rnns(m)[0].num_layers == 3
        assert rnns(m)[0].hidden_size == 128
    assert [l.out_features for l in dense(m)][-4:] == [128, 64, 32, 1]


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_models(name):
    """Three convolutions and three recurrent layers of 128, fused to 256, 257 with GC."""
    m = MODELS[name]()
    assert len(convs(m)) == 3
    assert [x.out_channels for x in convs(m)] == [128, 128, 128]
    assert rnns(m)[0].num_layers == 3
    assert rnns(m)[0].hidden_size == 128
    assert dense(m)[-4].in_features == 257
    assert [l.out_features for l in dense(m)][-4:] == [128, 64, 32, 1]


def test_transformer():
    """Eight attention heads, three layers of 128 hidden units, layer normalisation."""
    m = MODELS["Transformer"]()
    enc = [x for x in m.modules() if isinstance(x, nn.TransformerEncoderLayer)]
    assert len(enc) == 3
    assert all(x.self_attn.num_heads == 8 for x in enc)
    assert any(isinstance(x, nn.LayerNorm) for x in m.modules())


def test_random_forest():
    """A random forest of 100 estimators."""
    assert create_random_forest().n_estimators == 100
