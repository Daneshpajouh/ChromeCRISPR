"""The architectures must match ChromeCRISPR."""

import torch
import torch.nn as nn
import pytest

from src.models.architectures import MODELS, SEQ_LEN, N_BASES, create_random_forest

BATCH = 4


def _inputs(seed=0):
    """Varied sequences: identical rows give batch normalisation zero variance."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, N_BASES, (BATCH, SEQ_LEN), generator=g)
    x = torch.nn.functional.one_hot(idx, N_BASES).float()
    gc = ((idx == 1) | (idx == 2)).float().mean(dim=1)
    return x, gc


def _convs(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv1d) and m.kernel_size != (1,)]


def _rnns(model):
    return [m for m in model.modules() if isinstance(m, (nn.GRU, nn.LSTM))]


def _head_widths(model):
    return [l.out_features for l in model.head.modules() if isinstance(l, nn.Linear)]


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_runs(name):
    model = MODELS[name]().eval()
    x, gc = _inputs()
    with torch.no_grad():
        out = model(x, gc) if getattr(model, "use_gc_content", False) else model(x)
    assert out.shape == (BATCH, 1)


@pytest.mark.parametrize("name", ["CNN", "CNN+GC"])
def test_base_cnn_matches_article(name):
    """two convolutional layers, 128 filters, kernel 3, stride 1, padding 1."""
    c = _convs(MODELS[name]())
    assert len(c) == 2
    assert c[0].out_channels == 128
    assert c[0].kernel_size == (3,)
    assert c[0].stride == (1,)
    assert c[0].padding == (1,)


@pytest.mark.parametrize("name", ["GRU", "LSTM", "BiLSTM"])
def test_base_rnn_matches_article(name):
    """two recurrent layers of 128 hidden units."""
    r = _rnns(MODELS[name]())[0]
    assert r.num_layers == 2
    assert r.hidden_size == 128
    assert r.bidirectional == (name == "BiLSTM")


@pytest.mark.parametrize("name", ["CNN", "GRU", "LSTM", "BiLSTM"])
def test_base_head_matches_article(name):
    """Two dense layers: 64 units, then one output unit."""
    assert _head_widths(MODELS[name]()) == [64, 1]


@pytest.mark.parametrize("name", ["deepCNN", "deepGRU", "deepLSTM", "deepBiLSTM"])
def test_deep_models_match_article(name):
    """three specialized layers, dense 128, 64, 32, then the output."""
    model = MODELS[name]()
    layers = _convs(model) if name == "deepCNN" else _rnns(model)
    assert (len(layers) if name == "deepCNN" else layers[0].num_layers) == 3
    assert _head_widths(model) == [128, 64, 32, 1]


@pytest.mark.parametrize("name", ["CNN_GRU+GC", "CNN_LSTM+GC", "CNN_BiLSTM+GC"])
def test_hybrids_match_article(name):
    """3 conv x 128 and 3 recurrent x 128, fused to 256, plus GC to 257."""
    model = MODELS[name]()
    c, r = _convs(model), _rnns(model)[0]
    assert len(c) == 3 and c[0].out_channels == 128 and c[0].kernel_size == (3,)
    assert r.num_layers == 3 and r.hidden_size == 128
    assert r.input_size == 128, "CNN followed by RNN: the RNN reads the CNN feature map"
    head = [l for l in model.head.modules() if isinstance(l, nn.Linear)]
    assert head[0].in_features == 257
    assert _head_widths(model) == [128, 64, 32, 1]


@pytest.mark.parametrize("readout", ["flatten_proj", "max", "mean"])
def test_every_faithful_readout_runs(readout):
    """The article does not specify the CNN branch's reduction, so all readings must work."""
    model = MODELS["CNN_GRU+GC"](cnn_readout=readout).eval()
    x, gc = _inputs()
    with torch.no_grad():
        assert model(x, gc).shape == (BATCH, 1)


def test_unknown_readout_is_rejected():
    with pytest.raises(ValueError):
        MODELS["CNN_GRU+GC"](cnn_readout="not-a-reading")


def test_one_hot_and_index_inputs_agree():
    model = MODELS["CNN_GRU+GC"]().eval()
    idx = torch.randint(0, N_BASES, (BATCH, SEQ_LEN))
    one_hot = torch.nn.functional.one_hot(idx, N_BASES).float()
    gc = ((idx == 1) | (idx == 2)).float().mean(dim=1)
    with torch.no_grad():
        assert torch.allclose(model(idx, gc), model(one_hot, gc), atol=1e-6)


def test_every_parameter_receives_gradient():
    model = MODELS["CNN_GRU+GC"]().train()
    x, gc = _inputs()  # varied sequences, see _inputs
    model(x, gc).sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert dead == []


def test_transformer_matches_article():
    """8 attention heads, three layers of 128 hidden units, positional
    encoding, layer normalisation, then dense layers with batch normalisation."""
    model = MODELS["Transformer"]()
    enc = [l for l in model.modules() if isinstance(l, nn.TransformerEncoderLayer)]
    assert len(enc) == 3
    assert enc[0].self_attn.num_heads == 8
    assert enc[0].self_attn.embed_dim == 128
    assert any(isinstance(l, nn.LayerNorm) for l in model.modules())
    assert hasattr(model, "positional")
    assert [l.out_features for l in model.head.modules() if isinstance(l, nn.Linear)] == \
        [128, 64, 32, 1]


def test_random_forest_matches_article():
    """RandomForestRegressor with 100 estimators."""
    assert create_random_forest().n_estimators == 100
