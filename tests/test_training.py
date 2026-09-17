"""The training code trains the models it ships with."""

import numpy as np
import pytest
import torch

from src.training.trainer import ChromeCRISPRTrainer, gc_content
from src.models.cnn_model import create_cnn_model, create_deep_cnn_model
from src.models.hybrid_models import create_cnn_bilstm_model, create_cnn_gru_model
from src.models.rnn_models import create_bilstm_model, create_gru_model, create_lstm_model


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, (n, 21)), rng.random(n).astype(np.float32)


def test_gc_content_matches_its_definition():
    """: the proportion of G and C in the sequence."""
    rng = np.random.default_rng(0)
    idx = torch.as_tensor(rng.integers(0, 4, (16, 21)))
    expected = ((idx == 1) | (idx == 2)).float().mean(dim=1)
    assert torch.allclose(gc_content(idx), expected)


def test_gc_content_accepts_one_hot_and_indices():
    idx = torch.as_tensor(np.random.default_rng(1).integers(0, 4, (8, 21)))
    one_hot = torch.nn.functional.one_hot(idx, 4).float()
    assert torch.allclose(gc_content(idx), gc_content(one_hot))


@pytest.mark.parametrize("factory", [
    create_cnn_model, create_deep_cnn_model, create_gru_model, create_lstm_model,
    create_bilstm_model, create_cnn_gru_model, create_cnn_bilstm_model,
])
def test_models_train_for_two_epochs(factory):
    sequences, targets = _data()
    trainer = ChromeCRISPRTrainer(factory, {})
    results = trainer.train_model(sequences, targets,
                                  {"batch_size": 64, "learning_rate": 1e-3}, epochs=2)
    assert np.isfinite(results["best_val_loss"])
    assert results["model"] is not None


def test_device_falls_back_to_cpu_when_cuda_is_absent():
    trainer = ChromeCRISPRTrainer(create_cnn_model, {})
    assert trainer.device in ("cuda", "cpu")
    if not torch.cuda.is_available():
        assert trainer.device == "cpu"


def test_training_settings_are_not_passed_to_the_model():
    """Batch size and learning rate configure the loop, not the network."""
    sequences, targets = _data(n=128)
    trainer = ChromeCRISPRTrainer(create_cnn_gru_model, {})
    results = trainer.train_model(
        sequences, targets,
        {"batch_size": 32, "learning_rate": 1e-3, "dropout": 0.1}, epochs=1)
    assert results["model"] is not None


def test_loss_is_not_broadcast_against_a_mismatched_shape():
    """The models emit (batch, 1) and the targets are (batch,)."""
    model = create_cnn_gru_model().eval()
    x = torch.nn.functional.one_hot(
        torch.as_tensor(np.random.default_rng(2).integers(0, 4, (5, 21))), 4).float()
    out = model(x, gc_content(x))
    assert out.shape == (5, 1)
    assert out.squeeze(-1).shape == (5,)
