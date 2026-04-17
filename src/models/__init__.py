"""Public model factories exported for repo-local use examples."""

from .cnn_model import CNNModel, DeepCNNModel, create_cnn_model, create_deep_cnn_model
from .hybrid_models import (
    CNNBiLSTMModel,
    CNNGRUModel,
    CNNLSTMModel,
    create_cnn_bilstm_model,
    create_cnn_gru_model,
    create_cnn_lstm_model,
)
from .rnn_models import BiLSTMModel, GRUModel, LSTMModel, create_bilstm_model, create_gru_model, create_lstm_model

MODEL_FACTORIES = {
    "cnn": create_cnn_model,
    "deep_cnn": create_deep_cnn_model,
    "gru": create_gru_model,
    "lstm": create_lstm_model,
    "bilstm": create_bilstm_model,
    "cnn_gru_gc": create_cnn_gru_model,
    "cnn_lstm_gc": create_cnn_lstm_model,
    "cnn_bilstm_gc": create_cnn_bilstm_model,
}


def create_model(name: str, **kwargs):
    """Create a known model factory by normalized public name."""
    normalized = name.strip().lower()
    if normalized not in MODEL_FACTORIES:
        available = ", ".join(sorted(MODEL_FACTORIES))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_FACTORIES[normalized](**kwargs)


__all__ = [
    "BiLSTMModel",
    "CNNBiLSTMModel",
    "CNNGRUModel",
    "CNNLSTMModel",
    "CNNModel",
    "DeepCNNModel",
    "GRUModel",
    "LSTMModel",
    "MODEL_FACTORIES",
    "create_bilstm_model",
    "create_cnn_bilstm_model",
    "create_cnn_gru_model",
    "create_cnn_lstm_model",
    "create_cnn_model",
    "create_deep_cnn_model",
    "create_gru_model",
    "create_lstm_model",
    "create_model",
]
