
import json
from pathlib import Path

import torch

from dqn.encoders.engineered import EngineeredEncoder
from dqn.encoders.board import BoardEncoder

from dqn.models.mlp import DQNMLP
from dqn.models.cnn import DQNCNN


def load_trained_model(model_type, model_path, config_path=None):

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load training configuration if available.
    if config_path is None:
        config_path = model_path.with_suffix(".json")

    config_path = Path(config_path)

    config = {}

    if config_path.exists():
        with open(config_path, "r") as file:
            config = json.load(file)

    # Reconstruct architecture and encoder.
    if model_type == "mlp":

        encoder = EngineeredEncoder()

        model = DQNMLP(
            input_dim=encoder.input_shape[0]
        )

    elif model_type == "cnn":

        encoder = BoardEncoder()

        model = DQNCNN()

    else:
        raise ValueError(
            "model_type must be 'mlp' or 'cnn'"
        )

    # Load trained parameters.
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, encoder, device, config
