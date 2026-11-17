import json
from pathlib import Path

import torch

from dqn.encoders.engineered import EngineeredEncoder
from dqn.encoders.board import BoardEncoder, BoardEncoderWithTime
from dqn.models.mlp import DQNMLP
from dqn.models.cnn import DQNCNN
from dqn.models.hybrid import DQNHybrid


def load_trained_model(model_type, model_path, config_path=None):

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if config_path is None:
        config_path = model_path.with_suffix(".json")

    config_path = Path(config_path)

    config = {}

    if config_path.exists():
        with open(config_path, "r") as file:
            config = json.load(file)

    if model_type == "mlp":

        encoder = EngineeredEncoder()

        model = DQNMLP(
            input_dim=encoder.input_shape[0]
        )

    elif model_type == "cnn":

        encoder = BoardEncoder()

        model = DQNCNN()

    elif model_type == "hybrid":

        encoder = BoardEncoderWithTime()

        model = DQNHybrid()

    else:
        raise ValueError(
            "model_type must be 'mlp', 'cnn', or 'hybrid'"
        )

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, encoder, device, config
