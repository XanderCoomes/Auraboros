
from dqn.model_loader import load_trained_model


def test_model_loader(model_type, model_path):

    print(f"\nTesting {model_type.upper()} model")

    model, encoder, device, config = load_trained_model(
        model_type=model_type,
        model_path=model_path
    )

    print("Model:", model.__class__.__name__)
    print("Encoder:", encoder.__class__.__name__)
    print("Device:", device)
    print("Configuration:", config)

    print(f"{model_type.upper()} model loaded successfully!")


if __name__ == "__main__":

    test_model_loader(
        "cnn",
        "dqn/cnn_snake_dqn.pth"
    )

    # Uncomment when testing your trained MLP:
    # test_model_loader(
    #     "mlp",
    #     "dqn/mlp_snake_dqn.pth"
    # )
