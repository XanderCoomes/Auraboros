
import torch

from dqn.models.hybrid import DQNHybrid


def test_hybrid_model():
    batch_size = 16

    model = DQNHybrid()

    board = torch.randn(batch_size, 4, 8, 8)
    heading = torch.randn(batch_size, 4)
    features = torch.randn(batch_size, 3, 11)

    q_values = model(board, heading, features)

    assert q_values.shape == (batch_size, 3)
    assert torch.isfinite(q_values).all()

    # Verify gradients can flow through the network.
    loss = q_values.mean()
    loss.backward()

    assert model.cnn[0].weight.grad is not None
    assert model.feature_encoder[0].weight.grad is not None

    print("Output shape:", q_values.shape)
    print("Hybrid model test passed!")


if __name__ == "__main__":
    test_hybrid_model()
