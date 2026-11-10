
import numpy as np

from dqn.model_loader import load_trained_model
from dqn.visualize import record_episode

from dqn.visualize import save_gameplay_gif

from dqn.visualize import save_state_gif

def test_visualization():

    # Load trained CNN.
    model, encoder, device, config = load_trained_model(
        model_type="cnn",
        model_path="dqn/cnn_snake_dqn.pth"
    )

    # Record one greedy episode.
    frames = record_episode(
        model=model,
        encoder=encoder,
        device=device,
        size=config.get("size", 8),
        seed=21,    #42
        )

    print("\nVisualization test results")
    print("--------------------------")

    print("Total frames:", len(frames))
    print("Final score:", frames[-1]["score"])
    print("Final steps:", frames[-1]["steps"])
    print("Total reward:", frames[-1]["total_reward"])
    print("End reason:", frames[-1]["end_reason"])

    # Inspect first frame.
    first_frame = frames[0]

    print("\nFirst frame")
    print("Board shape:", first_frame["board"].shape)
    print("Heading:", first_frame["heading"])
    print("Action:", first_frame["action"])
    print("Q-values:", first_frame["q_values"])

    # Validate data.
    assert len(frames) > 0
    assert first_frame["board"].shape == (8, 8)
    assert first_frame["q_values"].shape == (3,)
    assert first_frame["action"] in [0, 1, 2]

    # Validate terminal information.
    assert frames[-1]["done"] is True
    assert frames[-1]["end_reason"] in [
        "collision",
        "timeout",
        "win",
    ]

    save_gameplay_gif(
        frames=frames,
        output_path="test_result/cnn2/gameplay.gif",
        fps=6
    )

    save_state_gif(
        frames=frames,
        model_type="cnn",
        output_path="test_result/cnn2/state_representation.gif",
        fps=6
    )

    print("\nAll visualization tests passed!")


if __name__ == "__main__":
    test_visualization()
