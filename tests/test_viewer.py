import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q_dqn import DQNAgent
from state_encoders import BoardEncoder, BoardSnakeEnv
from view_dqn import ReplayViewer, record_game


def test_replay_controls_and_output_contributions():
    env = BoardSnakeEnv(size=8, seed=0)
    encoder = BoardEncoder()
    agent = DQNAgent({"name": "rescnn8", "input_shape": encoder.shape(8), "context_dim": 7})
    metadata = {"encoder": encoder.config(), "board_size": 8, "rewards": env.rewards.weights,
                "initial_length": 3, "timeout_factor": 2}
    frames = record_game(agent, metadata, max_steps=10)
    weights = agent.online.output.weight.detach().numpy()
    bias = agent.online.output.bias.detach().numpy()
    for frame in frames:
        contributions = weights * np.asarray(frame["activations"]["fc2"])[None, :]
        np.testing.assert_allclose(contributions.sum(axis=1) + bias, frame["activations"]["q"], atol=1e-6)
    viewer = ReplayViewer(frames, weights, bias)
    viewer.seek(len(frames) - 1)
    assert viewer.index == len(frames) - 1
    assert int(viewer.slider.val) == viewer.index
    assert frames[-1]["reason"]
    viewer.toggle_play()
    assert viewer.playing and viewer.index == 0
    viewer.tick()
    assert viewer.index == 1
    viewer.seek(len(frames) - 1)
    viewer.tick()
    assert not viewer.playing
    viewer.figure.canvas.draw()
    plt.close(viewer.figure)


def test_ending_shows_wall_hit_and_preserves_last_decision():
    class StraightAgent:
        calls = 0

        def inspect(self, observation):
            self.calls += 1
            return {"fc1": [self.calls], "fc2": [self.calls], "q": [self.calls, 0, 0]}

    agent = StraightAgent()
    metadata = {"encoder": BoardEncoder().config(), "board_size": 8, "rewards": {},
                "initial_length": 3, "timeout_factor": 2}
    frames = record_game(agent, metadata)
    ending = frames[-1]
    assert ending["reason"] == "wall collision"
    assert ending["collision"] == (4, 8)
    assert agent.calls == ending["steps"]
    assert ending["activations"] == frames[-2]["activations"]
    assert ending["action"] == 0

    viewer = ReplayViewer(frames, np.array([[1], [0], [0]]), np.zeros(3))
    viewer.seek(len(frames) - 1)
    assert viewer.collision_marker.get_visible()
    assert list(viewer.collision_marker.get_xdata()) == [7.5]
    assert "last move" in viewer.q_title.get_text()
    viewer.seek(0)
    assert not viewer.collision_marker.get_visible()
    plt.close(viewer.figure)
