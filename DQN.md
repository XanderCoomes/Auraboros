# Deep Q-learning extension

Install into a Python 3.9+ environment and run from this repository.

```bash
python -m pip install -r requirements-dqn.txt
python train.py --agent dqn --model rescnn4 --encoder entities_tail --episodes 20000
python train.py --agent dqn --model rescnn8 --encoder entities_tail --episodes 20000
```

Each run prints a new checkpoint path under `runs/`. To choose a path, pass
`--out runs/my_experiment/model.pt`. Training writes `log.csv` and `model.json`
beside the checkpoint. Existing files are protected against overwrite.

## Model

The default board input is `(2, 8, 8, 4)` per observation. The leading axis stores
absent and present values. The four channels are body, head, apple, and tail.
The tail channel is a chosen default to fill the requested four-channel input.
The CNN rearranges this to eight input channels for PyTorch Conv2d.

For an 8×8 board the default convolutional path is

```text
2 × 8 × 8 × 4 observation
    -> 8 input channels × 8 × 8
    -> residual block, 8 channels × 8 × 8
    -> max pooling, 8 channels × 4 × 4
    -> residual block, 16 channels × 4 × 4
    -> max pooling, 16 channels × 2 × 2
    -> residual block, 32 channels × 2 × 2
    -> convolution, 4 or 8 channels × 2 × 2
    -> global average pooling and flatten, 4 or 8 learned values
    -> concatenate 7 game-context values
    -> fully connected 64, ReLU
    -> fully connected 32, ReLU
    -> fully connected 3, raw Q-values
```

`rescnn4` and `rescnn8` choose the number of learned values before the fully
connected layers. Spatial convolutions use 3×3 kernels, stride 1, and padding 1.
Each residual block has two convolutions and a skip connection. A 1×1 skip
projection aligns channel counts. Use `--base-channels` and `--hidden-dims 64 32`
to change widths. Input size comes from the selected encoder and `--size`.

## Encoders

| `--encoder` | Spatial channels | Default encoding |
| --- | --- | --- |
| `entities_tail` | Body excluding head, head, apple, tail | `(2,H,W,4)` |
| `directional` | Up, right, down, left, apple | `(2,H,W,5)` |
| `entities` | Body excluding head, head, apple | `(2,H,W,3)` |
| `ordered` | Body order including head, head, apple | `(H,W,3)` |

For the directional version, every body segment points to the adjacent segment
toward the head. The head's direction is its current heading. For the ordered
version, count from tail = 1 to head = snake length, divided by board area.
Empty cells are zero. The fixed divisor preserves length information.

All encoders also supply heading as four one-hot values, normalized head row
and column, and the fraction of the no-food timeout already used. This identifies
the head in the directional representation and exposes the game's timeout state.
The entity representations omit full body order, so they can alias distinct
snake configurations. The directional and ordered variants preserve the path.

`--encoding binary` uses one value per channel, removing the absent/present axis.
For ordered observations, this flag preserves the normalized numeric values.
`--encoding auto` chooses one-hot for binary masks and numeric values for ordered
observations. Explicit one-hot encoding is rejected for ordered values.

## Training

The agent uses epsilon-greedy actions, replay minibatches, a separate target
network, Huber loss, Adam, and gradient clipping. Collision, timeout, and win
all terminate the target according to the existing game's semantics.

Defaults are batch size 64, replay capacity 10,000, warmup 1,000 transitions,
one optimizer update every four moves, and a target copy every 250 updates.
Rewards are multiplied by 0.01 inside the learner to accommodate the existing
large death penalty. Episode returns and game scores retain the original units.
Use `--reward-scale 1` to train in original reward units.

```bash
python train.py --agent dqn --model rescnn8 --encoder directional --size 8 \
  --learning-rate 0.001 --batch-size 64 --warmup 1000 --target-every 250
```

Evaluation uses a separate seeded environment. Checkpoints include model and
encoder configuration, weights, target weights, optimizer state, counters, and
environment metadata. Loading supports inference and subsequent learning with
a fresh replay buffer. Exact training resumption, including replay and all RNG
states, is outside the checkpoint format.

## Watch a game

```bash
python view_dqn.py --checkpoint runs/my_experiment/model.pt
python view_dqn.py --checkpoint runs/my_experiment/model.pt --gif game.gif
python view_dqn.py --checkpoint runs/my_experiment/model.pt --snapshot ending.png
python plot.py --log runs/my_experiment/log.csv
```

The viewer provides play/pause, stepping, scrubbing, and an Ending button. It
shows the two hidden-layer activations, the three raw Q-values, and each last
hidden neuron's contribution to each action (`activation × output weight`).
The output biases are displayed as well. A positive contribution raises that
action's Q-value and a negative contribution lowers it. The final frame retains
the game's ending reason. GIF and snapshot export work without a display using
`MPLBACKEND=Agg`.

## Replace a component

Add an encoder function using `@register_encoder` in `state_encoders.py`. It
returns a fresh `(H,W,C)` array. Register the channel names and whether values
are binary. `BoardEncoder` supplies context and the optional one-hot expansion.

Add a model constructor to `MODELS` in `models.py`. The constructor accepts
`input_shape`, `context_dim`, `base_channels`, and `hidden_dims`. Its module
accepts batched `board, context` tensors and returns `(batch,3)` Q-values. For
the viewer, support `return_activations=True` with `fc1`, `fc2`, and `q` tensors
and expose the final linear layer as `output`.

`BoardSnakeEnv` overrides only observation construction. `train.py --agent dqn`
dispatches to `train_dqn.py`, which reuses `run_episode` and `evaluate_details`.
The original tabular command remains `python train.py --episodes 20000`.

## Verify

```bash
python -m pip install pytest
python -m pytest tests/test_dqn.py
```
