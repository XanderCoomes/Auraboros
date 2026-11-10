# Auraboros
Reinforcement Learning for Snake

Deep Q-learning with board encoders, residual CNNs, and an activation replay
viewer is documented in [DQN.md](DQN.md). Start with
`python train.py --agent dqn --model rescnn8` after installing
`requirements-dqn.txt`.

## Setup
```
pip install numpy matplotlib pillow
```

## Train
```
python train.py --size 8 --episodes 20000
```
Useful options: `--alpha`, `--gamma`, `--eps-start`, `--eps-end`, `--eps-decay-frac`, `--seed`, `--features`.

## State features
The state is a tuple of plug-in features defined in `features.py`. To add one, write a function of the
env that returns an int in `[0, n_values)` and register it:
```python
@feature("apple_adjacent", 2, labels=["no", "yes"])
def apple_adjacent(env):
    """Apple in a cell next to the head?"""
    return int(env.apple is not None and env._dist_to_apple(env.snake[0]) == 1)  # apple is None once the board is full
```
Then choose the features (in order) at train time; the state-space size and Q-table shape follow automatically:
```
python train.py --list-features
python train.py --features danger_straight,danger_left,danger_right,apple_row_sign,apple_col_sign,heading,apple_adjacent
```
The feature list is saved inside `q_table.pkl` (and `q_table.json`), so `plot.py` rebuilds the same state automatically.

## Reward terms
The reward per step is a weighted sum of terms defined in `rewards.py`
(defaults: `apple +100`, `death -6350`, `step -0.01`, `closer +/-0.1`). To add a term:
```python
@reward("length_bonus", 1.0)
def length_bonus(env, ev):
    """+len(snake) whenever an apple is eaten."""
    return len(env.snake) if ev["outcome"] in ("apple", "win") else 0
```
`ev["outcome"]` is one of `move`, `apple`, `win`, `collision`, `timeout`; `ev` also has `prev_dist`/`dist` to the apple
and `prev_head`/`head`. Pick terms and weights (omit `=w` for the default weight, or use `all`):
```
python train.py --list-rewards
python train.py --rewards apple=100,death=-1000,step,closer,length_bonus=5
```
The weights are saved in `q_table.json`; `plot.py` replays with them and logs each step's reward breakdown in `game.json`.

Outputs (in the current folder):
- `log.csv` - per-episode epsilon, score, steps, return
- `q_table.pkl` - the learned Q-table
- `q_table.json` - the training config used

## Plot
```
python plot.py --log log.csv --q q_table.pkl --size 8
```
- `--size` must match the board size used in training.
- Omit `--q` to plot only the learning curves (no GIF).
- `--window` sets the moving-average window (default 200).

## Check the results
Each plot run creates a new folder `test_result/train_<timestamp>/` containing:
- `curves.png` - learning curves (score and episode steps vs. episode)
- `game.gif` - one greedy game played by the trained agent, with state/Q-values/reward overlay
- `game.json` - method, state definition, training config, greedy evaluation over 100 games, per-step log
- copies of `log.csv`, `q_table.pkl`, `q_table.json` (the next training run overwrites the originals)

Open `curves.png` and `game.gif` with any image viewer or browser.
