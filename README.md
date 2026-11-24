# Auraboros
Reinforcement Learning for Snake

Deep Q-learning with board encoders, residual CNNs, and an activation replay
viewer is documented in [DQN.md](DQN.md). Start with
`python train.py --model rescnn8` after installing
`requirements-dqn.txt`.

## Setup
```
pip install -r requirements-dqn.txt
```

## Train
```
python train.py --model rescnn8 --encoder entities_tail --size 8 --episodes 20000
```
Useful options: `--gamma`, `--eps-start`, `--eps-end`, `--eps-decay-frac`, `--seed`, `--rewards`.
The model, encoder, and replay options are listed in [DQN.md](DQN.md).

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
The weights are saved in `model.json` and in the checkpoint; `view_dqn.py` replays with them.

Outputs (a new folder under `runs/` for each run):
- `log.csv` - per-episode epsilon, score, steps, return, loss
- `model.pt` - the checkpoint
- `model.json` - the training config used and the evaluation history

## Plot
```
python plot.py --log runs/<run>/log.csv
```
- `--window` sets the moving-average window (default 200).
- The title is read from the `model.json` beside the log.

## Check the results
Each plot run creates a new folder `test_result/train_<timestamp>_test_<timestamp>/` containing:
- `curves.png` - learning curves (score and episode steps vs. episode)
- a copy of `log.csv`

To watch a trained agent play, run `python view_dqn.py --checkpoint runs/<run>/model.pt`
(see [DQN.md](DQN.md)).
