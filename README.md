# Auraboros
Reinforcement Learning for Snake

## Setup
```
pip install numpy matplotlib pillow
```

## Train
```
python train.py --size 8 --episodes 20000
```
Useful options: `--alpha`, `--gamma`, `--eps-start`, `--eps-end`, `--eps-decay-frac`, `--seed`.

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
