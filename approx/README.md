# Approximate Q-learning

Approximate Q-learning for Snake, with a Pygame viewer for evaluation. The
implementation lives in **[snake.py](snake.py)**.

Edit the **Hyperparameters & Rewards** section near the top of `snake.py` to
change the defaults.

- `ApproxQAgent.getFeatures`: the 11 features and length interactions.
- `ApproxQAgent.getQValue` / `chooseUsingPolicy`: Q-values and epsilon-greedy selection.
- `ApproxQAgent.update`: TD target and weight update.
- `ApproxSnakeEnv.computeReward`: food, living, collision, timeout and distance rewards.
- Training/evaluation CLI and save/load logic are in the same file.

CLI flags override the editable defaults for a new run. Evaluation uses the
settings stored with the saved weights and, by default, loads the most recently
saved model.

## Running the agent

From the project root:

```bash
python3 approx/snake.py train
python3 approx/snake.py eval
python3 approx/snake.py eval --visual
```

From inside `approx/`:

```bash
python3 snake.py train
python3 snake.py eval
python3 snake.py eval --visual
```

Use your Python environment with the dependencies installed. A 100-game
evaluation runs sequentially:

```bash
python3 snake.py eval --games 100
```

Without `--model`, evaluation selects `runs/*/model.json` with the most recent
modification time (regardless of folder name) and prints its path. Pass `--model` to choose a particular run.

Train with `--visual` to animate training and evaluation, or `--visual-eval` to
animate only evaluation games. `--visual-every 100` samples every 100th training
episode; `--fps 30` sets playback speed.

Controls:

- **Space:** pause/resume.
- **Up/Down:** change speed.
- **Escape/window close:** stop.

Training saves current weights when interrupted. Its CSV contains completed
episodes.

## Defaults and tuning

| CLI setting            | Default | Meaning                                                        |
| ---------------------- | ------: | -------------------------------------------------------------- |
| `--size`               |       8 | Board width/height                                             |
| `--episodes`           |   30000 | Training games                                                 |
| `--alpha`              |   0.001 | Learning rate                                                  |
| `--gamma`              |     0.9 | Discount factor                                                |
| `--epsilon`            |     0.2 | Initial random-action probability                              |
| `--epsilon-end`        |    0.01 | Final random-action probability                                |
| `--epsilon-decay-frac` |     0.6 | Linear decay over the first 60% of training                    |
| `--living-reward`      |       0 | Every surviving move, including eating                         |
| `--apple-reward`       |     100 | Additional reward when eating                                  |
| `--wall-death-reward`  |   -6350 | Terminal wall collision reward                                 |
| `--body-death-reward`  |   -6350 | Terminal self-collision reward                                 |
| `--timeout-reward`     |   -6350 | Terminal starvation/timeout reward                             |
| `--distance-shaping`   |       0 | Optional potential-based distance shaping coefficient          |
| `--timeout-factor`     |       3 | Timeout after more than factor × board area steps without food |
| `--eval-every`         |    2000 | Training episodes between checks; 0 disables periodic checks   |
| `--eval-games`         |     100 | Games in each periodic/final check                             |
| `--seed`               |       0 | Training seed                                                  |
| `--eval-seed`          |    1000 | Independent, fixed evaluation seed                             |

For constant exploration at 0.2, pass `--epsilon 0.2 --epsilon-end 0.2`.
Evaluation always uses epsilon zero and performs no weight updates.

**Win** means every cell is occupied by the snake. On an 8×8 board, the initial
length is 3, so winning requires eating **61 apples**.

**Timeout** means too many moves without eating. The counter resets on each
apple. With `TIMEOUT_FACTOR = 3`, an 8×8 game ends on the **193rd consecutive move
without food** (`steps_since_apple > 192`). Change `TIMEOUT_FACTOR` at the top of
`snake.py` or pass `--timeout-factor` when training.

## Features and learning

The model is `Q(s, a) = weights[a] · features(s, a)`, with three independent weight
vectors for **straight**, **left** and **right**. Here are the 11 features:

1. Bias, set to 1.
2. Manhattan distance from the head's position after the candidate move to the
   current apple, normalized by the board's maximum possible Manhattan distance,
   `2 * (size - 1)`.
3. Would eat an apple (bool).
4. Would hit a wall (bool).
5. Would hit the body (bool).
6. Post-move snake length divided by board area.
7. Reachable space from the head divided by board area.
8. Tail reachable (bool).
9. Length fraction × apple distance.
10. Length fraction × reachable space.
11. Length fraction × tail reachable.
