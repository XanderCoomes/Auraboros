# Snake deep Q-learning

An agent that solves Snake via DQN and Pygame.

## Files

- **[model.py](model.py):** editable defaults, state encoding, network, rewards,
  replay memory, agent, weight updates and shared evaluation helpers.
- **[train.py](train.py):** training CLI, progress messages, checkpoints and CSV logs.
- **[eval.py](eval.py):** load frozen weights, score games or watch one in Pygame.

The comments follow the approximate agent's section layout. Settings and reward
defaults are together near the top of `model.py`. CLI flags override them for a
new run. Evaluation uses the settings stored with the selected model.

## Running

Use the project Python environment with `torch` and `numpy` installed. Visual
mode also needs `pygame`.

From inside `deepq/`, with that environment active:

```bash
# Train using the defaults in model.py. Each run directory must be new.
python3 train.py --run-dir runs/first

# Score the latest saved model over 100 games.
python3 eval.py --games 100

# Watch one complete game with frozen weights.
python3 eval.py --visual

# Select a particular model instead of the latest one.
python3 eval.py --model runs/first --games 100

# Watch game 6 from that seeded batch; replay the first five without animation.
python3 eval.py --model runs/first --seed 1000 --start-game 6 --visual
```

## Network and input

On an 8×8 board, the network is:

```text
203 inputs -> Linear(512) -> ReLU -> Linear(96) -> ReLU -> 3 Q-values
```

The final values estimate **straight**, **left** and
**right**.

| Input slots | Information                                                                |
| ----------- | -------------------------------------------------------------------------- |
| `0:64`      | Head-position grid: 1 at the head, 0 elsewhere                             |
| `64:128`    | Apple-position grid: 1 at the apple, 0 elsewhere                           |
| `128:192`   | Ordered-body grid: tail `1/64`, next segment `2/64`, ..., head `length/64` |
| `192:196`   | One-hot heading: up, right, down, left                                     |
| `196`       | Consecutive moves without food divided by the timeout threshold            |
| `197:199`   | Straight: wall collision, body collision                                   |
| `199:201`   | Left: wall collision, body collision                                       |
| `201:203`   | Right: wall collision, body collision                                      |

## Learning defaults

| Setting                 | Default | Meaning                                                        |
| ----------------------- | ------: | -------------------------------------------------------------- |
| `--size`                |       8 | Board width and height                                         |
| `--episodes`            |   30000 | Training games                                                 |
| `--hidden-dim`          |     128 | Width of each of the two hidden layers                         |
| `--alpha`               |  0.0003 | Adam learning rate                                             |
| `--gamma`               |    0.99 | Discount factor                                                |
| `--epsilon`             |     1.0 | Initial random-action probability                              |
| `--epsilon-end`         |    0.05 | Final random-action probability                                |
| `--epsilon-decay-frac`  |     0.6 | Fraction of training used for linear epsilon decay             |
| `--batch-size`          |      64 | Past moves sampled per optimizer update                        |
| `--replay-capacity`     |   50000 | Maximum stored moves                                           |
| `--learning-starts`     |    1000 | Training moves to collect before learning                      |
| `--train-every`         |       4 | Training moves between optimizer updates                       |
| `--target-update-every` |    1000 | Optimizer updates between target-network copies                |
| `--gradient-clip`       |      10 | Maximum gradient norm                                          |
| `--reward-scale`        |    0.01 | Uniform reward multiplier for learning targets                 |
| `--eval-every`          |    2000 | Episodes between evaluation checks; 0 disables periodic checks |
| `--eval-games`          |     100 | Games in each periodic/final check                             |
| `--seed`                |       0 | Training seed                                                  |
| `--eval-seed`           |    1000 | Independent evaluation seed                                    |
| `--print-every`         |     100 | Episodes between training progress messages                    |
| `--device`              |    auto | CUDA, then MPS, then CPU, according to availability            |
| `--threads`             |       1 | CPU threads used by PyTorch                                    |

Each training move is stored as `(state, action, reward, next_state, done)`.
After warmup, the agent samples random batches and calculates:

```text
prediction = policy_network(state)[chosen_action]
target = scaled_reward                                  if terminal
         scaled_reward + gamma * max(target_network(next_state)) otherwise
```

The agent uses Huber loss (`smooth_l1_loss`), backpropagation, gradient clipping
and Adam to update the policy. The target network stays fixed between
scheduled copies. `time_steps` counts training moves; `update_steps` counts
optimizer updates.

## Reward defaults

| Setting               | Default | Meaning                                       |
| --------------------- | ------: | --------------------------------------------- |
| `--living-reward`     |       0 | Every surviving move, including eating        |
| `--apple-reward`      |     100 | Added when eating                             |
| `--wall-death-reward` |   -6350 | Terminal wall collision                       |
| `--body-death-reward` |   -6350 | Terminal self-collision                       |
| `--timeout-reward`    |   -6350 | Terminal no-food timeout                      |
| `--distance-shaping`  |       0 | Optional potential-based distance shaping     |
| `--timeout-factor`    |       2 | No-food threshold as a multiple of board area |

Raw rewards are multiplied by `--reward-scale` only when forming learning
targets. At 0.01, a raw apple bonus of 100 becomes 1, and a death reward of -6350
becomes -63.5. Every reward is scaled by the same factor. Scores, CSV returns and
the viewer's rewards stay in raw units; displayed Q-values use scaled reward units.

These defaults start from zero living reward, separately from the approximate
agent's experiments. Collision/timeout rewards replace living and food rewards.
A win earns the final apple plus living reward, with no extra win bonus.

Optional distance shaping uses `coefficient * (gamma * Phi(next) - Phi(current))`,
where `Phi` is negative normalized Manhattan distance and is zero at termination.

On 8×8, winning means eating 61 apples to fill all 64 cells. With timeout factor
2, the game ends on the 129th consecutive move without food, where eating an apple resets the
counter.

## Visuals and output

Training accepts `--visual` or `--visual-eval`. Use `--visual-every 100` to animate
every 100th training game.

Each run contains:

- `model.pt`: network weights, architecture, encoding version, settings and run metadata.
- `config.json`: a readable copy of those settings and the last evaluation summary.
- `log.csv`: completed training episodes, scores, total steps, raw returns, endings and loss.
- `eval.csv`: per-game evaluation results at each checkpoint.

Evaluation prints mean apples, best score, win rate, average steps for all games,
and mean/min/max steps for wins.

This uses the standard DQN implementation as described in PyTorch's DQN tutorial
[PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
