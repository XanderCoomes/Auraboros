# Simple DQN for 8×8 Snake

A stripped-down playground version of [`complex_snake`](../complex_snake/README.md). It uses the same game, safety mask, observation, rewards and CNN trunk, trained with **plain DQN**:

| | complex_snake | simple_snake |
|---|---|---|
| Safety mask, 4-plane time-to-free observation, rewards | ✓ | ✓ (same) |
| Conv trunk (4 × 3×3 conv, 64 ch → 512 FC) | ✓ | ✓ (same) |
| Q head | dueling (value + advantage) | single linear head |
| Target | Double DQN, 5-step return | `r + γ max Q_target(s′, a′)`, 1 step |
| Replay | prioritized, with 8-fold symmetry augmentation | uniform, no augmentation |
| Demonstrations / pre-training (DQfD) | ✓ | ✗ |
| Mid-game starts | ✓ | ✗ (every episode starts from the opening) |
| ε schedule | 0.1 → 0.01 (demos make high ε unnecessary) | 1.0 → 0.01 over the first 30% |

## Files

| File | What it is |
|---|---|
| `snake_env.py` | Game, safety mask (bitboard flood fill), observation, rendering |
| `model.py` | `QNet`, the conv Q-network |
| `replay.py` | Uniform ring-buffer replay |
| `train.py` | DQN trainer (`select_actions`, `compute_td_loss`, `sync_target_network`), logging, checkpoints |
| `evaluate.py` | Greedy evaluation on fixed seeds (same seeds as complex_snake) |
| `watch.py` | Watch one game in the terminal, optionally with Q-values |

All commands are run from inside this folder (`cd simple_snake`). To watch and score a random (but safety-masked) snake as a baseline:

```bash
python watch.py --random
python evaluate.py --random --games 200
```

## Train, test, view

```bash
# quick run (~3 min on Apple-silicon MPS): enough to see learning happen
python train.py --run-name quick --total-steps 600000 --eval-every 100000 --eval-games 100

# default run: 2M env steps (~10 min on MPS)
python train.py --run-name main

python evaluate.py runs/main/best.pt --games 500    # test: greedy win rate + steps per win
python watch.py runs/main/best.pt --show-q          # view: one game, with Q(s, ·) each step
```

### Continue training a model (warm start)

```bash
python train.py --run-name main2 --init-from runs/main/best.pt --eps-start 0.05 --total-steps 1000000
```

`--init-from` loads only the network weights (into both the online and target nets). The optimizer, replay buffer, step counter and ε schedule start fresh, so:

- Use a **new `--run-name`**. Reusing the source run's name is refused, because it would overwrite that run's `best.pt`, `last.pt` and `log.csv`.
- Set a **low `--eps-start`** (e.g. `0.05`). The default of 1.0 would play randomly for a long time and refill the buffer with poor data.
- `--n`, `--width` and `--hidden` must match the checkpoint. Other flags (lr, γ, rewards, …) can change.
- The new `log.csv` counts env steps from 0 again.

`runs/<name>/log.csv` has one row per evaluation. `eval_win_rate` and `eval_steps_mean` are the numbers that matter. `best.pt` is chosen by win rate, then by fewest steps.

### Reference numbers (8×8, eval seeds 10000+)

| Policy | Win rate | Steps per win |
|---|---|---|
| Random among safe moves | 0% | — (mean final length ≈ 10, all hunger timeouts) |
| Greedy among safe moves | ~9% | ~730 |
| Hamiltonian + shortcuts | 100% | ~640 |
| complex_snake, full recipe (2M steps) | ~92% final, 100% best | ~475 |
| complex_snake without demos (2M steps) | ~75% | ~575 |
| simple_snake reference solution, quick run (600k steps) | 14% | ~790 (mean final length 21 → 56 over the run) |

A quick run should look roughly like that last row. Expect the full run to land below the "without demos" row: it also lacks mid-game starts, n-step returns, PER, augmentation, Double DQN and dueling. Measuring by how much is the point of this folder.

## Experiments to try once it works

Each one is a single flag or a few lines of code. Compare `log.csv` files.

1. **Safety mask:** `--mask-mode legal` (only immediately fatal moves blocked) or `--mask-mode none`. With `none`, does the agent ever learn to avoid walls?
2. **Target network:** `--target-update 1` vs `20000`.
3. **Discount:** `--gamma 0.95` / `0.995`. How does it change steps per win and the hunger-timeout rate?
4. **Step penalty:** `--r-step -0.01`.
5. **Double DQN** (≈3 lines in `compute_td_loss`): choose `a* = argmax_{a′ allowed} Q_online(s′, a′)` and score it with `Q_target(s′, a*)`.
6. **Observation:** in `obs()`, write a 0/1 body plane instead of time-to-free, or leave the hunger plane at 0.
7. **Bring back one bell or whistle** from complex_snake (n-step returns are the easiest; see `NStep` in `complex_snake/replay.py`) and measure what it buys on its own.
