# Deep-Q (basic): MLP: test results

Greedy play on 1000 fixed test games (8x8 board, env seeds 20000–20999). 95% confidence intervals: Wilson for the win rate, percentile bootstrap for means.

| Metric | Value | 95% CI |
|---|---|---|
| Win rate (board filled, 61 apples) | 0.0% | [0.0%, 0.4%] |
| Mean apples per game | 15.7 | [15.3, 16.0] |
| Best game (apples) | 34 | |
| Mean steps per apple | 6.34 | [6.28, 6.39] |
| Mean steps to win (won games) | – | |

Losses by cause and snake length (bucket 0: <10% of the board, 1: 10–25%, 2: 25–50%, 3: >50%):

| Cause / length bucket | Games |
|---|---|
| collision/bucket0 | 5 |
| collision/bucket1 | 284 |
| collision/bucket2 | 695 |
| collision/bucket3 | 16 |

Example game (`game_seed_42.gif`, env seed 42): 7 apples in 53 steps, ended by collision.
