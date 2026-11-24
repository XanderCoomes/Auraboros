# Deep-Q (medium): CNN + per-action features: test results

Greedy play on 1000 fixed test games (8x8 board, env seeds 20000–20999). 95% confidence intervals: Wilson for the win rate, percentile bootstrap for means.

| Metric | Value | 95% CI |
|---|---|---|
| Win rate (board filled, 61 apples) | 4.9% | [3.7%, 6.4%] |
| Mean apples per game | 39.2 | [38.4, 40.1] |
| Best game (apples) | 61 | |
| Mean steps per apple | 10.60 | [10.42, 10.78] |
| Mean steps to win (won games) | 799 | |

Losses by cause and snake length (bucket 0: <10% of the board, 1: 10–25%, 2: 25–50%, 3: >50%):

| Cause / length bucket | Games |
|---|---|
| collision/bucket1 | 5 |
| collision/bucket2 | 237 |
| collision/bucket3 | 548 |
| timeout/bucket1 | 1 |
| timeout/bucket2 | 6 |
| timeout/bucket3 | 154 |

Example game (`game_seed_42.gif`, env seed 42): 61 apples in 794 steps, ended by win.
