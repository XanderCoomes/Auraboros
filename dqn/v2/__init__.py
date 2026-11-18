"""DQN v2: hybrid CNN + per-action features, tuned to win 8x8 Snake in as few steps as possible.

Modules
-------
state.py      egocentric board encoder, global scalars, per-action lookahead features
shield.py     optional safety shield (filters actions whose escape route is lost)
model.py      dueling hybrid network
replay.py     prioritized n-step replay buffer (preallocated numpy arrays)
agent.py      Double DQN agent (masking, soft target updates, gradient clipping)
train.py      vectorized training loop + CSV logging + checkpoints
baselines.py  common policy interface: DQN v2 / random / greedy-BFS / Hamiltonian-cycle / tabular / old hybrid
evaluate.py   fixed-seed evaluation of any set of policies -> per-game CSV + stats JSON
stats.py      Wilson CI, bootstrap CI, two-proportion / Fisher / Mann-Whitney tests
report.py     all figures + results table for the final report
gif.py        gameplay GIFs for any policy
tests/        pytest suite (python -m pytest dqn/v2/tests)

Typical pipeline (from the repo root):
    python -m dqn.v2.train --episodes 5000 --seed 0
    python -m dqn.v2.evaluate --games 1000 --out dqn/v2/eval/main --policies hamiltonian random greedy_bfs \
        tabular old_hybrid v2:dqn/v2/runs/seed0/best.pth v2shield:dqn/v2/runs/seed0/best.pth
    python -m dqn.v2.report --runs dqn/v2/runs/seed0 --eval dqn/v2/eval/main --out dqn/v2/figures
    python -m dqn.v2.gif --policy v2shield:dqn/v2/runs/seed0/best.pth --seed 10000
"""
