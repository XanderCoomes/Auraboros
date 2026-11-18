#!/usr/bin/env bash
# Experiment queue for the report: evaluation of seed 0, then the remaining training runs (sequential,
# because parallel runs compete for the CPU that computes the BFS features).
set -e
python -u -m dqn.v2.evaluate --games 1000 --out dqn/v2/eval/seed0 --reference dqn_v2 \
  --policies v2:dqn/v2/runs/seed0/best.pth v2shield:dqn/v2/runs/seed0/best.pth \
             old_hybrid tabular tabular_bucket greedy_bfs random hamiltonian > dqn/v2/eval_seed0.log 2>&1
for s in 1 2; do
  python -u -m dqn.v2.train --episodes 5000 --seed $s --eval-interval 250 --out dqn/v2/runs/seed$s > dqn/v2/runs_seed$s.log 2>&1
done
for s in 0 1 2; do
  python -u -m dqn.v2.train --episodes 5000 --seed $s --eval-interval 250 --shield --out dqn/v2/runs/shield_seed$s > dqn/v2/runs_shield_seed$s.log 2>&1
done
echo "queue done"
