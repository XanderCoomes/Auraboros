"""Fixed-seed evaluation of any set of policies -> per-game CSV + summary JSON with confidence intervals.

Every policy plays the SAME games: game i uses env seed base_seed + i, so apple sequences match as long as
the policies make the same moves. This pairing removes apple-placement luck from comparisons.

Usage (from the repo root):
    python -m dqn.v2.evaluate --games 1000 --out dqn/v2/eval/main \
        --policies random greedy_bfs hamiltonian tabular old_hybrid \
                   v2:dqn/v2/runs/seed0/best.pth v2shield:dqn/v2/runs/seed0/best.pth
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from snake_env import SnakeEnv

from . import stats
from .baselines import build_policy

RECORD_FIELDS = ("policy", "game", "seed", "score", "length", "steps", "win", "reason", "bucket",
                 "head_r", "head_c", "steps_per_apple")


def run_games(policy, n_games, size=8, base_seed=10_000, batch=256, length_timeout=False, max_steps=None):
    """Play n_games greedy games of `policy` in lockstep batches -> list of per-game dicts."""
    records = []
    for start in range(0, n_games, batch):
        games = range(start, min(n_games, start + batch))
        envs = {g: SnakeEnv(size=size, seed=base_seed + g, length_timeout=length_timeout) for g in games}
        policy.reset(base_seed + start)
        while envs:
            live = list(envs)
            actions = policy.act_batch([envs[g] for g in live])
            for g, a in zip(live, actions):
                env = envs[g]
                _, _, done, info = env.step(int(a))
                if not done and max_steps is not None and env.steps >= max_steps:
                    done, info = True, {**info, "reason": "max_steps"}
                if done:
                    del envs[g]
                    head = env.snake[0]
                    records.append(dict(
                        policy=policy.name, game=g, seed=base_seed + g, score=info["score"],
                        length=info["length"], steps=env.steps, win=int(info.get("reason") == "win"),
                        reason=info.get("reason", ""), bucket=info["bucket"], head_r=head[0], head_c=head[1],
                        steps_per_apple=env.steps / max(1, info["score"])))
    records.sort(key=lambda r: r["game"])
    return records


def summarize(records, confidence=0.95):
    """Summary statistics of one policy's games."""
    n = len(records)
    wins = sum(r["win"] for r in records)
    scores = [r["score"] for r in records]
    spa = [r["steps_per_apple"] for r in records if r["score"] > 0]
    win_steps = [r["steps"] for r in records if r["win"]]
    lo, hi = stats.wilson_ci(wins, n, confidence)
    mean_score = stats.bootstrap_ci(scores, confidence=confidence)
    mean_spa = stats.bootstrap_ci(spa, confidence=confidence)
    mean_win_steps = stats.bootstrap_ci(win_steps, confidence=confidence)
    deaths = Counter((r["reason"], r["bucket"]) for r in records if not r["win"])
    return {
        "games": n, "wins": wins, "win_rate": wins / n if n else 0.0, "win_rate_ci": [lo, hi],
        "mean_score": mean_score[0], "mean_score_ci": list(mean_score[1:]),
        "max_score": max(scores) if scores else 0,
        "steps_per_apple": mean_spa[0], "steps_per_apple_ci": list(mean_spa[1:]),
        "steps_to_win": mean_win_steps[0], "steps_to_win_ci": list(mean_win_steps[1:]),
        "deaths_by_reason_and_bucket": {f"{reason}/bucket{b}": c for (reason, b), c in sorted(deaths.items())},
    }


def compare(rec_a, rec_b):
    """Tests of policy A vs policy B: win rates and steps to win."""
    ka, na = sum(r["win"] for r in rec_a), len(rec_a)
    kb, nb = sum(r["win"] for r in rec_b), len(rec_b)
    z, p_z = stats.two_proportion_test(ka, na, kb, nb)
    u, p_u = stats.mann_whitney([r["steps"] for r in rec_a if r["win"]],
                                [r["steps"] for r in rec_b if r["win"]])
    return {"win_rate_diff": ka / na - kb / nb, "z": z, "p_two_proportion": p_z,
            "p_fisher": stats.fisher_exact(ka, na, kb, nb), "steps_to_win_U": u, "p_mann_whitney": p_u}


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        w.writeheader()
        w.writerows(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", required=True, help="see baselines.build_policy")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=10_000)
    ap.add_argument("--length-timeout", action="store_true", help="evaluate with the length-scaled timeout")
    ap.add_argument("--reference", default=None, help="policy name every other policy is compared against "
                                                      "(default: the first policy)")
    ap.add_argument("--out", default="dqn/v2/eval/latest")
    args = ap.parse_args()

    out = Path(args.out)
    all_records, summary = {}, {"config": vars(args), "policies": {}, "comparisons": {}}
    for spec in args.policies:
        policy = build_policy(spec)
        t0 = time.time()
        rec = run_games(policy, args.games, args.size, args.base_seed, length_timeout=args.length_timeout)
        all_records[policy.name] = rec
        s = summarize(rec)
        summary["policies"][policy.name] = s
        print(f"{policy.name:>16}: win {s['win_rate']:.1%} [{s['win_rate_ci'][0]:.1%}, {s['win_rate_ci'][1]:.1%}]"
              f"  score {s['mean_score']:.2f}  steps/apple {s['steps_per_apple']:.2f}"
              f"  steps-to-win {s['steps_to_win']:.0f}  ({time.time() - t0:.0f}s)")

    ref = args.reference or next(iter(all_records))
    for name, rec in all_records.items():
        if name != ref:
            summary["comparisons"][f"{name} vs {ref}"] = compare(rec, all_records[ref])

    write_records(out / "games.csv", [r for rec in all_records.values() for r in rec])
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {out / 'games.csv'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
