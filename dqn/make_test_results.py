"""Build the published test results for the MLP and hybrid (CNN + features) Deep-Q agents.

    test_result/mlp/      curves.png, game_seed_42.gif, state_seed_42.gif, test_summary.md
    test_result/hybrid/   curves.png, game_seed_42.gif, state_seed_42.gif, test_summary.md

The test summaries come from the fixed-seed evaluation (dqn/v2/evaluate.py, 1000 games with env seeds
20000-20999); run it first if the summary files are missing:

    python -m dqn.v2.evaluate --games 1000 --out dqn/v2/eval/mlp_test    --policies mlp
    python -m dqn.v2.evaluate --games 1000 --out dqn/v2/eval/hybrid_test --policies old_hybrid

Usage (from the repo root):  python -m dqn.make_test_results
"""
import json
from pathlib import Path

from dqn.model_loader import load_trained_model
from dqn.plot import plot_learning_curves
from dqn.visualize import record_episode, record_episode_mlp, save_gameplay_gif, save_state_gif

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

MODELS = {
    "mlp": dict(model_type="mlp", path="dqn/snake_dqn.pth", log="dqn/dqn_training.csv", window=200,
                title="MLP DQN Training Analysis", record=record_episode_mlp, policy="mlp",
                summaries=["dqn/v2/eval/mlp_test/summary.json"], label="Deep-Q (basic): MLP"),
    "hybrid": dict(model_type="hybrid", path="dqn/hybrid/hybrid_snake_dqn.pth", log="dqn/hybrid/hybrid_training.csv",
                   window=500, title="Hybrid DQN Training Analysis", record=record_episode, policy="old_hybrid",
                   summaries=["dqn/v2/eval/hybrid_test/summary.json", "dqn/v2/eval/final/summary.json"],
                   label="Deep-Q (medium): CNN + per-action features"),
}


def write_summary(name, spec, out_dir, episode):
    path = next((ROOT / p for p in spec["summaries"] if (ROOT / p).exists()), None)
    if path is None:
        print(f"[{name}] no evaluation summary found; run dqn/v2/evaluate.py first (see module docstring)")
        return
    data = json.loads(path.read_text())
    s = data["policies"][spec["policy"]]
    cfg = data["config"]
    pct = lambda v: f"{100 * v:.1f}%"
    stw = f"{s['steps_to_win']:.0f}" if s["wins"] else "–"
    lines = [
        f"# {spec['label']}: test results",
        "",
        f"Greedy play on {s['games']} fixed test games (8x8 board, env seeds {cfg['base_seed']}–"
        f"{cfg['base_seed'] + s['games'] - 1}). 95% confidence intervals: Wilson for the win rate, "
        "percentile bootstrap for means.",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| Win rate (board filled, 61 apples) | {pct(s['win_rate'])} | "
        f"[{pct(s['win_rate_ci'][0])}, {pct(s['win_rate_ci'][1])}] |",
        f"| Mean apples per game | {s['mean_score']:.1f} | [{s['mean_score_ci'][0]:.1f}, {s['mean_score_ci'][1]:.1f}] |",
        f"| Best game (apples) | {s['max_score']} | |",
        f"| Mean steps per apple | {s['steps_per_apple']:.2f} | "
        f"[{s['steps_per_apple_ci'][0]:.2f}, {s['steps_per_apple_ci'][1]:.2f}] |",
        f"| Mean steps to win (won games) | {stw} | |",
        "",
        "Losses by cause and snake length (bucket 0: <10% of the board, 1: 10–25%, 2: 25–50%, 3: >50%):",
        "",
        "| Cause / length bucket | Games |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in s["deaths_by_reason_and_bucket"].items()],
        "",
        f"Example game (`game_seed_{SEED}.gif`, env seed {SEED}): {episode['score']} apples in {episode['steps']} "
        f"steps, ended by {episode['reason']}.",
        "",
    ]
    (out_dir / "test_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{name}] wrote {out_dir / 'test_summary.md'}")


def main():
    for name, spec in MODELS.items():
        out_dir = ROOT / "test_result" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_learning_curves(ROOT / spec["log"], out_dir, spec["window"], spec["title"])

        model, encoder, device, config = load_trained_model(spec["model_type"], ROOT / spec["path"])
        frames = spec["record"](model=model, encoder=encoder, device=device, size=config.get("size", 8), seed=SEED)
        save_gameplay_gif(frames, out_dir / f"game_seed_{SEED}.gif", fps=8)
        save_state_gif(frames, spec["model_type"], str(out_dir / f"state_seed_{SEED}.gif"), fps=8)
        episode = dict(score=frames[-1]["final_score"], steps=frames[-1]["steps"] + 1,
                       reason=frames[-1]["end_reason"])
        write_summary(name, spec, out_dir, episode)


if __name__ == "__main__":
    main()
