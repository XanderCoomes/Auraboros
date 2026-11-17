
from pathlib import Path

from dqn.model_loader import load_trained_model
from dqn.visualize import record_episode, save_gameplay_gif

from dqn.visualize import save_state_gif


# Load the trained hybrid model once.
model, encoder, device, config = load_trained_model(
    model_type="hybrid",
    model_path="dqn/hybrid/hybrid_snake_dqn.pth"
)

print("Device:", device)
print("Model:", type(model).__name__)
print("Encoder:", type(encoder).__name__)

# Create output directory.
output_dir = Path("test_result/hybrid/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

results = []

# Test 10 episodes with different seeds.
for seed in range(42, 52):

    print(f"\nTesting episode with seed {seed}...")

    frames = record_episode(
        model=model,
        encoder=encoder,
        device=device,
        size=8,
        seed=seed
    )

    save_state_gif(
        frames,
        model_type="hybrid",
        output_path=str(output_dir / f"state_seed_{seed}.gif"),
        fps=8
    )

    final_score = frames[-1]["final_score"]
    steps = frames[-1]["steps"]
    end_reason = frames[-1]["end_reason"]

    output_path = output_dir / f"game_seed_{seed}.gif"

    save_gameplay_gif(
        frames,
        output_path=str(output_path),
        fps=8
    )

    results.append({
        "seed": seed,
        "score": final_score,
        "steps": steps,
        "end_reason": end_reason
    })

    print("Frames:", len(frames))
    print("Final score:", final_score)
    print("End reason:", end_reason)
    print("Feature shape:", frames[0]["state"][2].shape)
    print("Saved:", output_path)

# Summary.
print("\n========== TEST SUMMARY ==========")

for result in results:
    print(result)

average_score = sum(r["score"] for r in results) / len(results)
wins = sum(r["end_reason"] == "win" for r in results)

print(f"\nAverage score: {average_score:.2f}")
print(f"Wins: {wins}/10")
print(f"Win rate: {wins * 10}%")
