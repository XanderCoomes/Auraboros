"""Shared game loop; rendering is optional and evaluation never updates the agent."""
import numpy as np


def run_episode(env, agent, epsilon, learn=True, viewer=None, title="Snake", summary=""):
    state = env.reset()
    done, total_reward, info = False, 0.0, {"score": 0}
    if viewer:
        viewer.show(env, title, epsilon, total_reward, summary=summary)
    while not done:
        action = agent.act(state, epsilon)
        next_state, reward, done, info = env.step(action)
        if learn:
            agent.update(state, action, reward, next_state, done)
        state, total_reward = next_state, total_reward + reward
        if viewer:
            viewer.show(env, title, epsilon, total_reward, reward, action,
                        summary=summary, reason=info.get("reason"))
    return info["score"], env.steps, total_reward, info


def evaluation_scores(env, agent, n=100, viewer=None, title="Evaluation"):
    """Greedy games, preserving the agent's RNG so checks don't change training."""
    scores = []
    rng_state = agent.rng.getstate()
    try:
        for game in range(n):
            summary = (f"Completed {game} | Mean {np.mean(scores):.2f} | Best {max(scores)}"
                       if scores else "Greedy policy | Learning disabled")
            scores.append(run_episode(env, agent, 0.0, learn=False, viewer=viewer,
                                      title=f"{title} | Game {game + 1}/{n}",
                                      summary=summary)[0])
    finally:
        agent.rng.setstate(rng_state)
    return scores


def evaluate(env, agent, n=100, viewer=None, title="Evaluation"):
    return float(np.mean(evaluation_scores(env, agent, n, viewer, title)))
