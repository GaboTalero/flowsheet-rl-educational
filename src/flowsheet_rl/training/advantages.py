import numpy as np


def generalized_advantages(
    rewards, values, next_values, bootstrap, episode_ends, gamma=0.99, lam=0.95
):
    """Separate value bootstrap from recursion across independent episodes."""
    rewards, values, next_values = (
        np.asarray(x, dtype=np.float64) for x in (rewards, values, next_values)
    )
    bootstrap, episode_ends = np.asarray(bootstrap), np.asarray(episode_ends)
    deltas = rewards + gamma * next_values * bootstrap - values
    advantages = np.zeros_like(deltas)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = deltas[t] + gamma * lam * (not episode_ends[t]) * running
        advantages[t] = running
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)
