from dataclasses import dataclass

import torch

from .advantages import generalized_advantages


@dataclass
class Rollout:
    observations: list
    actions: list
    log_probs: list
    advantages: object
    returns: object
    episodes: list


class Collector:
    def __init__(self, env, seed):
        self.env = env
        self.observation, _ = env.reset(seed=seed)
        self.episode_return = 0.0
        self.total_steps = 0
        self.total_episodes = 0
        self.first_feasible_step = None
        self.first_feasible_episode = None

    def state_dict(self):
        return {
            "graph": self.env.state.record(),
            "simulator_calls": self.env.simulator_calls,
            "history": self.env.history,
            "env_rng": self.env.np_random.bit_generator.state,
            "episode_return": self.episode_return,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "first_feasible_step": self.first_feasible_step,
            "first_feasible_episode": self.first_feasible_episode,
        }

    def load_state_dict(self, state):
        from ..graph_state import FlowsheetGraph

        self.env.state = FlowsheetGraph.from_record(state["graph"])
        self.env.result = self.env.simulator.simulate(self.env.state, self.env.config)
        if self.env.result.calculation_status != "ok":
            raise ValueError("invalid_restored_simulator_state")
        self.env.simulator_calls = state["simulator_calls"]
        self.env.history = state["history"]
        self.env.np_random.bit_generator.state = state["env_rng"]
        self.env.done = False  # Collector resets immediately after every completed episode.
        self.observation = self.env.observation()
        for name in (
            "episode_return",
            "total_steps",
            "total_episodes",
            "first_feasible_step",
            "first_feasible_episode",
        ):
            setattr(self, name, state[name])

    @torch.no_grad()
    def collect(self, model, count, gamma, lam):
        observations, actions, logs = [], [], []
        rewards, values, next_values, bootstraps, ends, episodes = [], [], [], [], [], []
        model.eval()
        for _ in range(count):
            out = model(self.observation)
            next_observation, reward, terminated, truncated, info = self.env.step(out.action)
            done = terminated or truncated
            observations.append(self.observation)
            actions.append(out.action)
            logs.append(float(out.log_prob))
            rewards.append(reward)
            values.append(float(out.value))
            bootstrap = info["bootstrap_allowed"]
            next_values.append(float(model.value(next_observation)) if bootstrap else 0.0)
            bootstraps.append(bootstrap)
            ends.append(done)
            self.episode_return += reward
            self.total_steps += 1
            self.observation = next_observation
            if done:
                self.total_episodes += 1
                episodes.append(
                    {
                        "episode": self.total_episodes,
                        "steps": self.total_steps,
                        "return": self.episode_return,
                        "feasible": info["feasible"],
                        "termination_reason": info["termination_reason"],
                        **{
                            k: info[k]
                            for k in (
                                "product_purity",
                                "product_recovery",
                                "energy",
                                "capital_cost",
                                "num_units",
                                "simulator_calls",
                            )
                        },
                    }
                )
                if info["feasible"] and self.first_feasible_step is None:
                    self.first_feasible_step = self.total_steps
                    self.first_feasible_episode = self.total_episodes
                self.observation, _ = self.env.reset()
                self.episode_return = 0.0
        advantages, returns = generalized_advantages(
            rewards, values, next_values, bootstraps, ends, gamma, lam
        )
        return Rollout(observations, actions, logs, advantages, returns, episodes)
