"""Gymnasium graph environment with transactional simulation and hybrid actions."""

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import CONTINUOUS_UNITS, UNIT_NAMES, Action, action_masks, validate_action
from .config import EnvironmentConfig, RewardConfig
from .graph_state import EDGE_DIM, GLOBAL_DIM, NODE_DIM, FlowsheetGraph
from .rewards import reward_breakdown
from .simulators import make_simulator


class FlowsheetEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["ansi"]}

    def __init__(self, config=None, reward_config=None, simulator=None, render_mode=None):
        self.config = config or EnvironmentConfig()
        self.reward_config = reward_config or RewardConfig()
        self.simulator = simulator or make_simulator(self.config.fidelity)
        self.render_mode = render_mode
        n = self.config.max_nodes
        self.action_space = spaces.Dict(
            {
                "stream": spaces.Discrete(n + 1),
                "unit": spaces.Discrete(6),
                "parameter": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "partner": spaces.Discrete(n),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "graph": spaces.Graph(
                    spaces.Box(0, np.inf, (NODE_DIM,), np.float32),
                    spaces.Box(0, np.inf, (EDGE_DIM,), np.float32),
                ),
                "globals": spaces.Box(0, np.inf, (GLOBAL_DIM,), np.float32),
                "stream_mask": spaces.MultiBinary(n + 1),
                "unit_mask": spaces.MultiBinary((n, 6)),
            }
        )
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.config
        rates = np.array([cfg.feed_a, cfg.feed_b]) * self.np_random.uniform(
            1 - cfg.feed_variation, 1 + cfg.feed_variation, 2
        )
        self.state = FlowsheetGraph.initial(*rates, cfg.feed_temperature)
        self.result = self.simulator.simulate(self.state, cfg)
        self.simulator_calls = 1
        self.done = False
        self.history = []
        return self.observation(), self.info("reset", {})

    def action_masks(self):
        return action_masks(self.state, self.config)

    def observation(self):
        cfg, r = self.config, self.result
        feed = self.state.feed_flow
        globals_ = [
            self.state.step / cfg.max_steps,
            self.state.num_units / cfg.max_units,
            r.energy / 10,
            r.capital_cost / 10,
            r.product_purity,
            r.product_recovery,
            max(0, cfg.max_steps - self.state.step) / cfg.max_steps,
            feed[0] / 2,
            feed[1] / 2,
        ]
        return {
            "graph": self.state.graph_instance(),
            "globals": np.asarray(globals_, dtype=np.float32),
            **self.action_masks(),
        }

    def info(self, reason, terms, failure_reason=None):
        metrics = self.result.as_dict()
        if reason == "invalid_action":
            metrics.update(
                topological_valid=False,
                feasible=False,
                calculation_status="invalid_action",
                infeasibility_reason=failure_reason,
            )
        elif reason == "simulation_failure":
            metrics.update(self.failed_result.as_dict())
        return {
            **metrics,
            "termination_reason": reason,
            "reward_breakdown": terms,
            "simulator_calls": self.simulator_calls,
            "num_units": self.state.num_units,
            "steps": self.state.step,
            "action_masks": self.action_masks(),
            "bootstrap_allowed": not self.done,
            "fidelity": self.simulator.fidelity,
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("reset_required_after_episode")
        failure_reason = None
        if isinstance(action, dict):
            # Validate the complete Gym representation before integer conversion.
            if not self.action_space.contains(action):
                failure_reason = "action_outside_space"
            else:
                action = Action.from_dict(action)
        elif not isinstance(action, Action):
            failure_reason = "unsupported_action_type"
        if failure_reason is None:
            failure_reason = validate_action(action, self.state, self.config)
        if isinstance(action, Action) and action.unit not in CONTINUOUS_UNITS:
            # Inactive parameters have no physical meaning and must not enter features.
            action = Action(action.stream, action.unit, 0.0, action.partner)
        self.state.step += 1
        terminated, truncated = False, False
        reason = "running"
        if failure_reason:
            terminated, reason = True, "invalid_action"
        elif action.stream == self.config.max_nodes:
            terminated, reason = True, "explicit_stop"
        else:
            candidate = self.state.copy()
            candidate.add_unit(
                action.stream, UNIT_NAMES[action.unit], action.parameter, action.partner
            )
            result = self.simulator.simulate(candidate, self.config)
            self.simulator_calls += 1
            if result.calculation_status != "ok":
                self.failed_result = result
                terminated, reason = True, "simulation_failure"
            else:
                self.state, self.result = candidate, result
                if not self.state.open_nodes:
                    terminated, reason = True, "all_streams_closed"
        if not terminated and self.state.step >= self.config.max_steps:
            truncated, reason = True, "design_budget"
        self.done = terminated or truncated
        failure = reason in ("invalid_action", "simulation_failure")
        terms = reward_breakdown(
            self.result, self.state, self.config, self.reward_config, self.done, failure
        )
        self.history.append(
            {
                "action": (
                    {
                        "stream": action.stream,
                        "unit": action.unit,
                        "parameter": float(action.parameter)
                        if np.isfinite(action.parameter)
                        else None,
                        "partner": action.partner,
                    }
                    if isinstance(action, Action)
                    else None
                ),
                "reward": float(sum(terms.values())),
                "reason": reason,
                "failure_reason": failure_reason,
            }
        )
        return (
            self.observation(),
            float(sum(terms.values())),
            terminated,
            truncated,
            self.info(reason, terms, failure_reason),
        )

    def render(self):
        return "\n".join(
            f"{n}: {a['kind']} -> {list(self.state.graph.successors(n))}"
            for n, a in self.state.graph.nodes(data=True)
        )
