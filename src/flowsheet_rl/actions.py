"""Environment-independent action records and deterministic engineering masks."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Unit(IntEnum):
    MIXER = 0
    THERMAL = 1
    REACTOR = 2
    SEPARATOR = 3
    PRODUCT = 4
    PURGE = 5


UNIT_NAMES = ("mixer", "thermal", "reactor", "separator", "product", "purge")
CONTINUOUS_UNITS = (Unit.THERMAL, Unit.REACTOR, Unit.SEPARATOR)


@dataclass(frozen=True)
class Action:
    stream: int
    unit: int = int(Unit.PURGE)
    parameter: float = 0.0
    partner: int = 0

    def as_dict(self):
        return {
            "stream": self.stream,
            "unit": self.unit,
            "parameter": np.array([self.parameter], dtype=np.float32),
            "partner": self.partner,
        }

    @classmethod
    def from_dict(cls, action):
        return cls(
            int(action["stream"]),
            int(action["unit"]),
            float(np.asarray(action["parameter"]).item()),
            int(action["partner"]),
        )


def action_masks(state, config):
    stream = np.zeros(config.max_nodes + 1, dtype=bool)
    units = np.zeros((config.max_nodes, len(Unit)), dtype=bool)
    stream[config.max_nodes] = True  # STOP remains available, including after closure.
    for n in state.open_nodes:
        stream[n] = True
        flow = state.graph.nodes[n]["stream"].flow
        units[n, Unit.PURGE] = True
        # Zero-C branches cannot be products; purity is NOT an action validity criterion.
        units[n, Unit.PRODUCT] = flow[2] > 1e-12
        if state.num_units < config.max_units:
            units[n, Unit.THERMAL] = True
            units[n, Unit.REACTOR] = min(flow[0], flow[1]) > 1e-12
            units[n, Unit.SEPARATOR] = flow[2] > 1e-12
            units[n, Unit.MIXER] = len(state.open_nodes) > 1
    return {"stream_mask": stream, "unit_mask": units}


def validate_action(action, state, config):
    masks = action_masks(state, config)
    if (
        not isinstance(action.stream, (int, np.integer))
        or not 0 <= action.stream <= config.max_nodes
    ):
        return "invalid_stream"
    if not masks["stream_mask"][action.stream]:
        return "closed_or_missing_stream"
    if action.stream == config.max_nodes:
        return None
    if not isinstance(action.unit, (int, np.integer)) or not 0 <= action.unit < len(Unit):
        return "invalid_unit"
    if not masks["unit_mask"][action.stream, action.unit]:
        return "masked_unit"
    if action.unit == Unit.MIXER and (
        action.partner == action.stream or action.partner not in state.open_nodes
    ):
        return "invalid_mixer_partner"
    if action.unit in CONTINUOUS_UNITS and (
        not np.isfinite(action.parameter) or not 0 <= action.parameter <= 1
    ):
        return "parameter_out_of_bounds"
    return None
