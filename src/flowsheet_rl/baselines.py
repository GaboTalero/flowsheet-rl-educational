"""Both baselines operate under exactly the environment's action masks."""

import numpy as np

from .actions import CONTINUOUS_UNITS, Action, Unit


class RandomValidPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        stream = int(self.rng.choice(np.flatnonzero(observation["stream_mask"])))
        stop = len(observation["stream_mask"]) - 1
        if stream == stop:
            return Action(stop)
        unit = int(self.rng.choice(np.flatnonzero(observation["unit_mask"][stream])))
        partner = 0
        if unit == Unit.MIXER:
            candidates = np.flatnonzero(observation["stream_mask"][:-1]).tolist()
            candidates.remove(stream)
            partner = int(self.rng.choice(candidates))
        parameter = float(self.rng.uniform()) if unit in CONTINUOUS_UNITS else 0.0
        return Action(stream, unit, parameter, partner)


class RuleBasedPolicy:
    def act(self, observation):
        from .graph_state import NODE_TYPES

        g = observation["graph"]
        opens = np.flatnonzero(observation["stream_mask"][:-1])
        types = g.nodes[:, : len(NODE_TYPES)].argmax(axis=1)
        if not len(opens):
            return Action(len(observation["stream_mask"]) - 1)
        n = int(opens[0])
        masks = observation["unit_mask"]
        if not (types == NODE_TYPES.index("mixer")).any() and masks[n, Unit.MIXER]:
            return Action(n, Unit.MIXER, partner=int(opens[1]))
        if not (types == NODE_TYPES.index("thermal")).any() and masks[n, Unit.THERMAL]:
            return Action(n, Unit.THERMAL, (370 - 280) / 170)
        if not (types == NODE_TYPES.index("reactor")).any() and masks[n, Unit.REACTOR]:
            return Action(n, Unit.REACTOR, 0.85)
        if not (types == NODE_TYPES.index("separator")).any() and masks[n, Unit.SEPARATOR]:
            return Action(n, Unit.SEPARATOR, 0.9)
        # Only observable incoming stream composition is used for routing.
        for node in opens:
            incoming = np.flatnonzero(g.edge_links[:, 1] == node)[0]
            if g.edges[incoming, 3] >= 0.9 and masks[node, Unit.PRODUCT]:
                return Action(int(node), Unit.PRODUCT)
        return Action(n, Unit.PURGE)
