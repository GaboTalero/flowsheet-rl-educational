import numpy as np

from flowsheet_rl.actions import Action, Unit, validate_action
from flowsheet_rl.config import EnvironmentConfig
from flowsheet_rl.environment import FlowsheetEnv


def test_conditional_masks_and_budget():
    env = FlowsheetEnv(EnvironmentConfig(max_units=1))
    obs, _ = env.reset(seed=0)
    assert np.flatnonzero(obs["stream_mask"]).tolist() == [1, 3, env.config.max_nodes]
    assert not obs["unit_mask"][1, Unit.REACTOR]
    assert not obs["unit_mask"][1, Unit.SEPARATOR]
    assert obs["unit_mask"][1, Unit.MIXER]
    assert validate_action(Action(1, Unit.MIXER, partner=1), env.state, env.config)
    obs, _, _, _, _ = env.step(Action(1, Unit.MIXER, partner=3))
    assert env.state.num_units == 1
    assert not obs["unit_mask"][:, :4].any()
    assert obs["unit_mask"][4, Unit.PURGE]


def test_physical_parameter_bounds_and_direct_invalid_mixer():
    env = FlowsheetEnv()
    env.reset(seed=0)
    assert validate_action(Action(1, Unit.THERMAL, np.nan), env.state, env.config)
    assert validate_action(Action(1, Unit.THERMAL, 1.1), env.state, env.config)
    import pytest

    with pytest.raises(ValueError):
        env.state.add_unit(1, "mixer", partner=1)
