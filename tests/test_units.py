import numpy as np
import pytest

from flowsheet_rl.chemistry import ELEMENTS, MOLECULAR_WEIGHTS, Stream
from flowsheet_rl.units import mixer, reactor, separator, thermal


@pytest.mark.parametrize("parameter", [0.0, 0.3, 1.0])
def test_units_balance_and_bounds(parameter):
    s = Stream(np.array([1.2, 0.8, 0.2]), 370)
    for unit in (
        reactor(s, parameter),
        separator(s, parameter),
        thermal(s, parameter),
        reactor(s, parameter, target=True),
        separator(s, parameter, target=True),
    ):
        assert np.max(np.abs(unit.residual([s]))) < 1e-12
        total = sum((o.flow for o in unit.outputs), np.zeros(3))
        np.testing.assert_allclose(ELEMENTS @ total, ELEMENTS @ s.flow, atol=1e-12)
        assert abs(MOLECULAR_WEIGHTS @ (total - s.flow)) < 1e-12
        assert unit.energy >= 0
        assert all((o.flow >= 0).all() for o in unit.outputs)


def test_independent_reactor_calculation():
    s = Stream(np.array([1.0, 1.0, 0.0]), 330)
    r = reactor(s, 1.0)
    extent = 1 - np.exp(-0.6 * 4)
    np.testing.assert_allclose(r.outputs[0].flow, [1 - extent, 1 - extent, extent])
    assert r.performance <= 0.98


def test_mixer_energy_and_material():
    a, b = Stream(np.array([1.0, 0.0, 0.0]), 300), Stream(np.array([0.0, 2.0, 0.0]), 390)
    result = mixer([a, b])
    assert result.outputs[0].temperature == 360
    np.testing.assert_allclose(result.outputs[0].flow, [1, 2, 0])
    assert np.max(np.abs(result.residual([a, b]))) == 0


@pytest.mark.parametrize("p", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_parameters(p):
    with pytest.raises(ValueError):
        reactor(Stream(np.ones(3)), p)


def test_zero_flow_and_no_reactant():
    zero = Stream(np.zeros(3))
    assert np.isfinite(zero.composition).all()
    assert mixer([zero, zero]).outputs[0].total == 0
    assert reactor(Stream(np.array([1.0, 0.0, 0.0])), 1).extent == 0
    assert thermal(Stream(np.ones(3), 400), 0).energy > 0


def test_configuration_rejects_nonfinite_weights():
    from flowsheet_rl.config import RewardConfig, TrainConfig

    with pytest.raises(ValueError):
        RewardConfig(product_value=np.nan)
    with pytest.raises(ValueError):
        TrainConfig(learning_rate=np.inf)
