import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from flowsheet_rl.actions import Action, Unit
from flowsheet_rl.baselines import RandomValidPolicy, RuleBasedPolicy
from flowsheet_rl.config import EnvironmentConfig
from flowsheet_rl.environment import FlowsheetEnv
from flowsheet_rl.simulators.low_fidelity import LowFidelitySimulator


def run_policy(policy, config=None, seed=0):
    env = FlowsheetEnv(config)
    obs, _ = env.reset(seed=seed)
    total, sizes = 0.0, []
    while not env.done:
        assert env.observation_space.contains(obs)
        sizes.append(len(obs["graph"].nodes))
        obs, r, _, _, info = env.step(policy.act(obs))
        total += r
        assert np.isfinite(r)
        assert max(abs(x) for x in info["component_balance_residual"]) < 1e-9
    return env, info, total, sizes


@pytest.mark.parametrize("fidelity", ["source", "target"])
def test_rule_reference_case(fidelity):
    env, info, total, sizes = run_policy(RuleBasedPolicy(), EnvironmentConfig(fidelity=fidelity))
    assert info["feasible"] and info["specification_satisfied"] and info["closed"]
    assert info["product_purity"] > 0.9 and info["product_recovery"] > 0.6
    assert total > 0 and max(sizes) > min(sizes)
    assert info["num_units"] == 4
    assert not info["bootstrap_allowed"]
    assert len(env.history) == 6


def test_random_valid_and_seed_replay():
    for seed in range(20):
        a = run_policy(RandomValidPolicy(seed), seed=seed)
        b = run_policy(RandomValidPolicy(seed), seed=seed)
        assert a[0].state.record() == b[0].state.record()
        assert a[2] == b[2]
        assert a[1]["termination_reason"] not in ("invalid_action", "simulation_failure")


def test_checker():
    check_env(FlowsheetEnv(), skip_render_check=True)


def test_invalid_action_preserves_topology():
    env = FlowsheetEnv()
    env.reset(seed=0)
    record = env.state.record()
    _, _, term, trunc, info = env.step(Action(0, Unit.REACTOR))
    assert term and not trunc
    assert info["infeasibility_reason"] == "closed_or_missing_stream"
    assert info["topological_valid"] is False
    assert record["nodes"] == env.state.record()["nodes"]
    with pytest.raises(RuntimeError):
        env.step(Action(1))


def test_truncation_and_poor_but_valid_case():
    cfg = EnvironmentConfig(max_steps=1)
    env = FlowsheetEnv(cfg)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(Action(1, Unit.PURGE))
    assert truncated and not terminated and info["termination_reason"] == "design_budget"
    assert info["topological_valid"] and not info["bootstrap_allowed"]
    env, info, _, _ = run_policy(RandomValidPolicy(0))
    env = FlowsheetEnv()
    env.reset(seed=0)
    env.step(Action(1, Unit.PURGE))
    _, reward, term, _, info = env.step(Action(3, Unit.PURGE))
    assert term and info["physical_feasible"] and info["topological_valid"]
    assert not info["specification_satisfied"] and reward < 0


def test_terminal_reward_not_repeated_and_no_free_product():
    env = FlowsheetEnv()
    obs, _ = env.reset(seed=0)
    assert not obs["unit_mask"][1, Unit.PRODUCT]
    _, reward, _, _, info = env.step(Action(env.config.max_nodes))
    assert reward < 0 and info["reward_breakdown"]["product_value"] == 0
    env = FlowsheetEnv()
    obs, _ = env.reset(seed=0)
    while not env.done:
        obs, r, _, _, info = env.step(RuleBasedPolicy().act(obs))
        if not env.done:
            assert r == -env.reward_config.step
            assert info["reward_breakdown"]["success"] == 0
    assert sum(info["reward_breakdown"].values()) == r


def test_failures_keep_diagnostic_reason():
    class Broken(LowFidelitySimulator):
        def calculate(self, kind, inputs, parameter):
            raise FloatingPointError("injected_solver_failure")

    env = FlowsheetEnv(simulator=Broken())
    env.reset(seed=0)
    _, r, done, _, info = env.step(Action(1, Unit.THERMAL, 0.5))
    assert done and r < 0
    assert info["calculation_status"] == "numerical_failure"
    assert not info["numerical_converged"]
    assert info["infeasibility_reason"] == "injected_solver_failure"
    assert env.state.num_units == 0


def test_inactive_parameter_cannot_leak_nan_and_failure_history_serializes():
    import json

    env = FlowsheetEnv()
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(Action(1, Unit.PURGE, np.nan))
    assert np.isfinite(obs["graph"].nodes).all()
    json.dumps(env.history, allow_nan=False)
    env.reset(seed=0)
    _, _, done, _, info = env.step(Action(1, Unit.THERMAL, np.nan))
    assert done and info["infeasibility_reason"] == "parameter_out_of_bounds"
    json.dumps(env.history, allow_nan=False)


def test_nonfinite_unit_output_diagnostic_is_numerical_failure():
    from flowsheet_rl.units.base import UnitResult

    class Nonfinite(LowFidelitySimulator):
        def calculate(self, kind, inputs, parameter):
            return UnitResult((inputs[0],), energy=np.inf)

    env = FlowsheetEnv(simulator=Nonfinite())
    env.reset(seed=0)
    obs, _, done, _, info = env.step(Action(1, Unit.THERMAL, 0.5))
    assert done and info["calculation_status"] == "numerical_failure"
    assert np.isfinite(obs["graph"].nodes).all()
