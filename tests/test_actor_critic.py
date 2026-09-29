import torch

from flowsheet_rl.actions import Action, Unit, validate_action
from flowsheet_rl.environment import FlowsheetEnv
from flowsheet_rl.models import ActorCritic
from flowsheet_rl.models.distributions import masked_categorical


def test_masked_probabilities_and_entropy():
    dist = masked_categorical(torch.tensor([1.0, 2.0, 3.0]), [True, False, True])
    assert dist.probs[1] == 0
    assert torch.isfinite(dist.entropy())
    for _ in range(100):
        assert int(dist.sample()) != 1


def test_joint_likelihood_and_replay():
    torch.manual_seed(4)
    model, env = ActorCritic(), FlowsheetEnv()
    obs, _ = env.reset(seed=0)
    for action in [
        Action(1, Unit.MIXER, partner=3),
        Action(1, Unit.THERMAL, 0.3),
        Action(env.config.max_nodes),
    ]:
        out = model(obs, action)
        torch.testing.assert_close(out.log_prob, sum(out.component_log_probs.values()))
        assert torch.isfinite(out.value) and torch.isfinite(out.entropy)
        logp, entropy, value = model.evaluate_actions([obs, obs], [action, action])
        torch.testing.assert_close(logp[0], out.log_prob)
        torch.testing.assert_close(entropy[0], out.entropy)
        torch.testing.assert_close(value[0], out.value)
    stop = model(obs, Action(env.config.max_nodes))
    assert list(stop.component_log_probs) == ["stream"]


def test_actor_samples_valid_actions_and_gradients():
    torch.manual_seed(2)
    env, model = FlowsheetEnv(), ActorCritic()
    seen_continuous = False
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        while not env.done:
            out = model(obs)
            assert validate_action(out.action, env.state, env.config) is None
            assert 0 <= out.action.parameter <= 1
            seen_continuous |= "parameter" in out.component_log_probs
            replay = model(obs, out.action)
            torch.testing.assert_close(replay.log_prob, out.log_prob)
            (-out.log_prob + out.value**2 - 0.01 * out.entropy).backward()
            obs, *_ = env.step(out.action)
    assert seen_continuous
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_chain_entropy_matches_independent_monte_carlo():
    torch.manual_seed(3)
    env, model = FlowsheetEnv(), ActorCritic()
    obs, _ = env.reset(seed=0)
    with torch.no_grad():
        nodes, graphs = model.encoder(
            __import__("flowsheet_rl.graph_state", fromlist=["to_pyg"]).to_pyg(obs)
        )
        stream, units, params, partners, entropy = model._distributions(obs, nodes, graphs[0])
        draws = stream.sample((12000,))
        surprisal = -stream.log_prob(draws)
        for node, unit_dist in units.items():
            selection = draws == node
            count = int(selection.sum())
            u = unit_dist.sample((count,))
            value = -unit_dist.log_prob(u)
            for unit in (Unit.THERMAL, Unit.REACTOR, Unit.SEPARATOR):
                chosen = u == unit
                p = params[node][unit].sample((int(chosen.sum()),))
                value[chosen] -= params[node][unit].log_prob(p)
            if node in partners:
                chosen = u == Unit.MIXER
                p = partners[node].sample((int(chosen.sum()),))
                value[chosen] -= partners[node].log_prob(p)
            surprisal[selection] += value
        assert abs(float(surprisal.mean() - entropy)) < 0.06
