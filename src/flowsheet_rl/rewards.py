"""Terminal scoring: no product revenue or success bonus paid twice."""


def reward_breakdown(result, state, config, reward_config, done, failure=False):
    r = reward_config
    terms = {
        name: 0.0
        for name in (
            "product_value",
            "utility",
            "capital",
            "complexity",
            "success",
            "specification",
            "open_streams",
            "failure",
        )
    }
    terms["step"] = -r.step
    if done:
        terms.update(
            product_value=r.product_value * result.product_recovery * result.product_purity,
            utility=-r.utility * result.energy,
            capital=-r.capital * result.capital_cost,
            complexity=-r.complexity * state.num_units,
            success=r.success * float(result.feasible and not failure),
            specification=-r.specification
            * (
                max(0.0, config.purity_min - result.product_purity)
                + max(0.0, config.recovery_min - result.product_recovery)
            ),
            open_streams=-r.open_stream * len(state.open_nodes),
            failure=-r.failure * float(failure),
        )
    return terms
