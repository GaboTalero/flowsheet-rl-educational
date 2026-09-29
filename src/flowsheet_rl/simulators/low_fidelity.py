"""Sequential modular DAG evaluator. Every proposal is recomputed from the feeds."""

import networkx as nx
import numpy as np

from ..chemistry import ELEMENTS, MOLECULAR_WEIGHTS, STOICHIOMETRY
from ..units import mixer, reactor, separator, thermal
from ..units.base import UnitResult
from .base import SimulationResult


class LowFidelitySimulator:
    fidelity = "source"

    def calculate(self, kind, inputs, parameter):
        if kind == "mixer":
            return mixer(inputs)
        if kind == "thermal":
            return thermal(inputs[0], parameter)
        if kind == "reactor":
            return reactor(inputs[0], parameter)
        if kind == "separator":
            return separator(inputs[0], parameter)
        return UnitResult((inputs[0],))

    def simulate(self, state, config):
        result = SimulationResult()
        try:
            state.validate()
        except ValueError as error:
            return SimulationResult(
                calculation_status="topology_error",
                topological_valid=False,
                numerical_converged=False,
                physical_feasible=False,
                infeasibility_reason=str(error),
            )
        product = np.zeros(3)
        frontier = np.zeros(3)
        g = state.graph
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for n in nx.topological_sort(g):
                    attrs = g.nodes[n]
                    kind = attrs["kind"]
                    inputs = [g.edges[s, n]["stream"] for s in sorted(g.predecessors(n))]
                    if kind == "feed":
                        unit = UnitResult((attrs["feed"],))
                    elif kind in ("open", "product", "purge"):
                        stream = inputs[0]
                        attrs["stream"] = stream
                        frontier += stream.flow
                        if kind == "product":
                            product += stream.flow
                        continue
                    else:
                        unit = self.calculate(kind, inputs, attrs.get("parameter", 0.0))
                        residual = float(np.abs(unit.residual(inputs)).max())
                        result.max_unit_balance_residual = max(
                            result.max_unit_balance_residual, residual
                        )
                        result.unit_diagnostics.append(
                            {
                                "node": n,
                                "kind": kind,
                                "balance_residual": residual,
                                "energy": unit.energy,
                                "extent": unit.extent,
                            }
                        )
                    attrs["stream"] = unit.outputs[0]
                    attrs["performance"] = unit.performance
                    result.energy += unit.energy
                    result.capital_cost += unit.cost
                    result.extent += unit.extent
                    for _, target, edge in g.out_edges(n, data=True):
                        edge["stream"] = unit.outputs[edge["port"]]
        except (FloatingPointError, OverflowError) as error:
            result.calculation_status = "numerical_failure"
            result.numerical_converged = False
            result.physical_feasible = False
            result.infeasibility_reason = str(error)
            result.warnings.append(type(error).__name__)
            return result
        except ValueError as error:
            result.calculation_status = "physical_violation"
            result.physical_feasible = False
            result.infeasibility_reason = str(error)
            return result
        feed = state.feed_flow
        residual = feed + STOICHIOMETRY * result.extent - frontier
        result.component_balance_residual = residual.tolist()
        result.elemental_balance_residual = (ELEMENTS @ (feed - frontier)).tolist()
        result.mass_balance_residual = float(MOLECULAR_WEIGHTS @ (feed - frontier))
        result.physical_feasible = bool(
            max(
                np.abs(residual).max(),
                result.max_unit_balance_residual,
                abs(result.mass_balance_residual),
            )
            <= config.balance_tolerance
        )
        if not result.physical_feasible:
            result.calculation_status = "physical_violation"
            result.infeasibility_reason = "balance_violation"
        result.product_c = float(product[2])
        result.product_purity = float(product[2] / product.sum()) if product.sum() > 0 else 0.0
        result.product_recovery = float(product[2] / min(feed[0], feed[1]))
        result.specification_satisfied = bool(
            result.product_purity >= config.purity_min
            and result.product_recovery >= config.recovery_min
        )
        result.closed = not state.open_nodes
        result.feasible = bool(
            result.closed and result.physical_feasible and result.specification_satisfied
        )
        result.profit_proxy = (
            10 * result.product_recovery - 0.2 * result.energy - 0.3 * result.capital_cost
        )
        result.economically_attractive = result.feasible and result.profit_proxy > 0
        return result
