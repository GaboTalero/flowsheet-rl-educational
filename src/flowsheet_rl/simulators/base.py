from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class SimulationResult:
    calculation_status: str = "ok"
    topological_valid: bool = True
    numerical_converged: bool = True
    physical_feasible: bool = True
    specification_satisfied: bool = False
    closed: bool = False
    feasible: bool = False
    economically_attractive: bool = False
    component_balance_residual: list = field(default_factory=lambda: [0.0] * 3)
    elemental_balance_residual: list = field(default_factory=lambda: [0.0] * 2)
    mass_balance_residual: float = 0.0
    max_unit_balance_residual: float = 0.0
    energy: float = 0.0
    capital_cost: float = 0.0
    product_purity: float = 0.0
    product_recovery: float = 0.0
    product_c: float = 0.0
    profit_proxy: float = 0.0
    extent: float = 0.0
    infeasibility_reason: str | None = None
    warnings: list = field(default_factory=list)
    unit_diagnostics: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


class Simulator(Protocol):
    fidelity: str

    def simulate(self, state, config) -> SimulationResult:
        """Populate calculated stream attributes and return structured diagnostics."""
        ...
