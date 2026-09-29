from dataclasses import dataclass

import numpy as np

from ..chemistry import STOICHIOMETRY, Stream


@dataclass
class UnitResult:
    outputs: tuple[Stream, ...]
    energy: float = 0.0
    cost: float = 0.0
    extent: float = 0.0
    performance: float = 0.0

    def __post_init__(self):
        for name in ("energy", "cost", "extent", "performance"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite_unit_diagnostic")
            if value < 0:
                raise ValueError("negative_unit_diagnostic")
            setattr(self, name, value)

    def residual(self, inputs):
        return (
            sum((s.flow for s in self.outputs), np.zeros(3))
            - sum((s.flow for s in inputs), np.zeros(3))
            - STOICHIOMETRY * self.extent
        )


def bounded_parameter(parameter):
    if not np.isfinite(parameter) or not 0 <= parameter <= 1:
        raise ValueError("parameter_out_of_bounds")
    return float(parameter)
