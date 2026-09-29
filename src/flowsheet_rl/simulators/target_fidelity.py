"""A deliberately perturbed analytical target, not a rigorous process simulator."""

from ..units import reactor, separator, thermal
from .low_fidelity import LowFidelitySimulator


class TargetFidelitySimulator(LowFidelitySimulator):
    fidelity = "target"

    def calculate(self, kind, inputs, parameter):
        if kind == "reactor":
            return reactor(inputs[0], parameter, target=True)
        if kind == "separator":
            return separator(inputs[0], parameter, target=True)
        if kind == "thermal":
            return thermal(inputs[0], parameter, utility_factor=1.2)
        return super().calculate(kind, inputs, parameter)
