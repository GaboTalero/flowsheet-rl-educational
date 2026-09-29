import numpy as np

from ..chemistry import Stream
from .base import UnitResult, bounded_parameter


def separator(stream, parameter, target=False):
    parameter = bounded_parameter(parameter)
    recovery = 0.7 + 0.28 * parameter
    leakage = 0.02 + 0.08 * (1.0 - parameter)
    if target:
        recovery -= 0.03 * parameter**2
        leakage *= 1.2
    fractions = np.array([leakage, leakage, recovery])
    rich = Stream(stream.flow * fractions, stream.temperature, stream.pressure)
    residual = Stream(stream.flow * (1.0 - fractions), stream.temperature, stream.pressure)
    energy = stream.total * (0.2 + parameter**2) * (1.3 if target else 1.0)
    return UnitResult((rich, residual), energy, 0.6 + 0.3 * parameter, performance=recovery)
