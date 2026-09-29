from ..chemistry import Stream
from .base import UnitResult, bounded_parameter


def thermal(stream, parameter, utility_factor=1.0):
    temperature = 280.0 + 170.0 * bounded_parameter(parameter)
    energy = stream.total * abs(temperature - stream.temperature) * 0.01 * utility_factor
    return UnitResult((Stream(stream.flow, temperature, stream.pressure),), energy, 0.3)
