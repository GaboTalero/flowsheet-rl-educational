import numpy as np

from ..chemistry import STOICHIOMETRY, Stream
from .base import UnitResult, bounded_parameter


def reactor(stream, parameter, target=False):
    residence = 0.1 + 3.9 * bounded_parameter(parameter)
    rate = 0.6 * np.exp((stream.temperature - 330.0) / 80.0)
    if target:
        rate *= 0.8 / (1.0 + 0.15 * stream.flow[2])
    conversion = min(0.98, float(-np.expm1(-rate * residence)))
    extent = conversion * min(stream.flow[0], stream.flow[1])
    flow = stream.flow + STOICHIOMETRY * extent
    energy = 0.1 * residence * stream.total + 0.2 * extent
    return UnitResult(
        (Stream(flow, stream.temperature, stream.pressure),),
        energy * (1.2 if target else 1.0),
        1.0 + 0.1 * residence,
        float(extent),
        conversion,
    )
