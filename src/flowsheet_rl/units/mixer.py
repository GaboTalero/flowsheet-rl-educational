import numpy as np

from ..chemistry import Stream
from .base import UnitResult


def mixer(inputs):
    if len(inputs) != 2:
        raise ValueError("mixer_requires_two_inlets")
    flow = sum((s.flow for s in inputs), np.zeros(3))
    total = sum(s.total for s in inputs)
    temperature = (
        sum(s.total * s.temperature for s in inputs) / total
        if total > 0
        else sum(s.temperature for s in inputs) / len(inputs)
    )
    return UnitResult((Stream(flow, temperature, min(s.pressure for s in inputs)),), cost=0.1)
