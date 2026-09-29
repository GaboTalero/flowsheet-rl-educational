"""Synthetic molar flows; A+B->C conserves two abstract elemental inventories."""

from dataclasses import dataclass

import numpy as np

STOICHIOMETRY = np.array([-1.0, -1.0, 1.0])
MOLECULAR_WEIGHTS = np.array([1.0, 1.0, 2.0])
ELEMENTS = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])


@dataclass(frozen=True)
class Stream:
    flow: np.ndarray
    temperature: float = 300.0
    pressure: float = 1.0

    def __post_init__(self):
        flow = np.array(self.flow, dtype=np.float64, copy=True)
        if flow.shape != (3,) or not np.isfinite(flow).all() or (flow < 0).any():
            raise ValueError("invalid_component_flow")
        if not np.isfinite([self.temperature, self.pressure]).all():
            raise ValueError("nonfinite_stream_condition")
        if not 280 <= self.temperature <= 450 or self.pressure <= 0:
            raise ValueError("stream_condition_out_of_bounds")
        flow.setflags(write=False)
        object.__setattr__(self, "flow", flow)

    @property
    def total(self):
        return float(self.flow.sum())

    @property
    def composition(self):
        return self.flow / self.total if self.total > 0 else np.zeros(3)

    def record(self):
        return {
            "flow": self.flow.tolist(),
            "temperature": float(self.temperature),
            "pressure": float(self.pressure),
        }
