from .low_fidelity import LowFidelitySimulator
from .target_fidelity import TargetFidelitySimulator


def make_simulator(fidelity):
    if fidelity == "source":
        return LowFidelitySimulator()
    if fidelity == "target":
        return TargetFidelitySimulator()
    raise ValueError(f"unknown fidelity: {fidelity}")
