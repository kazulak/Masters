from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tnsim.circuits import Circuit


@dataclass
class TensorValue:
    id: str
    labels: tuple[int, ...]
    array: np.ndarray
    structure: str
    produced_by: str | None = None


@dataclass
class TensorNetwork:
    circuit: Circuit
    tensors: list[TensorValue]
    output_labels: tuple[int, ...]
    einsum_expression: str


@dataclass
class ExecutionRun:
    output: np.ndarray
    profiles: list[dict]
    execution_seconds: float
    energy_joules: float | None
    energy_source: str
    estimated_power_watts: float | None = None

