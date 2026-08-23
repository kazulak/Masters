"""Temporary historical tensor-network value adapter.

The canonical circuit-to-network lowering lives in ``quantum_bench.lowering``.
This module remains only for historical routes that still require tensors and
their values bundled together.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantum_bench.core.records import TensorValue
from quantum_bench.lowering import lower_tensor_network
from quantum_bench.model import CircuitSpec, TensorNetwork, make_simulation_job


@dataclass
class TensorNetworkValue:
    spec: TensorNetwork
    tensors: list[TensorValue]


def build_tensor_network(circuit: CircuitSpec) -> TensorNetworkValue:
    """Build the historical combined structure/value view."""

    network, inputs = lower_tensor_network(make_simulation_job(circuit))
    specs_by_id = {tensor.id: tensor for tensor in network.tensors}
    return TensorNetworkValue(
        network,
        [TensorValue(specs_by_id[tensor_id], inputs[tensor_id]) for tensor_id in inputs],
    )


def interleaved_einsum_args(network: TensorNetworkValue) -> list[object]:
    args: list[object] = []
    for tensor in network.tensors:
        args.extend([tensor.array, list(tensor.spec.labels)])
    args.append(list(network.spec.output_labels))
    return args


__all__ = ["TensorNetworkValue", "build_tensor_network", "interleaved_einsum_args"]
