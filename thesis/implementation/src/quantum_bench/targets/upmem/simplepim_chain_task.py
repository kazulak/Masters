from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from quantum_bench.core.records import CircuitSpec, TensorNetworkSpec, TensorSpec, TensorValue, TaskGraph
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.tn.task_graph import plan_task_graph


CHAIN_LENGTH = 256
CHAIN_TILE_LENGTH = 64
CHAIN_CASE_ID = "simplepim_two_task_chain_fixture"
CHAIN_INPUT_IDS = ("chain_a", "chain_b", "chain_c")
CHAIN_INTERMEDIATE_ID = "result_0"
CHAIN_OUTPUT_ID = "result_1"


@dataclass(frozen=True)
class ChainTile:
    tile_id: int
    start: int
    stop: int


@dataclass(frozen=True)
class SimplePimChainWorkload:
    network: TensorNetworkValue
    graph: TaskGraph
    operands: tuple[np.ndarray, np.ndarray, np.ndarray]
    reference_int64: int
    tiles: tuple[ChainTile, ...]
    operand_sha256: str


def _operands() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.arange(CHAIN_LENGTH, dtype=np.int16)
    return (
        ((values * 3) % 31 - 15).astype(np.int8),
        ((values * 5) % 29 - 14).astype(np.int8),
        ((values * 7) % 23 - 11).astype(np.int8),
    )


def build_simplepim_chain_workload() -> SimplePimChainWorkload:
    operands = _operands()
    circuit = CircuitSpec(
        name=CHAIN_CASE_ID,
        n_qubits=0,
        operations=(),
        source={"kind": "taskgraph_fixture", "workload_type": "simplepim_two_task_chain", "developer_only": True},
    )
    specs = tuple(
        TensorSpec(tensor_id, (0,), (CHAIN_LENGTH,), "dense", dtype="int8")
        for tensor_id in CHAIN_INPUT_IDS
    )
    network_spec = TensorNetworkSpec(
        circuit=circuit,
        tensors=specs,
        output_labels=(),
        einsum_expression="a,a,a->",
    )
    network = TensorNetworkValue(
        spec=network_spec,
        tensors=[TensorValue(spec, array.copy()) for spec, array in zip(specs, operands, strict=True)],
    )
    graph = plan_task_graph(network, optimize="greedy")
    tiles = tuple(
        ChainTile(tile_id, start, start + CHAIN_TILE_LENGTH)
        for tile_id, start in enumerate(range(0, CHAIN_LENGTH, CHAIN_TILE_LENGTH))
    )
    operand_bytes = b"".join(array.tobytes() for array in operands)
    return SimplePimChainWorkload(
        network=network,
        graph=graph,
        operands=operands,
        reference_int64=int(np.sum(operands[0].astype(np.int64) * operands[1] * operands[2], dtype=np.int64)),
        tiles=tiles,
        operand_sha256=hashlib.sha256(operand_bytes).hexdigest(),
    )


def validate_simplepim_chain_workload(workload: SimplePimChainWorkload) -> None:
    graph = workload.graph
    if graph.network != workload.network.spec or len(graph.tasks) != 2:
        raise ValueError("simplepim_chain_graph_shape_mismatch")
    if graph.path != ((0, 1), (0, 1)):
        raise ValueError("simplepim_chain_path_mismatch")
    first, second = graph.tasks
    if first.input_tensor_ids != CHAIN_INPUT_IDS[:2] or first.output_tensor_id != CHAIN_INTERMEDIATE_ID:
        raise ValueError("simplepim_chain_first_task_mismatch")
    if second.dependencies != (first.id,) or second.output_tensor_id != CHAIN_OUTPUT_ID:
        raise ValueError("simplepim_chain_dependency_mismatch")
    if set(second.input_tensor_ids) != {CHAIN_INPUT_IDS[2], CHAIN_INTERMEDIATE_ID}:
        raise ValueError("simplepim_chain_second_inputs_mismatch")
    if first.input_shapes != ((CHAIN_LENGTH,), (CHAIN_LENGTH,)) or first.output_shape != (CHAIN_LENGTH,):
        raise ValueError("simplepim_chain_intermediate_shape_mismatch")
    if second.input_shapes != ((CHAIN_LENGTH,), (CHAIN_LENGTH,)) or second.output_shape != ():
        raise ValueError("simplepim_chain_output_shape_mismatch")
    if any(array.dtype != np.dtype(np.int8) or array.shape != (CHAIN_LENGTH,) for array in workload.operands):
        raise ValueError("simplepim_chain_operand_contract_mismatch")
    if len(workload.tiles) != 4 or any(tile.stop - tile.start != CHAIN_TILE_LENGTH for tile in workload.tiles):
        raise ValueError("simplepim_chain_tile_contract_mismatch")
    expected = int(np.sum(workload.operands[0].astype(np.int64) * workload.operands[1] * workload.operands[2], dtype=np.int64))
    if workload.reference_int64 != expected:
        raise ValueError("simplepim_chain_reference_mismatch")

