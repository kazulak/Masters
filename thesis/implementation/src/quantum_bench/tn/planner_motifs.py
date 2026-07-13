"""Small real-valued tensor-network motifs for planner diagnostics.

These are deliberately *not* quantum-circuit workloads. They give the path
planner a compact set of controlled graph shapes where arithmetic work,
intermediate size, and tiled local movement can pull in different directions.
They are usable only by the modeled ``compare-planners`` command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quantum_bench.core.records import CircuitSpec, JsonDict, TensorNetworkSpec, TensorSpec, TensorValue
from quantum_bench.tn.network import TensorNetworkValue, build_full_einsum_expression


PLANNER_MOTIF_KIND = "planner_motif"
PLANNER_MOTIF_WORKLOAD_TYPE = "synthetic_planner_motif"
SUPPORTED_PLANNER_MOTIFS = frozenset(
    {
        "chain",
        "balanced_tree",
        "star",
        "cycle",
        "grid",
        "flop_memory_tradeoff",
    }
)


@dataclass(frozen=True)
class PlannerMotifWorkload:
    """A modeled-only graph and the metadata needed to identify it honestly."""

    circuit: CircuitSpec
    network: TensorNetworkValue
    metadata: JsonDict


def is_planner_motif_case(case: dict[str, Any]) -> bool:
    circuit = case.get("circuit")
    return isinstance(circuit, dict) and circuit.get("kind") == PLANNER_MOTIF_KIND


def require_planner_motif_metadata(case: dict[str, Any]) -> None:
    metadata = case.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("planner_motif workloads must define metadata")
    if metadata.get("workload_type") != PLANNER_MOTIF_WORKLOAD_TYPE:
        raise ValueError("planner_motif metadata.workload_type must be synthetic_planner_motif")
    if metadata.get("execution_scope") != "model_only":
        raise ValueError("planner_motif metadata.execution_scope must be model_only")
    if metadata.get("not_real_quantum_circuit") is not True:
        raise ValueError("planner_motif metadata.not_real_quantum_circuit must be true")


def build_planner_motif_workload(case: dict[str, Any], root_dir: Path | None = None) -> PlannerMotifWorkload:
    """Construct a deterministic real-valued tensor-network motif.

    ``root_dir`` is accepted to match ordinary workload loading, although
    motifs are completely self-contained and do not read external files.
    """
    del root_dir
    if not is_planner_motif_case(case):
        raise ValueError("case is not a planner_motif workload")
    require_planner_motif_metadata(case)
    circuit_payload = dict(case["circuit"])
    motif_name = str(circuit_payload.get("name") or "")
    if motif_name not in SUPPORTED_PLANNER_MOTIFS:
        raise ValueError(f"Unsupported planner_motif: {motif_name}")

    nodes, dimensions = _motif_definition(motif_name)
    edge_labels = {edge: index for index, edge in enumerate(sorted(dimensions))}
    tensors: list[TensorValue] = []
    for node_index, (node_name, edges) in enumerate(nodes):
        labels = tuple(edge_labels[edge] for edge in edges)
        shape = tuple(int(dimensions[edge]) for edge in edges)
        spec = TensorSpec(
            id=f"{motif_name}_{node_name}",
            labels=labels,
            shape=shape,
            structure="planner_motif",
            dtype="float64",
        )
        tensors.append(TensorValue(spec, _deterministic_real_array(shape, node_index)))

    circuit = CircuitSpec(
        name=f"planner_motif_{motif_name}",
        n_qubits=0,
        operations=(),
        source={
            "kind": PLANNER_MOTIF_KIND,
            "name": motif_name,
            "workload_type": PLANNER_MOTIF_WORKLOAD_TYPE,
            "execution_scope": "model_only",
            "not_real_quantum_circuit": True,
            "purpose": "controlled_contraction_path_tradeoff",
        },
    )
    specs = tuple(tensor.spec for tensor in tensors)
    network_spec = TensorNetworkSpec(
        circuit=circuit,
        tensors=specs,
        output_labels=(),
        einsum_expression=build_full_einsum_expression(list(specs), ()),
    )
    metadata: JsonDict = {
        "workload_kind": PLANNER_MOTIF_KIND,
        "workload_type": PLANNER_MOTIF_WORKLOAD_TYPE,
        "execution_scope": "model_only",
        "not_real_quantum_circuit": True,
        "planner_motif": motif_name,
        "network_tensor_count": len(specs),
        "network_index_count": len(edge_labels),
        "network_max_rank": max(len(spec.labels) for spec in specs),
        "network_max_tensor_elements": max(int(np.prod(spec.shape, dtype=np.int64)) for spec in specs),
        "network_size_proxy": len(specs),
    }
    return PlannerMotifWorkload(circuit, TensorNetworkValue(network_spec, tensors), metadata)


def _motif_definition(name: str) -> tuple[list[tuple[str, tuple[str, ...]]], dict[str, int]]:
    """Return node incidences and deliberately non-uniform edge dimensions."""
    definitions: dict[str, tuple[list[tuple[str, tuple[str, ...]]], dict[str, int]]] = {
        "chain": (
            [
                ("n0", ("e0",)),
                ("n1", ("e0", "e1")),
                ("n2", ("e1", "e2")),
                ("n3", ("e2", "e3")),
                ("n4", ("e3",)),
            ],
            {"e0": 4, "e1": 32, "e2": 2, "e3": 16},
        ),
        "balanced_tree": (
            [
                ("leaf0", ("e0",)),
                ("leaf1", ("e1",)),
                ("left", ("e0", "e1", "e4")),
                ("leaf2", ("e2",)),
                ("leaf3", ("e3",)),
                ("right", ("e2", "e3", "e5")),
                ("root", ("e4", "e5")),
            ],
            {"e0": 4, "e1": 16, "e2": 8, "e3": 2, "e4": 32, "e5": 4},
        ),
        "star": (
            [
                ("center", ("e0", "e1", "e2", "e3")),
                ("leaf0", ("e0",)),
                ("leaf1", ("e1",)),
                ("leaf2", ("e2",)),
                ("leaf3", ("e3",)),
            ],
            {"e0": 2, "e1": 8, "e2": 16, "e3": 4},
        ),
        "cycle": (
            [
                ("n0", ("e0", "e3")),
                ("n1", ("e0", "e1")),
                ("n2", ("e1", "e2")),
                ("n3", ("e2", "e3")),
            ],
            {"e0": 2, "e1": 32, "e2": 4, "e3": 16},
        ),
        "grid": (
            [
                ("n00", ("h0", "v0")),
                ("n01", ("h0", "h1", "v1")),
                ("n02", ("h1", "v2")),
                ("n10", ("v0", "g0")),
                ("n11", ("v1", "g0", "g1")),
                ("n12", ("v2", "g1")),
            ],
            {"h0": 4, "h1": 16, "v0": 2, "v1": 8, "v2": 4, "g0": 32, "g1": 2},
        ),
        "flop_memory_tradeoff": (
            [
                ("a", ("p", "q")),
                ("b", ("q", "r", "u")),
                ("c", ("r", "s")),
                ("d", ("s", "p", "v")),
                ("u_leaf", ("u",)),
                ("v_leaf", ("v",)),
            ],
            {"p": 64, "q": 2, "r": 32, "s": 4, "u": 2, "v": 16},
        ),
    }
    return definitions[name]


def _deterministic_real_array(shape: tuple[int, ...], node_index: int) -> np.ndarray:
    element_count = int(np.prod(shape, dtype=np.int64))
    values = np.arange(element_count, dtype=np.float64).reshape(shape)
    # Avoid all-ones tensors while keeping the result deterministic and benign.
    return ((values + 1.0 + node_index) % 17.0 + 1.0) / 17.0


__all__ = [
    "PLANNER_MOTIF_KIND",
    "PLANNER_MOTIF_WORKLOAD_TYPE",
    "SUPPORTED_PLANNER_MOTIFS",
    "PlannerMotifWorkload",
    "build_planner_motif_workload",
    "is_planner_motif_case",
    "require_planner_motif_metadata",
]
