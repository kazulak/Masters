"""Functional adapter from ``ContractionDAG`` to the existing M5 v4 engine.

The adapter owns orchestration only.  The native ABI, tiling, quantization,
placement, and persistent session implementation remain in the M5 modules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from quantum_bench.execution.contracts import (
    BackendFacts,
    ExecutionPlan,
    ExecutionResult,
    NumericMode,
    RunContext,
    Target,
    TimingBreakdown,
    UpmemPlan,
    UpmemRuntimeResources,
    canonical_serialize,
    validate_execution_plan,
    validate_execution_result,
    validate_transfer_bytes,
    validate_upmem_runtime_resources,
)
from quantum_bench.execution.compiler import validate_upmem_plan_for_dag
from quantum_bench.tn.graph import (
    ContractNode,
    ContractionDAG,
    ReduceNode,
    TensorView,
    contraction_dag_hash,
    validate_contraction_dag,
)
from quantum_bench.tn.network import TensorInputs, tensor_input_map, validate_dag_inputs


@dataclass
class _Aggregate:
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    host_quantization_s: float = 0.0
    h2d_s: float = 0.0
    kernel_s: float = 0.0
    d2h_s: float = 0.0
    host_dequantization_s: float = 0.0
    reduction_s: float = 0.0
    route_total_s: float = 0.0
    physical_plan_consumed: bool = False

    def add(self, result: Any) -> None:
        metadata = getattr(result, "metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        timing = metadata.get("timing", {})
        if not isinstance(timing, Mapping):
            timing = {}
        if metadata.get("physical_plan_consumed") is not True:
            raise RuntimeError("UPMEM task result did not consume the compiled physical plan")
        self.physical_plan_consumed = True
        self.h2d_bytes += _required_byte_count(
            metadata,
            "application_visible_h2d_bytes",
            "h2d_bytes",
        )
        self.d2h_bytes += _required_byte_count(
            metadata,
            "application_visible_d2h_bytes",
            "d2h_bytes",
        )
        self.host_quantization_s += _seconds(
            timing.get("host_quantization_time_s", metadata.get("host_quantization_time_s", 0.0))
        )
        self.h2d_s += _seconds(timing.get("h2d_time_s", 0.0))
        self.kernel_s += _seconds(timing.get("kernel_time_s", 0.0))
        self.d2h_s += _seconds(timing.get("d2h_time_s", 0.0))
        if metadata.get("host_dequantization_timing_overlap") is not True:
            self.host_dequantization_s += _seconds(
                timing.get(
                    "host_dequantization_time_s",
                    metadata.get("host_dequantization_time_s", 0.0),
                )
            )
        self.reduction_s += _seconds(
            timing.get(
                "host_tile_assembly_time_s",
                metadata.get("host_tile_assembly_time_s", 0.0),
            )
        )
        if metadata.get("target_observed") == "sdk_simulator" or bool(
            metadata.get("simulator_kernel_executed", False)
        ):
            raise RuntimeError("UPMEM adapter refuses simulator execution")
        if bool(metadata.get("cpu_fallback_used", False)):
            raise RuntimeError("UPMEM adapter refuses CPU fallback execution")


def run_upmem(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    inputs: TensorInputs,
    context: RunContext,
) -> ExecutionResult:
    """Execute a compiled M5 plan with one persistent session and no fallback.

    Deterministic unsupported dispatch is represented by ``ExecutionFailure``
    at the public dispatcher. Malformed inputs and native/session failures
    raise so their original failure stage remains available to the experiment
    orchestrator.
    """

    if len({value.tensor_id for value in inputs.values}) != len(inputs.values):
        raise ValueError("Tensor inputs contain duplicate tensor ids")
    tensors = tensor_input_map(inputs)
    _validate_invocation(plan, dag, tensors, context)
    upmem_plan = plan.payload
    assert isinstance(upmem_plan, UpmemPlan)
    resources = context.target_resources
    assert resources is not None
    resource_hashes = _validate_resources(resources)
    validate_upmem_runtime_resources(resources, upmem_plan.topology)
    aggregate = _Aggregate()
    output: np.ndarray | None = None
    output_digest: str | None = None
    session: Any | None = None
    terminal_metadata: Mapping[str, Any] | None = None
    completed_node_ids: tuple[str, ...] = ()
    execution_error: BaseException | None = None
    close_error: BaseException | None = None
    session_open_s = 0.0
    session_close_s = 0.0
    try:
        open_started = time.perf_counter()
        session = _open_session(plan, context)
        session_open_s = time.perf_counter() - open_started
        # Warmups run on the persistent session but are deliberately outside
        # route_total_s.  The measured route is the sum of session lifecycle
        # and measured repetitions, including host DAG and reduction work.
        aggregate.route_total_s += session_open_s
        for _ in range(context.warmups):
            _execute_once(
                session,
                dag,
                tensors,
                upmem_plan,
                resources=None,
                aggregate=None,
            )
        for _ in range(context.repetitions):
            route_started = time.perf_counter()
            output, completed_node_ids = _execute_once(
                session,
                dag,
                tensors,
                upmem_plan,
                resources=resources,
                aggregate=aggregate,
            )
            aggregate.route_total_s += time.perf_counter() - route_started
            # This reproducibility check is intentionally outside route timing.
            digest = _array_hash(output)
            if output_digest is None:
                output_digest = digest
            elif digest != output_digest:
                raise RuntimeError("UPMEM execution produced non-deterministic output")
    except BaseException as exc:
        execution_error = exc
    finally:
        if session is not None:
            close_started = time.perf_counter()
            try:
                terminal_metadata = session.close()
            except BaseException as exc:
                close_error = exc
            session_close_s = time.perf_counter() - close_started
            aggregate.route_total_s += session_close_s

    if execution_error is not None and close_error is not None:
        raise RuntimeError(
            f"UPMEM execution failed: {execution_error}; "
            f"session close failed: {close_error}"
        ) from execution_error
    if execution_error is not None:
        raise execution_error
    if close_error is not None:
        raise RuntimeError("UPMEM session close failed") from close_error
    _validate_terminal_metadata(terminal_metadata, upmem_plan)

    if output is None or output_digest is None:
        raise RuntimeError("UPMEM execution did not produce an output")
    h2d = aggregate.h2d_bytes
    d2h = aggregate.d2h_bytes
    transfer = h2d + d2h
    validate_transfer_bytes(h2d, d2h, transfer)
    facts = replace(
        _facts_from_metadata(terminal_metadata),
        **resource_hashes,
        rank_binding_sha256=_rank_binding_sha256(resources.rank_paths),
        physical_plan_consumed=aggregate.physical_plan_consumed,
    )
    result = ExecutionResult(
        contraction_dag_hash=contraction_dag_hash(dag),
        target=Target.UPMEM,
        output=np.array(output, copy=True),
        executed_node_ids=completed_node_ids,
        timing=TimingBreakdown(
            host_quantization_s=aggregate.host_quantization_s or None,
            h2d_s=aggregate.h2d_s or None,
            kernel_s=aggregate.kernel_s or None,
            d2h_s=aggregate.d2h_s or None,
            host_dequantization_s=aggregate.host_dequantization_s or None,
            reduction_s=aggregate.reduction_s or None,
            session_open_s=session_open_s or None,
            session_close_s=session_close_s or None,
            route_total_s=aggregate.route_total_s or None,
        ),
        h2d_bytes=h2d,
        d2h_bytes=d2h,
        transfer_bytes=transfer,
        output_hash=output_digest,
        backend_facts=facts,
    )
    validate_execution_result(result)
    return result


def _open_session(plan: ExecutionPlan, context: RunContext) -> Any:
    """Create the real M5 session; tests replace this single seam."""

    from quantum_bench.targets.upmem.m5_whole_circuit_engine import (
        M5WholeCircuitEngine,
    )
    from quantum_bench.whole_circuit.core import DeviceTopology
    from quantum_bench.whole_circuit.policies import (
        Float32RealPolicy,
        HostPackedInt8Policy,
    )

    payload = plan.payload
    assert isinstance(payload, UpmemPlan)
    resources = context.target_resources
    assert resources is not None
    if resources.session_opener is not None:
        return resources.session_opener(plan, context)
    timeout_s = 60.0 if context.timeout_s is None else context.timeout_s
    engine = M5WholeCircuitEngine(
        session_root=Path(resources.session_root),
        host_binary=Path(resources.host_binary),
        dpu_binary=Path(resources.dpu_binary),
        initialization_binary=Path(resources.initialization_binary),
        rank_paths=resources.rank_paths,
        dpu_count=payload.topology.dpu_count,
        tasklets_per_dpu=payload.topology.tasklets_per_dpu,
        timeout_s=timeout_s,
    )
    policy = (
        HostPackedInt8Policy()
        if payload.numeric_mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
        else Float32RealPolicy()
    )
    topology = DeviceTopology(
        backend="upmem",
        device_ids=tuple(f"dpu_{index}" for index in range(payload.topology.dpu_count)),
        tasklets_per_device=payload.topology.tasklets_per_dpu,
    )
    return engine.open_session(policy, topology)


def _execute_once(
    session: Any,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    plan: UpmemPlan,
    *,
    resources: UpmemRuntimeResources | None,
    aggregate: _Aggregate | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    tensors = dict(inputs)
    nodes = {node.node_id: node for node in dag.nodes}
    remaining_consumers = _remaining_consumers(dag)
    produced_tensor_ids = {node.output.id for node in dag.nodes}
    completed_node_ids: list[str] = []
    for node_plan in plan.node_plans:
        node_id = node_plan.node_id
        node = nodes[node_id]
        if isinstance(node, ContractNode):
            left = _resolve_view(node.left, tensors)
            right = _resolve_view(node.right, tensors)
            task_result = session.execute(node, left, right, node_plan=node_plan)
            if aggregate is not None:
                assert resources is not None
                aggregate.add(task_result)
            value = np.asarray(task_result.output)
        elif isinstance(node, ReduceNode):
            reduction_started = time.perf_counter()
            value = np.sum(
                np.stack([_resolve_view(view, tensors) for view in node.inputs], axis=0),
                axis=0,
            )
            if aggregate is not None:
                aggregate.reduction_s += time.perf_counter() - reduction_started
        else:  # pragma: no cover - graph validation closes this union
            raise TypeError(f"unsupported UPMEM DAG node: {type(node).__name__}")
        if tuple(value.shape) != node.output.shape:
            raise ValueError(
                f"UPMEM node {node_id} produced shape {value.shape}; expected {node.output.shape}"
            )
        tensors[node.output.id] = value
        if (
            remaining_consumers.get(node.output.id, 0) == 0
            and node.output.id != dag.output.tensor_id
        ):
            tensors.pop(node.output.id, None)
        for tensor_id in _node_input_tensor_ids(node):
            remaining_consumers[tensor_id] -= 1
            if (
                remaining_consumers[tensor_id] == 0
                and tensor_id in produced_tensor_ids
                and tensor_id != dag.output.tensor_id
            ):
                tensors.pop(tensor_id, None)
        # The host coordinator has validated and published this node output
        # for its dependants. This does not claim native-kernel exactly once.
        completed_node_ids.append(node_id)
    return _resolve_view(dag.output, tensors), tuple(completed_node_ids)


def _validate_invocation(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    context: RunContext,
) -> None:
    validate_contraction_dag(dag)
    validate_execution_plan(plan)
    if plan.target is not Target.UPMEM or not isinstance(plan.payload, UpmemPlan):
        raise ValueError("run_upmem requires an UPMEM execution plan")
    if context.target is not Target.UPMEM:
        raise ValueError("run_upmem requires an UPMEM RunContext")
    if context.warmups < 0 or context.repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    if context.timeout_s is not None and (
        context.timeout_s <= 0 or not math.isfinite(context.timeout_s)
    ):
        raise ValueError("timeout_s must be finite and positive when provided")
    actual_hash = contraction_dag_hash(dag)
    if plan.contraction_dag_hash != actual_hash:
        raise ValueError("execution plan hash does not match supplied DAG")
    if plan.payload.numeric_mode in {
        NumericMode.FLOAT32_REAL,
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
    }:
        for value in inputs.values():
            array = np.asarray(value)
            if np.iscomplexobj(array) and np.any(np.imag(array) != 0):
                raise ValueError(
                    "M5 real-valued UPMEM numeric modes reject nonzero imaginary inputs"
                )
    validate_dag_inputs(dag, inputs)
    validate_upmem_plan_for_dag(dag, plan.payload)
    if context.target_resources is None:
        raise ValueError("UPMEM runtime resources are required")
    _validate_m5_plan(plan.payload, dag, context.target_resources)


def _validate_m5_plan(
    plan: UpmemPlan, dag: ContractionDAG, resources: UpmemRuntimeResources | None
) -> None:
    expected = (
        ("profile_id", "m5_whole_circuit_v4_v1"),
        ("abi_id", "execution_plan_v4"),
        ("session_id", "persistent_rank_session_v1"),
        ("dispatch_id", "bulk_set_synchronous_v1"),
        ("kernel_id", "dpu_gemm_tile_v4"),
        ("decomposition_id", "m5_v4_tile_decomposition"),
        ("placement_id", "m5_rank_wave_placement"),
        ("reduction_id", "m5_tile_host_reduction"),
    )
    for field, expected_value in expected:
        if getattr(plan, field) != expected_value:
            raise ValueError(f"unsupported M5 UPMEM {field}: {getattr(plan, field)!r}")
    topology = plan.topology
    if resources is None:
        raise ValueError("UPMEM runtime resources are required")
    validate_upmem_runtime_resources(resources, topology)
    if topology.dpu_count // topology.rank_count > 64:
        raise ValueError("UPMEM plan exceeds 64 DPUs per rank")
    if not 1 <= topology.tasklets_per_dpu <= 24:
        raise ValueError("UPMEM plan tasklets_per_dpu must be in [1, 24]")


def _node_input_tensor_ids(node: ContractNode | ReduceNode) -> tuple[str, ...]:
    if isinstance(node, ContractNode):
        return (node.left.tensor_id, node.right.tensor_id)
    return tuple(view.tensor_id for view in node.inputs)


def _remaining_consumers(dag: ContractionDAG) -> dict[str, int]:
    remaining: dict[str, int] = {}
    for node in dag.nodes:
        for tensor_id in _node_input_tensor_ids(node):
            remaining[tensor_id] = remaining.get(tensor_id, 0) + 1
    return remaining


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_resources(resources: UpmemRuntimeResources) -> dict[str, str]:
    paths = {
        "host_binary": Path(resources.host_binary),
        "dpu_binary": Path(resources.dpu_binary),
        "initialization_binary": Path(resources.initialization_binary),
    }
    if not resources.session_root:
        raise ValueError("UPMEM session_root must be non-empty")
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"UPMEM {label} is not a regular file: {path}")
        if label == "host_binary" and not os.access(path, os.X_OK):
            raise ValueError("UPMEM host_binary is not executable")
    return {
        "host_binary_sha256": _file_sha256(paths["host_binary"]),
        "dpu_binary_sha256": _file_sha256(paths["dpu_binary"]),
        "initialization_binary_sha256": _file_sha256(paths["initialization_binary"]),
    }


def _facts_from_metadata(
    metadata: Mapping[str, Any],
) -> BackendFacts:
    return BackendFacts(
        backend_id=_observed_text(metadata, "backend_id"),
        profile_id=_observed_text(metadata, "profile", "physical_profile"),
        abi_id=_observed_text(metadata, "abi", "abi_version"),
        session_id=_observed_text(metadata, "session_protocol"),
        dispatch_id=_observed_text(metadata, "dispatch_mode"),
        kernel_id=_observed_text(metadata, "kernel_identity"),
        execution_class=_observed_text(metadata, "execution_class"),
        intermediate_placement=_observed_text(
            metadata, "graph_intermediate_placement"
        ),
        intermediate_placement_origin=_observed_text(
            metadata, "graph_intermediate_placement_origin"
        ),
        native_identity_verified=bool(metadata.get("native_identity_verified", False)),
        target_observed=metadata.get("target_observed"),
        hardware_allocation_verified=bool(
            metadata.get("hardware_allocation_verified", False)
        ),
        hardware_release_verified=bool(
            metadata.get("hardware_release_verified", False)
        ),
        hardware_release_confirmed=bool(
            metadata.get("hardware_release_confirmed", False)
        ),
        requested_dpu_count=_optional_int(metadata.get("requested_dpu_count")),
        allocated_dpu_count=_optional_int(metadata.get("allocated_dpu_count")),
        observed_rank_count=_optional_int(metadata.get("observed_rank_count")),
        tasklets_per_dpu=_optional_int(
            metadata.get("observed_tasklets_per_dpu", metadata.get("tasklets_per_dpu"))
        ),
        native_kernel_executed=bool(metadata.get("native_kernel_executed", False)),
        hardware_kernel_executed=bool(metadata.get("hardware_kernel_executed", False)),
        simulator_kernel_executed=bool(metadata.get("simulator_kernel_executed", False)),
        cpu_fallback_used=bool(metadata.get("cpu_fallback_used", False)),
        physical_plan_consumed=bool(metadata.get("physical_plan_consumed", False)),
    )


def _validate_terminal_metadata(
    metadata: Mapping[str, Any] | None, plan: UpmemPlan
) -> None:
    """Admit a result only when the terminal M5 close contract is complete."""

    if not isinstance(metadata, Mapping):
        raise RuntimeError("UPMEM session close returned no terminal metadata")
    required = {
        "target_observed": "physical_hardware",
        "hardware_allocation_verified": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_release_verified": True,
        "hardware_release_confirmed": True,
        "native_identity_verified": True,
        "failure_stage": None,
        "requested_dpu_count": plan.topology.dpu_count,
        "allocated_dpu_count": plan.topology.dpu_count,
        "observed_rank_count": plan.topology.rank_count,
        "observed_tasklets_per_dpu": plan.topology.tasklets_per_dpu,
        "session_protocol": plan.session_id,
        "dispatch_mode": plan.dispatch_id,
        "kernel_identity": plan.kernel_id,
        "execution_class": "physical_v4_output_tile",
        "graph_intermediate_placement": "host_managed",
        "graph_intermediate_placement_origin": "m5_host_coordinator_v1",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"UPMEM terminal metadata is not physically verified: "
                f"{key}={metadata.get(key)!r}"
            )

    aliases = {
        "profile": ("physical_profile",),
        "abi": ("abi_version",),
    }
    for canonical, alternatives in aliases.items():
        observed = _observed_text(metadata, canonical, *alternatives)
        expected = plan.profile_id if canonical == "profile" else plan.abi_id
        if observed != expected:
            raise RuntimeError(
                "UPMEM terminal metadata is not physically verified: "
                f"{canonical}={observed!r}"
            )


def _observed_text(metadata: Mapping[str, Any], *keys: str) -> str:
    """Read one observed terminal value, rejecting absent or conflicting aliases."""

    values = [str(metadata[key]) for key in keys if metadata.get(key) is not None]
    if not values:
        raise RuntimeError(f"UPMEM terminal metadata is missing {keys[0]}")
    if len(set(values)) != 1:
        raise RuntimeError(f"UPMEM terminal metadata has conflicting {keys[0]} values")
    return values[0]


def _required_byte_count(metadata: Mapping[str, Any], *keys: str) -> int:
    """Return an observed application-visible byte count, never a guessed zero."""

    values = [metadata[key] for key in keys if key in metadata]
    if not values:
        raise RuntimeError(f"UPMEM task metadata is missing {keys[0]}")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise RuntimeError(f"UPMEM task metadata has invalid {keys[0]}")
    parsed = [int(value) for value in values]
    if len(set(parsed)) != 1:
        raise RuntimeError(f"UPMEM task metadata has conflicting {keys[0]} values")
    return parsed[0]


def _rank_binding_sha256(rank_paths: tuple[str, ...]) -> str:
    """Hash ordered runtime rank bindings without exposing their raw paths."""

    return hashlib.sha256(
        canonical_serialize(tuple(rank_paths)).encode("utf-8")
    ).hexdigest()


def _resolve_view(view: TensorView, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
    if view.tensor_id not in tensors:
        raise ValueError(f"UPMEM tensor {view.tensor_id} is not available")
    value = tensors[view.tensor_id]
    if not view.slice_spec:
        return value
    indices: list[slice | int] = [slice(None)] * value.ndim
    for axis, index in view.slice_spec:
        indices[axis] = index
    sliced = value[tuple(indices)]
    if tuple(sliced.shape) != view.shape:
        raise ValueError(f"UPMEM sliced tensor {view.tensor_id} has wrong shape")
    return sliced


def _topological_order(dag: ContractionDAG) -> tuple[str, ...]:
    nodes = {node.node_id: node for node in dag.nodes}
    remaining = {node_id: len(node.dependencies) for node_id, node in nodes.items()}
    dependents: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node in dag.nodes:
        for dependency in node.dependencies:
            dependents[dependency].append(node.node_id)
    ready = sorted(node_id for node_id, count in remaining.items() if count == 0)
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for dependent in sorted(dependents[node_id]):
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
        ready.sort()
    if len(order) != len(nodes):
        raise ValueError("UPMEM DAG cannot be topologically ordered")
    return tuple(order)


def _seconds(value: Any) -> float:
    result = float(value or 0.0)
    if result < 0 or not math.isfinite(result):
        raise ValueError("UPMEM timing values must be finite and non-negative")
    return result


def _nonnegative_int(value: Any) -> int:
    result = int(value or 0)
    if result < 0:
        raise ValueError("UPMEM byte values must be non-negative")
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise ValueError("UPMEM count values must be non-negative")
    return result


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["run_upmem"]
