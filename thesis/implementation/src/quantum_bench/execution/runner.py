"""Explicit execution dispatch for the functional execution slice."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from quantum_bench.execution.contracts import (
    BackendFacts,
    ExecutionFailure,
    ExecutionPlan,
    ExecutionResult,
    NumericMode,
    RunContext,
    Target,
    TimingBreakdown,
    UpmemNodePlan,
    UpmemPlan,
    UpmemRuntimeResources,
    UpmemTopology as LegacyUpmemTopology,
    UpmemWorkUnit as LegacyUpmemWorkUnit,
    validate_execution_plan,
    validate_execution_result,
    validate_transfer_bytes,
    validate_upmem_runtime_resources,
)
from quantum_bench.execution.cpu import run_cpu
from quantum_bench.execution.compiler import (
    validate_active_upmem_plan,
    validate_upmem_plan_for_dag,
)
from quantum_bench.lowering import (
    contraction_dag_hash,
    validate_contraction_dag,
    validate_dag_inputs,
)
from quantum_bench.upmem.runtime import UpmemV4Executor, UpmemV4Session
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode, TensorView
from quantum_bench.numerics import NumericPolicy
from quantum_bench.upmem.plan import (
    UpmemStage,
    UpmemTopology as FinalUpmemTopology,
    UpmemWorkUnit as FinalUpmemWorkUnit,
)


def _observed_text(metadata: Mapping[str, Any], *keys: str) -> str:
    values = [str(metadata[key]) for key in keys if metadata.get(key) is not None]
    if not values:
        raise RuntimeError(f"UPMEM terminal metadata is missing {keys[0]}")
    if len(set(values)) != 1:
        raise RuntimeError(f"UPMEM terminal metadata has conflicting {keys[0]} values")
    return values[0]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise ValueError("UPMEM count values must be non-negative")
    return result


class _Aggregate:
    def __init__(self) -> None:
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.host_quantization_s = 0.0
        self.preparation_s = 0.0
        self.h2d_s = 0.0
        self.kernel_s = 0.0
        self.d2h_s = 0.0
        self.host_dequantization_s = 0.0
        self.reduction_s = 0.0
        self.route_total_s = 0.0
        self.physical_plan_consumed = False

    def add(self, result: tuple[np.ndarray, Mapping[str, Any]]) -> None:
        _, metadata = result
        timing = metadata.get("timing", {})
        if metadata.get("physical_plan_consumed") is not True:
            raise RuntimeError(
                "UPMEM task result did not consume the compiled physical plan"
            )
        self.physical_plan_consumed = True
        self.h2d_bytes += _required_byte_count(
            metadata, "application_visible_h2d_bytes", "h2d_bytes"
        )
        self.d2h_bytes += _required_byte_count(
            metadata, "application_visible_d2h_bytes", "d2h_bytes"
        )
        self.host_quantization_s += _seconds(
            timing.get(
                "host_quantization_time_s",
                metadata.get("host_quantization_time_s", 0.0),
            )
        )
        self.preparation_s += _seconds(
            timing.get("preparation_time_s", metadata.get("preparation_time_s", 0.0))
        )
        self.h2d_s += _seconds(timing.get("h2d_time_s", 0.0))
        self.kernel_s += _seconds(timing.get("kernel_time_s", 0.0))
        self.d2h_s += _seconds(timing.get("d2h_time_s", 0.0))
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


def _execute_once(
    session: Any,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    plan: UpmemPlan,
    *,
    resources: UpmemRuntimeResources | None,
    aggregate: _Aggregate | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    del resources
    tensors = dict(inputs)
    nodes = {node.node_id: node for node in dag.nodes}
    remaining_consumers = _remaining_consumers(dag)
    produced_tensor_ids = {node.output.id for node in dag.nodes}
    completed_node_ids: list[str] = []
    for node_plan in plan.node_plans:
        node = nodes[node_plan.node_id]
        if isinstance(node, ContractNode):
            left = _resolve_view(node.left, tensors)
            right = _resolve_view(node.right, tensors)
            if isinstance(session, UpmemV4Session):
                value, metadata = session.execute_real(
                    node,
                    left,
                    right,
                    stage=_final_stage(node_plan),
                    numeric_policy=_final_numeric_policy(plan.numeric_mode),
                )
            else:
                value, metadata = session.execute(
                    node, left, right, node_plan=node_plan
                )
            if aggregate is not None:
                aggregate.add((value, metadata))
            value = np.asarray(value)
        elif isinstance(node, ReduceNode):
            started = time.perf_counter()
            value = np.sum(
                np.stack(
                    [_resolve_view(view, tensors) for view in node.inputs], axis=0
                ),
                axis=0,
            )
            if aggregate is not None:
                aggregate.reduction_s += time.perf_counter() - started
        else:
            raise TypeError(f"unsupported UPMEM DAG node: {type(node).__name__}")
        if tuple(value.shape) != node.output.shape:
            raise ValueError(
                f"UPMEM node {node.node_id} produced shape {value.shape}; expected {node.output.shape}"
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
        completed_node_ids.append(node.node_id)
    return _resolve_view(dag.output, tensors), tuple(completed_node_ids)


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


def _required_byte_count(metadata: Mapping[str, Any], *keys: str) -> int:
    values = [metadata[key] for key in keys if key in metadata]
    if not values:
        raise RuntimeError(f"UPMEM task metadata is missing {keys[0]}")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise RuntimeError(f"UPMEM task metadata has invalid {keys[0]}")
    parsed = [int(value) for value in values]
    if len(set(parsed)) != 1:
        raise RuntimeError(f"UPMEM task metadata has conflicting {keys[0]} values")
    return parsed[0]


def _seconds(value: Any) -> float:
    result = float(value or 0.0)
    if result < 0 or not math.isfinite(result):
        raise ValueError("UPMEM timing values must be finite and non-negative")
    return result


def _rank_binding_sha256(rank_paths: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(tuple(rank_paths), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_numeric_policy(mode: NumericMode) -> NumericPolicy:
    if mode is NumericMode.FLOAT32_REAL:
        return "split_complex_float32_v1"
    if mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1:
        return "split_complex_int8_shared_scale_v1"
    raise ValueError(f"unsupported historical UPMEM numeric mode: {mode!r}")


def _final_topology(topology: LegacyUpmemTopology) -> FinalUpmemTopology:
    return FinalUpmemTopology(
        dpu_count=topology.dpu_count,
        tasklets_per_dpu=topology.tasklets_per_dpu,
        rank_count=topology.rank_count,
    )


def _final_work_unit(unit: LegacyUpmemWorkUnit) -> FinalUpmemWorkUnit:
    return FinalUpmemWorkUnit(
        node_id=unit.node_id,
        stable_tile_id=unit.stable_tile_id,
        wave=unit.wave,
        logical_rank=unit.logical_rank,
        logical_dpu=unit.logical_dpu,
        batch_start=unit.batch_start,
        batch_size=unit.batch_size,
        m_start=unit.m_start,
        m_size=unit.m_size,
        n_start=unit.n_start,
        n_size=unit.n_size,
        k_start=unit.k_start,
        k_size=unit.k_size,
        estimated_input_bytes=unit.estimated_input_bytes,
        estimated_output_bytes=unit.estimated_output_bytes,
        aligned_mram_bytes=unit.aligned_mram_bytes,
        estimated_arithmetic_work=unit.estimated_arithmetic_work,
    )


def _final_stage(node_plan: UpmemNodePlan) -> UpmemStage:
    if node_plan.node_kind != "contract":
        raise ValueError("legacy UPMEM stage conversion requires a contract node")
    return UpmemStage(
        stage_id=f"contract_batch:{node_plan.node_id}",
        kind="contract_batch",
        node_ids=(node_plan.node_id,),
        work_units=tuple(_final_work_unit(unit) for unit in node_plan.work_units),
    )


def execute(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    context: RunContext,
) -> ExecutionResult | ExecutionFailure:
    """Dispatch one compiled plan without fallback.

    ``ExecutionFailure`` represents deterministic dispatch rejection only.
    Malformed requests and native/session failures raise unchanged; experiment
    orchestration records them as failure rows with their original stage.
    """

    match plan.target:
        case Target.CPU:
            return run_cpu(plan, dag, inputs, context)
        case Target.UPMEM:
            if (
                not getattr(plan.payload, "node_order", ())
                and context.target_resources is None
            ):
                return ExecutionFailure(
                    contraction_dag_hash=plan.contraction_dag_hash,
                    target=Target.UPMEM,
                    stage="execution_dispatch",
                    reason="UPMEM execution adapter is not implemented for legacy plans",
                )
            return run_upmem(plan, dag, inputs, context)
        case Target.GPU:
            return ExecutionFailure(
                contraction_dag_hash=plan.contraction_dag_hash,
                target=Target.GPU,
                stage="execution_dispatch",
                reason="GPU execution adapter is not implemented in this slice",
            )
        case _:
            raise TypeError(f"Unsupported execution target: {plan.target!r}")


def run_upmem(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    context: RunContext,
) -> ExecutionResult:
    """Historical coordinator retained until T12 removes the generic stack."""

    tensors = {tensor_id: np.asarray(array) for tensor_id, array in inputs.items()}
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
        aggregate.route_total_s += session_open_s
        for _ in range(context.warmups):
            _execute_once(
                session, dag, tensors, upmem_plan, resources=None, aggregate=None
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
            f"UPMEM execution failed: {execution_error}; session close failed: {close_error}"
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
            preparation_s=aggregate.preparation_s or None,
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
    payload = plan.payload
    assert isinstance(payload, UpmemPlan)
    resources = context.target_resources
    assert resources is not None
    if resources.session_opener is not None:
        return resources.session_opener(plan, context)
    timeout_s = 60.0 if context.timeout_s is None else context.timeout_s
    engine = UpmemV4Executor(
        session_root=Path(resources.session_root),
        host_binary=Path(resources.host_binary),
        dpu_binary=Path(resources.dpu_binary),
        initialization_binary=Path(resources.initialization_binary),
        rank_paths=resources.rank_paths,
        dpu_count=payload.topology.dpu_count,
        tasklets_per_dpu=payload.topology.tasklets_per_dpu,
        timeout_s=timeout_s,
    )
    return engine.open_session(
        _final_numeric_policy(payload.numeric_mode),
        _final_topology(payload.topology),
    )


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
        raise ValueError(
            "warmups must be non-negative and repetitions must be positive"
        )
    if context.timeout_s is not None and (
        context.timeout_s <= 0 or not math.isfinite(context.timeout_s)
    ):
        raise ValueError("timeout_s must be finite and positive when provided")
    if plan.contraction_dag_hash != contraction_dag_hash(dag):
        raise ValueError("execution plan hash does not match supplied DAG")
    if plan.payload.numeric_mode.value in {
        "float32_real",
        "host_packed_int8_per_task_v1",
    }:
        for value in inputs.values():
            array = np.asarray(value)
            if np.iscomplexobj(array) and np.any(np.imag(array) != 0):
                raise ValueError(
                    "M5 real-valued UPMEM numeric modes reject nonzero imaginary inputs"
                )
    validate_dag_inputs(dag, inputs)
    validate_upmem_plan_for_dag(dag, plan.payload)
    validate_active_upmem_plan(plan.payload)
    if context.target_resources is None:
        raise ValueError("UPMEM runtime resources are required")
    _validate_upmem_resources(plan.payload, context.target_resources)


def _validate_upmem_resources(
    plan: UpmemPlan, resources: UpmemRuntimeResources | None
) -> None:
    if resources is None:
        raise ValueError("UPMEM runtime resources are required")
    validate_upmem_runtime_resources(resources, plan.topology)
    if plan.topology.dpu_count // plan.topology.rank_count > 64:
        raise ValueError("UPMEM plan exceeds 64 DPUs per rank")
    if not 1 <= plan.topology.tasklets_per_dpu <= 24:
        raise ValueError("UPMEM plan tasklets_per_dpu must be in [1, 24]")


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


def _facts_from_metadata(metadata: Mapping[str, Any]) -> BackendFacts:
    return BackendFacts(
        backend_id=_observed_text(metadata, "backend_id"),
        profile_id=_observed_text(metadata, "profile", "physical_profile"),
        abi_id=_observed_text(metadata, "abi", "abi_version"),
        session_id=_observed_text(metadata, "session_protocol"),
        dispatch_id=_observed_text(metadata, "dispatch_mode"),
        kernel_id=_observed_text(metadata, "kernel_identity"),
        execution_class=_observed_text(metadata, "execution_class"),
        intermediate_placement=_observed_text(metadata, "graph_intermediate_placement"),
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
        simulator_kernel_executed=bool(
            metadata.get("simulator_kernel_executed", False)
        ),
        cpu_fallback_used=bool(metadata.get("cpu_fallback_used", False)),
        physical_plan_consumed=bool(metadata.get("physical_plan_consumed", False)),
    )


def _validate_terminal_metadata(
    metadata: Mapping[str, Any] | None, plan: UpmemPlan
) -> None:
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
                f"UPMEM terminal metadata is not physically verified: {key}={metadata.get(key)!r}"
            )
    for canonical, alternatives in {
        "profile": ("physical_profile",),
        "abi": ("abi_version",),
    }.items():
        observed = _observed_text(metadata, canonical, *alternatives)
        expected = plan.profile_id if canonical == "profile" else plan.abi_id
        if observed != expected:
            raise RuntimeError(
                f"UPMEM terminal metadata is not physically verified: {canonical}={observed!r}"
            )


__all__ = ["execute", "run_upmem"]
