"""Bounded real-float32 UPMEM execution-plan sidecar.

The Block 1 contract is intentionally small: one real resident contract
operation per logical task, one tasklet, and one or two DPUs.  Package bytes
and package validation remain owned by hardware_taskgraph_resident.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Callable, Mapping

from quantum_bench.core.records import JsonDict, TaskGraph, to_jsonable
from quantum_bench.tn.execution_bundle import canonical_hash, with_execution_identity


SCHEMA_VERSION = "upmem_execution_plan_v1"
RUNTIME_PROVIDER_ID = "simplepim_management_v1"
KERNEL_PROVIDER_ID = "thesis_resident_generic_contract_v1"
COMMUNICATION_PROVIDER_ID = "host_mediated_v1"
NUMERIC_MODE = "none"
PLACEMENT_SINGLE = "single_dpu_sequential_v1"
PLACEMENT_FRONTIER = "two_dpu_frontier_round_robin_v1"
MAX_OPERATIONS = 8
MAX_WAVES = 8
TASKLETS_PER_DPU = 1
PROVIDER_COUNT = 3

SCHEDULE_MAGIC = b"UPXPLAN1"
SCHEDULE_VERSION = 1
SCHEDULE_HEADER_FORMAT = "<8s10I32s"
SCHEDULE_RECORD_FORMAT = "<8I"
SCHEDULE_HEADER_BYTES = struct.calcsize(SCHEDULE_HEADER_FORMAT)
SCHEDULE_RECORD_BYTES = struct.calcsize(SCHEDULE_RECORD_FORMAT)


@dataclass(frozen=True)
class TaskAssignment:
    operation_id: int
    package_operation_index: int
    task_id: str
    component: str
    wave_index: int
    dpu_id: int
    dependency_operation_ids: tuple[int, ...]
    dependency_bitmask: int
    input_slot_ids: tuple[int, int]
    output_slot_id: int
    output_elements: int


@dataclass(frozen=True)
class CrossDpuTransfer:
    producer_operation_id: int
    consumer_operation_id: int
    producer_task_id: str
    consumer_task_id: str
    producer_dpu_id: int
    consumer_dpu_id: int
    slot_id: int
    element_count: int
    transfer_bytes: int


@dataclass(frozen=True)
class FinalOutput:
    component: str
    slot_id: int
    tensor_id: str
    element_count: int


@dataclass(frozen=True)
class ExecutionPlan:
    placement_policy: str
    requested_dpu_count: int
    tasklets_per_dpu: int
    package_file_sha256: str
    source_circuit_semantics_hash: str
    source_tensor_network_hash: str
    source_contraction_plan_hash: str
    package_circuit_semantics_hash: str
    package_tensor_network_hash: str
    package_contraction_plan_hash: str
    logical_task_count: int
    waves: tuple[tuple[str, ...], ...]
    assignments: tuple[TaskAssignment, ...]
    transfer_edges: tuple[CrossDpuTransfer, ...]
    final_outputs: tuple[FinalOutput, ...]
    schema_version: str = SCHEMA_VERSION
    runtime_provider_id: str = RUNTIME_PROVIDER_ID
    kernel_provider_id: str = KERNEL_PROVIDER_ID
    communication_provider_id: str = COMMUNICATION_PROVIDER_ID
    numeric_mode: str = NUMERIC_MODE

    @property
    def operation_count(self) -> int:
        return len(self.assignments)

    @property
    def wave_count(self) -> int:
        return len(self.waves)

    @property
    def host_to_dpu_bytes(self) -> int:
        return sum(item.transfer_bytes for item in self.transfer_edges)

    @property
    def dpu_to_host_bytes(self) -> int:
        return self.host_to_dpu_bytes

    @property
    def total_cross_dpu_transfer_bytes(self) -> int:
        return self.host_to_dpu_bytes + self.dpu_to_host_bytes

    @property
    def assignment_hash(self) -> str:
        return canonical_hash([
            {
                "package_operation_index": item.package_operation_index,
                "operation_id": item.operation_id,
                "wave_index": item.wave_index,
                "dpu_id": item.dpu_id,
                "dependency_bitmask": item.dependency_bitmask,
            }
            for item in self.assignments
        ])

    @property
    def execution_plan_hash(self) -> str:
        return canonical_hash(self._hash_payload())

    @property
    def schedule_sidecar_sha256(self) -> str:
        return hashlib.sha256(serialize_schedule(self)).hexdigest()

    def _hash_payload(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "placement_policy": self.placement_policy,
            "requested_dpu_count": self.requested_dpu_count,
            "tasklets_per_dpu": self.tasklets_per_dpu,
            "package_file_sha256": self.package_file_sha256,
            "providers": {
                "runtime": self.runtime_provider_id,
                "kernel": self.kernel_provider_id,
                "communication": self.communication_provider_id,
                "numeric_mode": self.numeric_mode,
            },
            "source_identity": {
                "circuit_semantics_hash": self.source_circuit_semantics_hash,
                "tensor_network_hash": self.source_tensor_network_hash,
                "contraction_plan_hash": self.source_contraction_plan_hash,
            },
            "package_identity": {
                "circuit_semantics_hash": self.package_circuit_semantics_hash,
                "tensor_network_hash": self.package_tensor_network_hash,
                "contraction_plan_hash": self.package_contraction_plan_hash,
            },
            "logical_task_count": self.logical_task_count,
            "waves": to_jsonable(self.waves),
            "task_assignments": [_assignment_json(item) for item in self.assignments],
            "transfer_edges": [_transfer_json(item) for item in self.transfer_edges],
            "final_outputs": [to_jsonable(item) for item in self.final_outputs],
        }

    def to_json(self) -> JsonDict:
        payload = self._hash_payload()
        payload.update({
            "operation_count": self.operation_count,
            "wave_count": self.wave_count,
            "provider_count": PROVIDER_COUNT,
            "assignment_hash": self.assignment_hash,
            "schedule_sidecar_sha256": self.schedule_sidecar_sha256,
            "schedule_sidecar_scope": "host_metadata_not_h2d",
            "transfer_summary": {
                "edge_count": len(self.transfer_edges),
                "host_to_dpu_bytes": self.host_to_dpu_bytes,
                "dpu_to_host_bytes": self.dpu_to_host_bytes,
                "total_bytes": self.total_cross_dpu_transfer_bytes,
            },
            "execution_plan_hash": self.execution_plan_hash,
        })
        return payload


@dataclass(frozen=True)
class ParsedSchedule:
    version: int
    package_file_sha256: str
    operation_count: int
    wave_count: int
    requested_dpu_count: int
    tasklets_per_dpu: int
    provider_count: int
    records: tuple[tuple[int, int, int, int, int, int, int, int], ...]


def compile_plan(
    graph: TaskGraph,
    package: Any,
    *,
    placement_policy: str = PLACEMENT_SINGLE,
) -> ExecutionPlan:
    """Compile a real resident package using one of the two fixed policies."""

    from quantum_bench.targets.upmem.hardware_taskgraph_resident import ResidentGraphPackage

    if not isinstance(package, ResidentGraphPackage):
        raise TypeError("package must be a ResidentGraphPackage")
    dpu_count = {PLACEMENT_SINGLE: 1, PLACEMENT_FRONTIER: 2}.get(placement_policy)
    if dpu_count is None:
        raise ValueError("unsupported execution-plan placement policy")
    _validate_package_contract(graph, package)
    source = with_execution_identity(graph)
    package_identity = with_execution_identity(package.graph)
    tasks = {task.id: task for task in graph.tasks}
    operations = {item.task_id: item for item in package.operations}
    operation_indices = {item.task_id: index for index, item in enumerate(package.operations)}
    dependency_waves = _dependency_waves(tasks)
    if dpu_count == 1:
        ordered = sorted(package.operations, key=lambda item: (dependency_waves[item.task_id], item.operation_id))
        waves_by_task = {item.task_id: index for index, item in enumerate(ordered)}
    else:
        waves_by_task = dependency_waves
    wave_count = max(waves_by_task.values()) + 1
    assignments = []
    waves = [[] for _ in range(wave_count)]
    for wave_index in range(wave_count):
        current = sorted(
            (item for item in package.operations if waves_by_task[item.task_id] == wave_index),
            key=lambda item: item.operation_id,
        )
        for position, operation in enumerate(current):
            task = tasks[operation.task_id]
            dependencies = tuple(sorted(operations[item].operation_id for item in task.dependencies))
            assignments.append(TaskAssignment(
                operation_id=operation.operation_id,
                package_operation_index=operation_indices[task.id],
                task_id=task.id,
                component=operation.component,
                wave_index=wave_index,
                dpu_id=0 if dpu_count == 1 else position % dpu_count,
                dependency_operation_ids=dependencies,
                dependency_bitmask=sum(1 << item for item in dependencies),
                input_slot_ids=(operation.slot_a, operation.slot_b),
                output_slot_id=operation.slot_out_real,
                output_elements=operation.output_elements,
            ))
            waves[wave_index].append(task.id)
    assignments.sort(key=lambda item: item.operation_id)
    by_task = {item.task_id: item for item in assignments}
    plan = ExecutionPlan(
        placement_policy=placement_policy,
        requested_dpu_count=dpu_count,
        tasklets_per_dpu=TASKLETS_PER_DPU,
        package_file_sha256=package_file_sha256(package),
        source_circuit_semantics_hash=source.circuit_semantics_hash,
        source_tensor_network_hash=source.tensor_network_hash,
        source_contraction_plan_hash=source.contraction_plan_hash,
        package_circuit_semantics_hash=package_identity.circuit_semantics_hash,
        package_tensor_network_hash=package_identity.tensor_network_hash,
        package_contraction_plan_hash=package_identity.contraction_plan_hash,
        logical_task_count=len(graph.tasks),
        waves=tuple(tuple(item) for item in waves),
        assignments=tuple(assignments),
        transfer_edges=_derive_transfers(graph, package, by_task),
        final_outputs=_final_outputs(package),
    )
    validate_plan(plan, graph=graph, package=package)
    return plan


def validate_plan(
    plan: ExecutionPlan,
    *,
    graph: TaskGraph | None = None,
    package: Any | None = None,
    package_bytes: bytes | None = None,
) -> None:
    if plan.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported execution-plan schema")
    if (
        plan.runtime_provider_id != RUNTIME_PROVIDER_ID
        or plan.kernel_provider_id != KERNEL_PROVIDER_ID
        or plan.communication_provider_id != COMMUNICATION_PROVIDER_ID
        or plan.numeric_mode != NUMERIC_MODE
    ):
        raise ValueError("unsupported execution-plan provider or numeric mode")
    expected_dpus = {PLACEMENT_SINGLE: 1, PLACEMENT_FRONTIER: 2}.get(plan.placement_policy)
    if expected_dpus != plan.requested_dpu_count or plan.tasklets_per_dpu != TASKLETS_PER_DPU:
        raise ValueError("invalid DPU/tasklet execution-plan resources")
    if not 1 <= plan.operation_count <= MAX_OPERATIONS or not 1 <= plan.wave_count <= MAX_WAVES:
        raise ValueError("execution plan cap exceeded")
    if plan.logical_task_count != plan.operation_count:
        raise ValueError("Block 1 requires one real operation per logical task")
    if len(plan.final_outputs) != 1 or not _is_sha256(plan.package_file_sha256):
        raise ValueError("execution plan final output or package hash is invalid")
    if tuple(item.operation_id for item in plan.assignments) != tuple(range(plan.operation_count)):
        raise ValueError("operation IDs must be dense and ordered")
    if tuple(sorted(item.package_operation_index for item in plan.assignments)) != tuple(range(plan.operation_count)):
        raise ValueError("package operation indices must be dense")
    if len({item.task_id for item in plan.assignments}) != plan.operation_count:
        raise ValueError("task assignments are duplicated")
    expected_waves = tuple(
        tuple(item.task_id for item in plan.assignments if item.wave_index == index)
        for index in range(plan.wave_count)
    )
    if plan.waves != expected_waves:
        raise ValueError("waves do not match deterministic assignment order")
    by_id = {item.operation_id: item for item in plan.assignments}
    for item in plan.assignments:
        if item.component != "real":
            raise ValueError("assignment is outside the Block 1 real contract")
        if item.wave_index < 0 or item.wave_index >= plan.wave_count:
            raise ValueError("assignment wave index is out of range")
        if item.dpu_id < 0 or item.dpu_id >= plan.requested_dpu_count:
            raise ValueError("assignment DPU ID is out of range")
        if tuple(sorted(item.dependency_operation_ids)) != item.dependency_operation_ids:
            raise ValueError("dependency operation IDs are not ordered")
        if item.dependency_bitmask != sum(1 << dep for dep in item.dependency_operation_ids):
            raise ValueError("dependency bitmask mismatch")
        if any(dep < 0 or dep >= plan.operation_count for dep in item.dependency_operation_ids):
            raise ValueError("dependency operation is out of range")
        if any(by_id[dep].wave_index >= item.wave_index for dep in item.dependency_operation_ids):
            raise ValueError("dependency is not in an earlier wave")
    for edge in plan.transfer_edges:
        producer = by_id.get(edge.producer_operation_id)
        consumer = by_id.get(edge.consumer_operation_id)
        if producer is None or consumer is None or producer.dpu_id == consumer.dpu_id:
            raise ValueError("invalid cross-DPU transfer edge")
        if edge.transfer_bytes != _align8(edge.element_count * 4):
            raise ValueError("transfer edge byte count mismatch")
    if graph is not None:
        identity = with_execution_identity(graph)
        if (
            plan.source_circuit_semantics_hash,
            plan.source_tensor_network_hash,
            plan.source_contraction_plan_hash,
        ) != (
            identity.circuit_semantics_hash,
            identity.tensor_network_hash,
            identity.contraction_plan_hash,
        ):
            raise ValueError("source graph identity mismatch")
        if {item.id for item in graph.tasks} != {item.task_id for item in plan.assignments}:
            raise ValueError("TaskGraph task binding mismatch")
        by_task = {item.task_id: item for item in plan.assignments}
        for task in graph.tasks:
            expected = {by_task[item].operation_id for item in task.dependencies}
            if expected != set(by_task[task.id].dependency_operation_ids):
                raise ValueError("TaskGraph dependencies are not represented")
    if package is not None:
        _validate_package_binding(plan, package, graph)
        transfer_graph = graph or package.graph
        if plan.transfer_edges != _derive_transfers(
            transfer_graph, package, {item.task_id: item for item in plan.assignments}
        ):
            raise ValueError("transfer edges do not match package dependencies and placement")
    if package_bytes is not None:
        _validate_package_bytes(package_bytes)
        if hashlib.sha256(package_bytes).hexdigest() != plan.package_file_sha256:
            raise ValueError("package file hash mismatch")


def serialize_plan_json(plan: ExecutionPlan) -> bytes:
    validate_plan(plan)
    return json.dumps(plan.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def parse_plan_json(value: bytes | str | Mapping[str, Any]) -> ExecutionPlan:
    if isinstance(value, Mapping):
        payload = value
    else:
        try:
            payload = json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid execution-plan JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid execution-plan JSON schema")
    providers = _mapping(payload, "providers")
    source = _mapping(payload, "source_identity")
    package = _mapping(payload, "package_identity")
    waves = _list(payload, "waves")
    assignments = _records(payload, "task_assignments", _assignment_from_json)
    transfers = _records(payload, "transfer_edges", _transfer_from_json)
    outputs = _records(payload, "final_outputs", _final_from_json)
    if any(not isinstance(item, list) for item in waves):
        raise ValueError("execution-plan waves must be lists")
    plan = ExecutionPlan(
        placement_policy=_text(payload, "placement_policy"),
        requested_dpu_count=_uint(payload, "requested_dpu_count"),
        tasklets_per_dpu=_uint(payload, "tasklets_per_dpu"),
        package_file_sha256=_text(payload, "package_file_sha256"),
        source_circuit_semantics_hash=_text(source, "circuit_semantics_hash"),
        source_tensor_network_hash=_text(source, "tensor_network_hash"),
        source_contraction_plan_hash=_text(source, "contraction_plan_hash"),
        package_circuit_semantics_hash=_text(package, "circuit_semantics_hash"),
        package_tensor_network_hash=_text(package, "tensor_network_hash"),
        package_contraction_plan_hash=_text(package, "contraction_plan_hash"),
        logical_task_count=_uint(payload, "logical_task_count"),
        waves=tuple(tuple(_text_item(item) for item in wave) for wave in waves),
        assignments=assignments,
        transfer_edges=transfers,
        final_outputs=outputs,
        runtime_provider_id=_text(providers, "runtime"),
        kernel_provider_id=_text(providers, "kernel"),
        communication_provider_id=_text(providers, "communication"),
        numeric_mode=_text(providers, "numeric_mode"),
    )
    validate_plan(plan)
    if payload.get("execution_plan_hash") != plan.execution_plan_hash:
        raise ValueError("execution plan hash mismatch")
    if payload.get("assignment_hash") != plan.assignment_hash:
        raise ValueError("assignment hash mismatch")
    if payload.get("operation_count") != plan.operation_count or payload.get("wave_count") != plan.wave_count:
        raise ValueError("execution plan count metadata mismatch")
    if payload.get("provider_count") != PROVIDER_COUNT:
        raise ValueError("execution plan provider count metadata mismatch")
    if payload.get("schedule_sidecar_sha256") != plan.schedule_sidecar_sha256:
        raise ValueError("schedule sidecar hash mismatch")
    return plan


def serialize_schedule(plan: ExecutionPlan) -> bytes:
    validate_plan(plan)
    header = struct.pack(
        SCHEDULE_HEADER_FORMAT,
        SCHEDULE_MAGIC,
        SCHEDULE_VERSION,
        SCHEDULE_HEADER_BYTES,
        plan.operation_count,
        plan.wave_count,
        plan.requested_dpu_count,
        plan.tasklets_per_dpu,
        PROVIDER_COUNT,
        SCHEDULE_RECORD_BYTES,
        0,
        0,
        bytes.fromhex(plan.package_file_sha256),
    )
    records = b"".join(
        struct.pack(
            SCHEDULE_RECORD_FORMAT,
            item.package_operation_index,
            item.operation_id,
            item.dependency_bitmask,
            item.wave_index,
            item.dpu_id,
            item.input_slot_ids[0],
            item.input_slot_ids[1],
            item.output_slot_id,
        )
        for item in plan.assignments
    )
    return header + records


def parse_schedule(value: bytes | bytearray | memoryview) -> ParsedSchedule:
    data = bytes(value)
    if len(data) < SCHEDULE_HEADER_BYTES:
        raise ValueError("execution schedule is truncated")
    fields = struct.unpack_from(SCHEDULE_HEADER_FORMAT, data)
    magic, version, header_bytes, operation_count, wave_count, dpu_count, tasklets, providers, record_bytes, reserved0, reserved1, package_hash = fields
    if magic != SCHEDULE_MAGIC or version != SCHEDULE_VERSION:
        raise ValueError("execution schedule magic or version is invalid")
    if header_bytes != SCHEDULE_HEADER_BYTES or record_bytes != SCHEDULE_RECORD_BYTES or reserved0 or reserved1:
        raise ValueError("execution schedule ABI header is invalid")
    if providers != PROVIDER_COUNT:
        raise ValueError("execution schedule provider count is invalid")
    _check_caps(operation_count, wave_count, dpu_count, tasklets)
    if len(data) != header_bytes + operation_count * record_bytes:
        raise ValueError("execution schedule length is invalid")
    records = tuple(
        struct.unpack_from(SCHEDULE_RECORD_FORMAT, data, header_bytes + index * record_bytes)
        for index in range(operation_count)
    )
    if tuple(item[0] for item in records) != tuple(range(operation_count)):
        raise ValueError("execution schedule package operation indices are not dense")
    if tuple(item[1] for item in records) != tuple(range(operation_count)):
        raise ValueError("execution schedule operation IDs are not dense")
    for _, _, mask, wave, dpu, _, _, _ in records:
        if wave >= wave_count or dpu >= dpu_count or mask >= (1 << operation_count):
            raise ValueError("execution schedule record is out of range")
    return ParsedSchedule(version, package_hash.hex(), operation_count, wave_count, dpu_count, tasklets, providers, records)


def validate_schedule(
    value: bytes | bytearray | memoryview,
    plan: ExecutionPlan,
    *,
    package_bytes: bytes | None = None,
) -> ParsedSchedule:
    validate_plan(plan, package_bytes=package_bytes)
    parsed = parse_schedule(value)
    expected = tuple(
        (
            item.package_operation_index,
            item.operation_id,
            item.dependency_bitmask,
            item.wave_index,
            item.dpu_id,
            item.input_slot_ids[0],
            item.input_slot_ids[1],
            item.output_slot_id,
        )
        for item in plan.assignments
    )
    if parsed.package_file_sha256 != plan.package_file_sha256 or parsed.records != expected:
        raise ValueError("execution schedule does not match execution plan")
    return parsed


def build_request_manifest(
    plan: ExecutionPlan,
    package: Any,
    schedule_bytes: bytes,
    *,
    package_path: str,
    schedule_path: str,
    dpu_binary: str,
    requested_dpu_count: int | None = None,
    tasklets_per_dpu: int | None = None,
    final_outputs: tuple[FinalOutput, ...] | None = None,
) -> JsonDict:
    """Create the complete Block 2 request without later manifest mutation."""

    package_bytes = package_bytes_for(package)
    validate_plan(plan, package=package, package_bytes=package_bytes)
    validate_schedule(schedule_bytes, plan, package_bytes=package_bytes)
    if requested_dpu_count is not None and requested_dpu_count != plan.requested_dpu_count:
        raise ValueError("request DPU count does not match execution plan")
    if tasklets_per_dpu is not None and tasklets_per_dpu != plan.tasklets_per_dpu:
        raise ValueError("request tasklet count does not match execution plan")
    if final_outputs is not None and tuple(final_outputs) != plan.final_outputs:
        raise ValueError("request final outputs do not match execution plan")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "upmem_execution_plan_request",
        "runtime_provider_id": plan.runtime_provider_id,
        "kernel_provider_id": plan.kernel_provider_id,
        "communication_provider_id": plan.communication_provider_id,
        "numeric_mode": plan.numeric_mode,
        "requested_dpu_count": plan.requested_dpu_count,
        "tasklets_per_dpu": plan.tasklets_per_dpu,
        "package_path": package_path,
        "schedule_path": schedule_path,
        "dpu_binary": dpu_binary,
        "package_file_sha256": plan.package_file_sha256,
        "schedule_sidecar_sha256": hashlib.sha256(schedule_bytes).hexdigest(),
        "schedule_sidecar_h2d_bytes": 0,
        "schedule_sidecar_scope": "host_metadata_not_h2d",
        "execution_plan_hash": plan.execution_plan_hash,
        "source_identity": {
            "circuit_semantics_hash": plan.source_circuit_semantics_hash,
            "tensor_network_hash": plan.source_tensor_network_hash,
            "contraction_plan_hash": plan.source_contraction_plan_hash,
        },
        "package_identity": {
            "circuit_semantics_hash": plan.package_circuit_semantics_hash,
            "tensor_network_hash": plan.package_tensor_network_hash,
            "contraction_plan_hash": plan.package_contraction_plan_hash,
        },
        "final_outputs": [to_jsonable(item) for item in (final_outputs or plan.final_outputs)],
    }


def package_bytes_for(package: Any) -> bytes:
    path = getattr(package, "package_path", None)
    if isinstance(path, Path) and path.is_file():
        return path.read_bytes()
    from quantum_bench.targets.upmem.hardware_taskgraph_resident import _encode_package
    return _encode_package(package.allocation.slots, package.operations)


def package_file_sha256(package: Any) -> str:
    return hashlib.sha256(package_bytes_for(package)).hexdigest()


def _validate_package_bytes(package_bytes: bytes) -> None:
    from quantum_bench.targets.upmem.hardware_taskgraph_resident import validate_resident_graph_package_bytes
    validate_resident_graph_package_bytes(package_bytes)


def _validate_package_contract(graph: TaskGraph, package: Any) -> None:
    if package.quantization_mode != NUMERIC_MODE or len(package.operations) != len(graph.tasks):
        raise ValueError("Block 1 supports only real float32 one-operation-per-task packages")
    if tuple(item.operation_id for item in package.operations) != tuple(range(len(package.operations))):
        raise ValueError("resident package operation IDs must be dense")
    task_ids = {task.id for task in graph.tasks}
    if len(task_ids) != len(graph.tasks) or {item.task_id for item in package.operations} != task_ids:
        raise ValueError("resident package task mapping does not match TaskGraph")
    if len(package.allocation.final_components) != 1:
        raise ValueError("resident package must have one final output")
    if any(tensor.dtype != "float32" for tensor in package.graph.network.tensors):
        raise ValueError("Block 1 requires real float32 package tensors")
    for operation in package.operations:
        if operation.component != "real" or operation.kind != "contract" or operation.mode != NUMERIC_MODE:
            raise ValueError("Block 1 supports only real contract operations")
        task = next(item for item in graph.tasks if item.id == operation.task_id)
        expected = (
            package.allocation.slot_for(task.input_tensor_ids[0]),
            package.allocation.slot_for(task.input_tensor_ids[1]),
            package.allocation.slot_for(task.output_tensor_id),
        )
        if (operation.slot_a, operation.slot_b, operation.slot_out_real) != expected:
            raise ValueError("resident package slot binding does not match TaskGraph")
        if _slot(package, operation.slot_out_real).element_count != operation.output_elements:
            raise ValueError("resident package output size does not match slot descriptor")
    for component, slot_id, elements in package.allocation.final_components:
        if component != "real" or _slot(package, slot_id).element_count != int(elements):
            raise ValueError("resident package final output binding is invalid")


def _validate_package_binding(plan: ExecutionPlan, package: Any, graph: TaskGraph | None) -> None:
    _validate_package_contract(package.graph if graph is None else graph, package)
    identity = with_execution_identity(package.graph)
    if (
        plan.package_circuit_semantics_hash,
        plan.package_tensor_network_hash,
        plan.package_contraction_plan_hash,
    ) != (identity.circuit_semantics_hash, identity.tensor_network_hash, identity.contraction_plan_hash):
        raise ValueError("lowered/package graph identity mismatch")
    package_bytes = package_bytes_for(package)
    _validate_package_bytes(package_bytes)
    if hashlib.sha256(package_bytes).hexdigest() != plan.package_file_sha256:
        raise ValueError("package file hash mismatch")
    if plan.final_outputs != _final_outputs(package):
        raise ValueError("final output producer binding mismatch")
    index_by_id = {item.operation_id: index for index, item in enumerate(package.operations)}
    operation_by_task = {item.task_id: item for item in package.operations}
    for assignment in plan.assignments:
        operation = operation_by_task.get(assignment.task_id)
        if operation is None or index_by_id[operation.operation_id] != assignment.package_operation_index:
            raise ValueError("package operation index binding mismatch")
        if operation.operation_id != assignment.operation_id or operation.component != assignment.component:
            raise ValueError("package operation/component binding mismatch")
        if (operation.slot_a, operation.slot_b, operation.slot_out_real) != (
            assignment.input_slot_ids[0], assignment.input_slot_ids[1], assignment.output_slot_id
        ):
            raise ValueError("package operation slot binding mismatch")


def _derive_transfers(
    graph: TaskGraph,
    package: Any,
    assignments: Mapping[str, TaskAssignment],
) -> tuple[CrossDpuTransfer, ...]:
    operations = {item.task_id: item for item in package.operations}
    result = []
    for task in sorted(graph.tasks, key=lambda item: assignments[item.id].operation_id):
        consumer = assignments[task.id]
        for dependency in task.dependencies:
            producer = assignments[dependency]
            if producer.dpu_id == consumer.dpu_id:
                continue
            operation = operations[dependency]
            elements = int(_slot(package, operation.slot_out_real).element_count)
            result.append(CrossDpuTransfer(
                producer_operation_id=producer.operation_id,
                consumer_operation_id=consumer.operation_id,
                producer_task_id=dependency,
                consumer_task_id=task.id,
                producer_dpu_id=producer.dpu_id,
                consumer_dpu_id=consumer.dpu_id,
                slot_id=operation.slot_out_real,
                element_count=elements,
                transfer_bytes=_align8(elements * 4),
            ))
    return tuple(result)


def _dependency_waves(tasks: Mapping[str, Any]) -> dict[str, int]:
    waves: dict[str, int] = {}
    remaining = set(tasks)
    while remaining:
        ready = sorted(item for item in remaining if all(dep in waves for dep in tasks[item].dependencies))
        if not ready:
            raise ValueError("TaskGraph dependency cycle or unknown dependency")
        for item in ready:
            deps = tasks[item].dependencies
            waves[item] = 0 if not deps else max(waves[dep] for dep in deps) + 1
            remaining.remove(item)
    if max(waves.values(), default=-1) + 1 > MAX_WAVES:
        raise ValueError("execution plan wave cap exceeded")
    return waves


def _final_outputs(package: Any) -> tuple[FinalOutput, ...]:
    result = []
    for component, slot_id, elements in package.allocation.final_components:
        tensor_id = next((item.tensor_id for item in _slot(package, slot_id).lifetimes if item.final), "")
        if not tensor_id:
            raise ValueError("final slot has no final tensor producer")
        result.append(FinalOutput(str(component), int(slot_id), tensor_id, int(elements)))
    return tuple(result)


def _slot(package: Any, slot_id: int) -> Any:
    for item in package.allocation.slots:
        if int(item.slot_id) == int(slot_id):
            return item
    raise ValueError(f"resident package slot {slot_id} is missing")


def _check_caps(operations: int, waves: int, dpus: int, tasklets: int) -> None:
    if not 1 <= operations <= MAX_OPERATIONS or not 1 <= waves <= MAX_WAVES:
        raise ValueError("execution plan cap exceeded")
    if dpus not in {1, 2} or tasklets != TASKLETS_PER_DPU:
        raise ValueError("execution plan supports one or two DPUs and one tasklet")


def _align8(value: int) -> int:
    return (int(value) + 7) & ~7


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(item in "0123456789abcdef" for item in value)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"execution-plan field {key!r} must be an object")
    return result


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list):
        raise ValueError(f"execution-plan field {key!r} must be a list")
    return result


def _records(value: Mapping[str, Any], key: str, factory: Callable[[Mapping[str, Any]], Any]) -> tuple[Any, ...]:
    result = []
    for index, item in enumerate(_list(value, key)):
        if not isinstance(item, Mapping):
            raise ValueError(f"execution-plan field {key!r}[{index}] must be an object")
        result.append(factory(item))
    return tuple(result)


def _assignment_from_json(value: Mapping[str, Any]) -> TaskAssignment:
    return TaskAssignment(
        _uint(value, "operation_id"),
        _uint(value, "package_operation_index"),
        _text(value, "task_id"),
        _text(value, "component"),
        _uint(value, "wave_index"),
        _uint(value, "dpu_id"),
        tuple(_uint_list(value, "dependency_operation_ids")),
        _uint(value, "dependency_bitmask"),
        _pair(_uint_list(value, "input_slot_ids")),
        _uint(value, "output_slot_id"),
        _uint(value, "output_elements"),
    )


def _transfer_from_json(value: Mapping[str, Any]) -> CrossDpuTransfer:
    if value.get("transport") != COMMUNICATION_PROVIDER_ID:
        raise ValueError("unsupported execution-plan transfer transport")
    return CrossDpuTransfer(
        _uint(value, "producer_operation_id"),
        _uint(value, "consumer_operation_id"),
        _text(value, "producer_task_id"),
        _text(value, "consumer_task_id"),
        _uint(value, "producer_dpu_id"),
        _uint(value, "consumer_dpu_id"),
        _uint(value, "slot_id"),
        _uint(value, "element_count"),
        _uint(value, "transfer_bytes"),
    )


def _final_from_json(value: Mapping[str, Any]) -> FinalOutput:
    return FinalOutput(
        _text(value, "component"),
        _uint(value, "slot_id"),
        _text(value, "tensor_id"),
        _uint(value, "element_count"),
    )


def _assignment_json(value: TaskAssignment) -> JsonDict:
    return to_jsonable(value)


def _transfer_json(value: CrossDpuTransfer) -> JsonDict:
    return to_jsonable(value) | {"transport": COMMUNICATION_PROVIDER_ID}


def _text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"execution-plan field {key!r} must be non-empty text")
    return result


def _text_item(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("execution-plan wave item must be non-empty text")
    return value


def _uint(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError(f"execution-plan field {key!r} must be a non-negative integer")
    return result


def _uint_list(value: Mapping[str, Any], key: str) -> list[int]:
    result = _list(value, key)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in result):
        raise ValueError(f"execution-plan field {key!r} contains an invalid integer")
    return result


def _pair(value: list[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("input_slot_ids must contain two entries")
    return value[0], value[1]


__all__ = [
    "COMMUNICATION_PROVIDER_ID",
    "CrossDpuTransfer",
    "ExecutionPlan",
    "FinalOutput",
    "KERNEL_PROVIDER_ID",
    "MAX_OPERATIONS",
    "MAX_WAVES",
    "NUMERIC_MODE",
    "ParsedSchedule",
    "PLACEMENT_FRONTIER",
    "PLACEMENT_SINGLE",
    "PROVIDER_COUNT",
    "RUNTIME_PROVIDER_ID",
    "SCHEMA_VERSION",
    "SCHEDULE_HEADER_BYTES",
    "SCHEDULE_HEADER_FORMAT",
    "SCHEDULE_MAGIC",
    "SCHEDULE_RECORD_BYTES",
    "SCHEDULE_RECORD_FORMAT",
    "SCHEDULE_VERSION",
    "TASKLETS_PER_DPU",
    "TaskAssignment",
    "build_request_manifest",
    "compile_plan",
    "package_bytes_for",
    "package_file_sha256",
    "parse_plan_json",
    "parse_schedule",
    "serialize_plan_json",
    "serialize_schedule",
    "validate_plan",
    "validate_schedule",
]
