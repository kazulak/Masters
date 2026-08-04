"""Fixed two-DPU frontier contract for M3.1.

This module is deliberately a small policy layer over the resident graph ABI.
The binary package remains the package emitted by
``build_resident_graph_package`` and ``ResidentGraphPackage.write``.  Frontier
identity, placement, and dependency evidence live in the request manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np

from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict, TaskGraph, TensorValue
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
    RESIDENT_DESCRIPTOR_CONTROL_BYTES,
    RESIDENT_OPERATION_BYTES,
    RESIDENT_PACKAGE_HEADER_FORMAT,
    RESIDENT_SESSION_PROTOCOL,
    ResidentGraphPackage,
    _resident_operation_format,
    build_resident_graph_package,
    validate_resident_graph_package_bytes,
)
from quantum_bench.tn.execution_bundle import with_execution_identity
from quantum_bench.tn.network import TensorNetworkValue


PROFILE_ID = "hardware_frontier_two_dpu_m3_1_v2"
BACKEND_ID = "upmem_sdk_hardware_taskgraph_frontier_two_dpu"
ROUTE_ID = "upmem_tn_hardware_taskgraph_frontier_two_dpu"
NATIVE_SCHEMA = "generic_loop_resident_frontier_two_dpu_v2"
REQUEST_SCHEMA = RESIDENT_SESSION_PROTOCOL
SESSION_PROTOCOL = REQUEST_SCHEMA
TARGET = "hardware"
NUMERIC_MODE = "none"
TASKLETS_PER_DPU = 1
REQUESTED_DPUS = 2
TIMING_SCOPE = "two_dpu_frontier_resident_full_taskgraph_v1"
TIMING_COMPONENT_FIELDS = (
    "package_parse_time_s",
    "allocation_time_s",
    "binary_load_time_s",
    "initial_h2d_time_s",
    "descriptor_h2d_time_s",
    "control_h2d_time_s",
    "wave0_launch_time_s",
    "wave0_sync_time_s",
    "inter_wave_d2h_time_s",
    "inter_wave_h2d_time_s",
    "wave1_launch_time_s",
    "wave1_sync_time_s",
    "final_d2h_time_s",
    "output_write_time_s",
    "release_time_s",
)
TIMING_ALIAS_FIELDS = (
    "wave0_barrier_wait_time_s",
    "wave1_barrier_wait_time_s",
)
TIMING_FIELDS = (
    *TIMING_COMPONENT_FIELDS,
    *TIMING_ALIAS_FIELDS,
    "total_route_time_s",
)
OVERLAP_EVIDENCE = "co_dispatch_without_overlap_measurement"
TRANSFER_SCOPE = "native_sdk_observed_application_visible"
COMPLETION_SCOPE = "wave_dependency_order_not_intra_wave_finish_order"
EXPECTED_TASK_IDS = ("task_0", "task_1", "task_2")
EXPECTED_PATH = ((0, 1), (0, 1), (0, 1))
EXPECTED_DPU_IDS = (0, 1)
EXPECTED_DPU_TASK_COUNTS = (2, 1)
EXPECTED_BARRIER_COUNT = 2
EXPECTED_WAVES = (
    (("task_0", 0), ("task_1", 1)),
    (("task_2", 0),),
)


@dataclass(frozen=True)
class FrontierAssignment:
    task_id: str
    wave_index: int
    dpu_id: int

    def to_json_dict(self) -> JsonDict:
        return {
            "task_id": self.task_id,
            "wave_index": self.wave_index,
            "dpu_id": self.dpu_id,
        }


@dataclass(frozen=True)
class HardwareFrontierPlan:
    assignments: tuple[FrontierAssignment, ...]
    barrier_count: int = EXPECTED_BARRIER_COUNT
    co_dispatch: bool = True
    overlap_measured: bool = False

    @property
    def waves(self) -> tuple[tuple[FrontierAssignment, ...], ...]:
        return tuple(
            tuple(item for item in self.assignments if item.wave_index == wave)
            for wave in range(2)
        )

    @property
    def dpu_task_counts(self) -> tuple[int, int]:
        return tuple(
            sum(item.dpu_id == dpu_id for item in self.assignments)
            for dpu_id in EXPECTED_DPU_IDS
        )  # type: ignore[return-value]

    def to_json_dict(self) -> JsonDict:
        return {
            "plan_schema_version": "upmem_hardware_frontier_plan_m3_1_v1",
            "wave_count": 2,
            "waves": [
                {
                    "wave_index": index,
                    "tasks": [item.to_json_dict() for item in wave],
                    "barrier_after": True,
                }
                for index, wave in enumerate(self.waves)
            ],
            "assignments": [item.to_json_dict() for item in self.assignments],
            "barrier_count": self.barrier_count,
            "expected_dpu_task_counts": list(self.dpu_task_counts),
            "co_dispatch": self.co_dispatch,
            "overlap_measured": self.overlap_measured,
            "overlap_evidence": OVERLAP_EVIDENCE,
        }


def build_hardware_frontier_plan(
    graph: TaskGraph,
    network: Any,
) -> HardwareFrontierPlan:
    """Validate the fixed three-task graph and return its only valid plan."""

    _validate_frontier_graph(graph, network)
    plan = HardwareFrontierPlan(
        assignments=(
            FrontierAssignment("task_0", 0, 0),
            FrontierAssignment("task_1", 0, 1),
            FrontierAssignment("task_2", 1, 0),
        )
    )
    _validate_frontier_plan(plan)
    return plan


def validate_hardware_frontier_graph(graph: TaskGraph, network: Any) -> None:
    """Raise ``ValueError`` unless graph and tensor dataflow are M3.1-shaped."""

    _validate_frontier_graph(graph, network)


def build_hardware_frontier_graph_package(
    graph: TaskGraph,
    network: Any,
    *,
    case_id: str,
    suite_id: str,
    quantization_mode: str = NUMERIC_MODE,
    full_precision_output: np.ndarray | None = None,
) -> ResidentGraphPackage:
    """Build one resident package after enforcing the fixed frontier contract."""

    build_hardware_frontier_plan(graph, network)
    if quantization_mode != NUMERIC_MODE:
        raise ValueError("hardware_profile_violation: frontier numeric_mode must be 'none'")
    lowered_graph, lowered_network = _lower_real_float32(graph, network)
    return build_resident_graph_package(
        lowered_graph,
        lowered_network,
        case_id=case_id,
        suite_id=suite_id,
        quantization_mode=NUMERIC_MODE,
        full_precision_output=full_precision_output,
    )


def write_hardware_frontier_graph_package(
    package: ResidentGraphPackage,
    session_root: Path,
    *,
    dpu_binary: Path,
    request_id: str,
) -> ResidentGraphPackage:
    """Write one full resident package and augment its request manifest."""

    plan = build_hardware_frontier_plan(package.graph, _network_from_package(package))
    written = package.write(session_root, dpu_binary=dpu_binary, request_id=request_id)
    if written.manifest_path is None:
        raise ValueError("hardware_profile_violation: resident writer did not return a manifest")
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest_parse_failed: resident request manifest is not an object")
    if written.package_path is None:
        raise ValueError("hardware_profile_violation: resident writer did not return a package")
    _augment_frontier_manifest(
        manifest, package, plan, written.manifest_path.parent, written.package_path
    )
    write_json(written.manifest_path, manifest)
    return written


def validate_frontier_native_response(
    response: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Strictly validate native completion evidence for the fixed frontier."""

    if not isinstance(response, Mapping):
        raise ValueError("response_evidence_invalid: frontier response is not an object")
    _validate_manifest_identity(manifest)
    _require(response.get("status") == "completed", "native response is not completed")
    _require(response.get("failure_stage") is None, "completed response has a failure stage")
    for key, expected in (
        ("schema_version", NATIVE_SCHEMA),
        ("native_schema_version", NATIVE_SCHEMA),
        ("route_id", ROUTE_ID),
        ("backend_id", BACKEND_ID),
        ("hardware_profile_version", PROFILE_ID),
        ("target_requested", TARGET),
        ("target_observed", TARGET),
        ("numeric_mode", NUMERIC_MODE),
        ("tasklets_per_dpu", TASKLETS_PER_DPU),
        ("requested_dpus", REQUESTED_DPUS),
        ("allocated_dpus", REQUESTED_DPUS),
    ):
        _require(response.get(key) == expected, f"response field {key} mismatch")
    for key, expected in {
        "hardware_execution": True,
        "hardware_kernel_executed": True,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "no_cpu_fallback": True,
        "no_simulator_fallback": True,
    }.items():
        _require(response.get(key) is expected, f"response flag {key} invalid")
    _require(response.get("hardware_functionality_evidence") is True, "hardware evidence is missing")
    _require(response.get("native_failure_fallback_used") is False, "native fallback was used")
    _require(response.get("hardware_no_fallback") is True, "hardware no-fallback flag is missing")
    _require(response.get("performance_claim_applicable") is False, "frontier must not make performance claims")
    timing = _mapping(response, "timing")
    _require(response.get("timing_scope") == TIMING_SCOPE, "timing scope mismatch")
    _require(timing.get("clock") == "clock_monotonic", "timing clock mismatch")
    _require(timing.get("overlap_measured") is False, "timing overlap must be unmeasured")
    for field in TIMING_FIELDS:
        value = timing.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0,
            f"timing field {field} is invalid",
        )
    for alias, component in (
        ("wave0_barrier_wait_time_s", "wave0_sync_time_s"),
        ("wave1_barrier_wait_time_s", "wave1_sync_time_s"),
    ):
        _require(
            math.isclose(
                float(timing[alias]), float(timing[component]), rel_tol=1.0e-9, abs_tol=1.0e-12
            ),
            f"timing alias {alias} disagrees with {component}",
        )
    expected_total = sum(float(timing[field]) for field in TIMING_COMPONENT_FIELDS)
    _require(
        math.isclose(
            float(timing["total_route_time_s"]), expected_total, rel_tol=1.0e-9, abs_tol=1.0e-8
        ),
        "timing total_route_time_s does not equal its component sum",
    )

    allocation = _mapping(response, "allocation")
    _require(allocation.get("requested_dpus") == 2, "allocation requested DPU count mismatch")
    _require(allocation.get("allocated_dpus") == 2, "allocation count mismatch")
    _require(allocation.get("verified") is True, "allocation was not verified")
    load = _mapping(response, "load")
    _require(load.get("confirmed") is True, "binary load was not confirmed")
    _require(load.get("hardware") is True, "binary load was not hardware-backed")
    launch = _mapping(response, "launch")
    _require(launch.get("completed") is True, "launch was not completed")
    _require(launch.get("task_count") == 3, "launch task count mismatch")
    _require(launch.get("barrier_count") == EXPECTED_BARRIER_COUNT, "launch barrier count mismatch")
    release = _mapping(response, "release")
    _require(release.get("confirmed") is True, "DPU release was not confirmed")

    _validate_frontier_execution_evidence(response)
    _validate_transfer_invariant(response, manifest)
    _validate_response_tasks(response, manifest)
    _validate_final_output(response, manifest)


def validate_frontier_package_validation_response(
    response: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Validate native package validation evidence without touching hardware."""

    if not isinstance(response, Mapping):
        raise ValueError("response_evidence_invalid: validation response is not an object")
    _validate_manifest_identity(manifest)
    for key, expected in (
        ("schema_version", NATIVE_SCHEMA),
        ("native_schema_version", NATIVE_SCHEMA),
        ("route_id", ROUTE_ID),
        ("backend_id", BACKEND_ID),
        ("hardware_profile_version", PROFILE_ID),
        ("target", TARGET),
        ("session_protocol", NATIVE_SCHEMA),
        ("status", "valid"),
        ("valid", True),
        ("failure_stage", None),
        ("error", None),
        ("native_execution", False),
        ("allocation_attempted", False),
        ("launch_attempted", False),
        ("release_attempted", False),
        ("requested_dpus", REQUESTED_DPUS),
        ("tasklets_per_dpu", TASKLETS_PER_DPU),
        ("operation_count", 3),
        ("final_output_count", 1),
        ("quantization_mode", NUMERIC_MODE),
        ("profile_id", PROFILE_ID),
        ("wave_barrier_count", EXPECTED_BARRIER_COUNT),
    ):
        _require(response.get(key) == expected, f"validation response field {key} mismatch")


def validate_hardware_frontier_response(
    response: Mapping[str, Any], manifest: Mapping[str, Any]
) -> bool:
    """Boolean convenience wrapper for callers that do not want exceptions."""

    try:
        validate_frontier_native_response(response, manifest)
    except ValueError:
        return False
    return True


def _validate_frontier_graph(graph: TaskGraph, network: Any) -> None:
    tasks = tuple(graph.tasks)
    if tuple(task.id for task in tasks) != EXPECTED_TASK_IDS:
        raise ValueError("hardware_profile_violation: frontier requires task_0, task_1, task_2")
    if tuple(graph.path) != EXPECTED_PATH:
        raise ValueError("hardware_profile_violation: frontier path must be the greedy RY-H-RY path")
    if tasks[0].dependencies or tasks[1].dependencies:
        raise ValueError("hardware_profile_violation: wave0 tasks must be independent")
    if tasks[2].dependencies != ("task_0", "task_1"):
        raise ValueError("hardware_profile_violation: task_2 dependencies are not task_0/task_1")
    outputs = [task.output_tensor_id for task in tasks]
    if len(set(outputs)) != 3:
        raise ValueError("hardware_profile_violation: task output tensor IDs must be unique")
    source_ids = {tensor.spec.id for tensor in network.tensors}
    produced: dict[str, str] = {}
    for task in tasks:
        if task.output_tensor_id in source_ids:
            raise ValueError("hardware_profile_violation: task output aliases a source tensor")
        for tensor_id in task.input_tensor_ids:
            producer = produced.get(tensor_id)
            if producer is None and tensor_id not in source_ids:
                raise ValueError("hardware_profile_violation: task input has no source or producer")
            if producer is not None and producer not in task.dependencies:
                raise ValueError("hardware_profile_violation: produced tensor lacks declared dependency")
        for dependency in task.dependencies:
            dependency_task = next((item for item in tasks if item.id == dependency), None)
            if dependency_task is None or dependency_task.output_tensor_id not in task.input_tensor_ids:
                raise ValueError("hardware_profile_violation: dependency/dataflow mismatch")
        produced[task.output_tensor_id] = task.id
    if set(tasks[2].input_tensor_ids) != {tasks[0].output_tensor_id, tasks[1].output_tensor_id}:
        raise ValueError("hardware_profile_violation: task_2 must consume both wave0 outputs")
    for tensor in network.tensors:
        array = np.asarray(tensor.array)
        try:
            declared_dtype = np.dtype(tensor.spec.dtype)
        except TypeError as exc:
            raise ValueError("hardware_profile_violation: frontier tensor dtype is invalid") from exc
        if declared_dtype != np.dtype("complex128") or array.dtype != np.dtype("complex128"):
            raise ValueError("hardware_profile_violation: frontier tensors must use complex128 storage")
        if tuple(array.shape) != tuple(tensor.spec.shape):
            raise ValueError("hardware_profile_violation: frontier tensor shape disagrees with its spec")
        if not np.all(np.isfinite(array.real)) or (
            np.iscomplexobj(array) and not np.all(np.isfinite(array.imag))
        ):
            raise ValueError("hardware_profile_violation: frontier tensors must be finite")
        if np.any(array.imag != 0):
            raise ValueError("hardware_profile_violation: frontier tensors must have exactly zero imaginary components")


def _lower_real_float32(
    graph: TaskGraph, network: Any
) -> tuple[TaskGraph, TensorNetworkValue]:
    if graph.network != network.spec:
        raise ValueError("hardware_profile_violation: frontier graph/network specs differ")
    lowered_specs = tuple(replace(tensor.spec, dtype="float32") for tensor in network.tensors)
    lowered_spec = replace(network.spec, tensors=lowered_specs)
    lowered_values = []
    for spec, tensor in zip(lowered_specs, network.tensors, strict=True):
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(np.asarray(tensor.array).real, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError("hardware_profile_violation: frontier float32 lowering overflowed")
        lowered_values.append(TensorValue(spec, array))
    lowered_graph = with_execution_identity(
        replace(
            graph,
            network=lowered_spec,
            circuit_semantics_hash="",
            tensor_network_hash="",
            contraction_plan_hash="",
        )
    )
    return lowered_graph, TensorNetworkValue(lowered_spec, lowered_values)


def _validate_frontier_plan(plan: HardwareFrontierPlan) -> None:
    expected = tuple(
        FrontierAssignment(task_id, wave, dpu)
        for wave in range(2)
        for task_id, dpu in EXPECTED_WAVES[wave]
    )
    if plan.assignments != expected:
        raise ValueError("hardware_profile_violation: frontier assignment plan is not fixed")
    if plan.barrier_count != 2 or plan.dpu_task_counts != EXPECTED_DPU_TASK_COUNTS:
        raise ValueError("hardware_profile_violation: frontier barrier or DPU counts mismatch")
    if not plan.co_dispatch or plan.overlap_measured:
        raise ValueError("hardware_profile_violation: frontier overlap evidence is invalid")


def _network_from_package(package: ResidentGraphPackage) -> Any:
    # The resident package stores slot data, not the original network.  Graph
    # validation still has all required dataflow information; package writing
    # is therefore checked with a lightweight synthetic source view.
    source_specs = package.graph.network.tensors
    values = []
    for spec in source_specs:
        components = [
            slot for logical, slot in package.allocation.logical_to_slot.items()
            if logical == f"{spec.id}::real"
        ]
        if not components:
            raise ValueError("hardware_profile_violation: source tensor slot is missing")
        values.append(
            type(
                "Tensor",
                (),
                {
                    "spec": replace(spec, dtype="complex128"),
                    "array": np.asarray(package.initial_data[components[0]], dtype=np.complex128),
                },
            )()
        )
    return type("Network", (), {"tensors": tuple(values)})()


def _augment_frontier_manifest(
    manifest: JsonDict,
    package: ResidentGraphPackage,
    plan: HardwareFrontierPlan,
    manifest_dir: Path,
    package_path: Path,
) -> None:
    operations = []
    operation_by_task = {operation.task_id: operation for operation in package.operations}
    for assignment in plan.assignments:
        operation = operation_by_task.get(assignment.task_id)
        if operation is None or operation.component != "real" or operation.kind != "contract":
            raise ValueError("hardware_profile_violation: frontier requires one real contract per task")
        task = next(task for task in package.graph.tasks if task.id == assignment.task_id)
        if (
            operation.mode != NUMERIC_MODE
            or operation.slot_a != package.allocation.slot_for(task.input_tensor_ids[0])
            or operation.slot_b != package.allocation.slot_for(task.input_tensor_ids[1])
            or operation.slot_out_real != package.allocation.slot_for(task.output_tensor_id)
        ):
            raise ValueError("hardware_profile_violation: frontier operation slot flow disagrees with task dataflow")
        operations.append(
            {
                **assignment.to_json_dict(),
                "operation_id": operation.operation_id,
                "component": operation.component,
                "kind": operation.kind,
                "mode": operation.mode,
                "input_tensor_ids": list(task.input_tensor_ids),
                "output_tensor_id": task.output_tensor_id,
                "dependencies": list(task.dependencies),
                "input_slot_ids": [operation.slot_a, operation.slot_b],
                "output_slot_id": operation.slot_out_real,
                "output_elements": operation.output_elements,
            }
        )
    if len(operations) != 3 or tuple(item["operation_id"] for item in operations) != (0, 1, 2):
        raise ValueError("hardware_profile_violation: frontier operation flow is not one-to-one")
    package_bytes = package_path.read_bytes()
    package_metadata = validate_resident_graph_package_bytes(package_bytes)
    package_operations = _resident_package_operations(package_bytes, package_metadata)
    _require_frontier_package_operations(package_operations, operations)
    final_output = _manifest_final_binding(manifest)
    operation_one = package_operations[1]
    inter_wave_bytes = _align8(int(operation_one["output_elements"]) * 4)
    descriptor_package_one_dpu = _nonnegative_int(
        manifest.get("descriptor_h2d_bytes"), "descriptor_h2d_bytes"
    ) + RESIDENT_DESCRIPTOR_CONTROL_BYTES
    initial_one_dpu = _nonnegative_int(manifest.get("initial_h2d_bytes"), "initial_h2d_bytes")
    operation_control = len(operations) * RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH
    transfer = {
        "descriptor_package_h2d_bytes": descriptor_package_one_dpu * REQUESTED_DPUS,
        "initial_h2d_bytes": initial_one_dpu * REQUESTED_DPUS,
        "operation_control_h2d_bytes": operation_control,
        "inter_wave_h2d_bytes": inter_wave_bytes,
        "inter_wave_d2h_bytes": inter_wave_bytes,
        "final_d2h_bytes": _nonnegative_int(manifest.get("final_d2h_bytes"), "final_d2h_bytes"),
    }
    transfer["h2d_bytes"] = (
        transfer["descriptor_package_h2d_bytes"]
        + transfer["initial_h2d_bytes"]
        + transfer["operation_control_h2d_bytes"]
        + transfer["inter_wave_h2d_bytes"]
    )
    transfer["d2h_bytes"] = transfer["inter_wave_d2h_bytes"] + transfer["final_d2h_bytes"]
    transfer["total_bytes"] = transfer["h2d_bytes"] + transfer["d2h_bytes"]
    _require(manifest.get("schema_version") == REQUEST_SCHEMA, "resident base schema_version was not preserved")
    manifest.update(
        {
            "manifest_kind": "frontier_graph_request",
            "session_protocol": REQUEST_SCHEMA,
            "native_schema_version": NATIVE_SCHEMA,
            "route_id": ROUTE_ID,
            "backend_id": BACKEND_ID,
            "hardware_profile_version": PROFILE_ID,
            "requested_dpus": REQUESTED_DPUS,
            "tasklets": TASKLETS_PER_DPU,
            "tasklets_per_dpu": TASKLETS_PER_DPU,
            "numeric_mode": NUMERIC_MODE,
            "quantization_mode": NUMERIC_MODE,
            "target": TARGET,
            "timing_scope": TIMING_SCOPE,
            "timing_fields": list(TIMING_FIELDS),
            "frontier_identity": {
                "profile_id": PROFILE_ID,
                "backend_id": BACKEND_ID,
                "route_id": ROUTE_ID,
                "native_schema": NATIVE_SCHEMA,
            },
            "frontier_plan": plan.to_json_dict(),
            "frontier_task_operations": operations,
            "resident_package_binding": {
                "package_sha256": hashlib.sha256(package_bytes).hexdigest(),
                "slot_descriptors": package_metadata["slot_descriptors"],
                "operations": package_operations,
            },
            "expected_task_ids": list(EXPECTED_TASK_IDS),
            "expected_dpu_task_counts": list(EXPECTED_DPU_TASK_COUNTS),
            "barrier_count": EXPECTED_BARRIER_COUNT,
            "co_dispatch": True,
            "overlap_measured": False,
            "overlap_evidence": OVERLAP_EVIDENCE,
            "transfer_accounting_scope": TRANSFER_SCOPE,
            "expected_frontier_transfer": transfer,
            "no_cpu_fallback": True,
            "no_simulator_fallback": True,
            "hardware_no_fallback": True,
            "performance_claim_applicable": False,
            "final_output_binding": final_output,
        }
    )
    _require(manifest_dir.is_dir(), "frontier manifest directory is missing")


def _validate_manifest_identity(manifest: Mapping[str, Any]) -> None:
    for key, expected in (
        ("schema_version", REQUEST_SCHEMA),
        ("native_schema_version", NATIVE_SCHEMA),
        ("route_id", ROUTE_ID),
        ("backend_id", BACKEND_ID),
        ("hardware_profile_version", PROFILE_ID),
        ("session_protocol", REQUEST_SCHEMA),
        ("target", TARGET),
        ("requested_dpus", REQUESTED_DPUS),
        ("tasklets", TASKLETS_PER_DPU),
        ("tasklets_per_dpu", TASKLETS_PER_DPU),
        ("numeric_mode", NUMERIC_MODE),
        ("quantization_mode", NUMERIC_MODE),
        ("barrier_count", EXPECTED_BARRIER_COUNT),
        ("performance_claim_applicable", False),
    ):
        _require(manifest.get(key) == expected, f"manifest field {key} mismatch")
    _require(manifest.get("expected_task_ids") == list(EXPECTED_TASK_IDS), "manifest task identity mismatch")
    _require(manifest.get("expected_dpu_task_counts") == list(EXPECTED_DPU_TASK_COUNTS), "manifest DPU count mismatch")
    expected_plan = HardwareFrontierPlan(
        assignments=tuple(
            FrontierAssignment(task_id, wave, dpu_id)
            for wave in range(2)
            for task_id, dpu_id in EXPECTED_WAVES[wave]
        )
    ).to_json_dict()
    _require(manifest.get("frontier_plan") == expected_plan, "manifest frontier plan drifted")
    _require(manifest.get("co_dispatch") is True, "manifest co-dispatch binding is invalid")
    _require(manifest.get("overlap_evidence") == OVERLAP_EVIDENCE, "manifest overlap evidence is invalid")
    _require(manifest.get("overlap_measured") is False, "manifest overlap must be unmeasured")
    operations = manifest.get("frontier_task_operations")
    _require(isinstance(operations, list) and len(operations) == 3, "manifest operation plan is invalid")
    for index, (item, expected) in enumerate(zip(operations, EXPECTED_WAVES[0] + EXPECTED_WAVES[1], strict=True)):
        _require(isinstance(item, Mapping), "manifest operation entry is invalid")
        task_id, dpu_id = expected
        _require(item.get("task_id") == task_id, "manifest task operation order mismatch")
        _require(item.get("wave_index") == (0 if index < 2 else 1), "manifest wave binding mismatch")
        _require(item.get("dpu_id") == dpu_id, "manifest DPU binding mismatch")
        _require(item.get("operation_id") == index, "manifest operation ID mismatch")
        input_slots = item.get("input_slot_ids")
        _require(isinstance(input_slots, list) and len(input_slots) == 2, "manifest input slot flow is invalid")
        _require(isinstance(item.get("output_slot_id"), int), "manifest output slot flow is invalid")
        _require(item.get("component") == "real", "manifest operation component mismatch")
        _require(item.get("kind") == "contract", "manifest operation kind mismatch")
        _require(item.get("mode") == NUMERIC_MODE, "manifest operation numeric mode mismatch")
        _require(item.get("dependencies") == ([] if index < 2 else ["task_0", "task_1"]), "manifest operation dependency mismatch")
    package_binding = manifest.get("resident_package_binding")
    _require(isinstance(package_binding, Mapping), "manifest resident package binding is missing")
    _require(isinstance(package_binding.get("slot_descriptors"), list), "manifest package slots are missing")
    _require(isinstance(package_binding.get("operations"), list), "manifest package operations are missing")
    final_output = manifest.get("final_output_binding")
    _require(isinstance(final_output, Mapping), "manifest final output binding is missing")
    _require(final_output.get("component") == "real", "manifest final output component is invalid")
    _require(final_output.get("slot_id") == operations[2].get("output_slot_id"), "manifest final output slot drifted")
    _require(final_output.get("elements") == operations[2].get("output_elements"), "manifest final output elements drifted")
    _require(_is_safe_relative_path(final_output.get("output_path")), "manifest final output path is invalid")
    _require(_is_nonnegative_int(final_output.get("raw_bytes")), "manifest final output raw byte count is invalid")
    transfer = manifest.get("expected_frontier_transfer")
    _require(isinstance(transfer, Mapping), "manifest transfer expectations are missing")
    for key in (
        "descriptor_package_h2d_bytes", "initial_h2d_bytes", "operation_control_h2d_bytes",
        "inter_wave_d2h_bytes", "inter_wave_h2d_bytes", "final_d2h_bytes",
        "h2d_bytes", "d2h_bytes", "total_bytes",
    ):
        _require(_is_nonnegative_int(transfer.get(key)), f"manifest transfer expectation {key} is invalid")
    _require(
        transfer["h2d_bytes"] == transfer["initial_h2d_bytes"] + transfer["descriptor_package_h2d_bytes"] + transfer["operation_control_h2d_bytes"] + transfer["inter_wave_h2d_bytes"],
        "manifest H2D transfer expectation is inconsistent",
    )
    _require(
        transfer["d2h_bytes"] == transfer["inter_wave_d2h_bytes"] + transfer["final_d2h_bytes"],
        "manifest D2H transfer expectation is inconsistent",
    )
    _require(
        transfer["total_bytes"] == transfer["h2d_bytes"] + transfer["d2h_bytes"],
        "manifest transfer expectation is inconsistent",
    )


def _manifest_final_binding(manifest: Mapping[str, Any]) -> JsonDict:
    outputs = manifest.get("final_outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], Mapping):
        raise ValueError("hardware_profile_violation: frontier requires one real final output")
    output = outputs[0]
    if output.get("component") != "real":
        raise ValueError("hardware_profile_violation: frontier final component must be real")
    return {
        "component": "real",
        "slot_id": output.get("slot_id"),
        "elements": output.get("elements"),
        "output_path": output.get("output_path"),
        "raw_bytes": output.get("raw_bytes"),
        "hash_fnv1a64_required": True,
    }


def _validate_frontier_execution_evidence(response: Mapping[str, Any]) -> None:
    _require(response.get("co_dispatch_confirmed") is True, "co-dispatch was not confirmed")
    _require(response.get("overlap_measured") is False, "overlap measurement is not allowed")
    _require(response.get("overlap_claim") == "unmeasured", "overlap claim is not unmeasured")
    _require(response.get("overlap_evidence") == OVERLAP_EVIDENCE, "overlap evidence is ambiguous")
    _require(response.get("wave0_complete_before_wave1") is True, "wave ordering was not confirmed")
    _require(response.get("completed_task_ids") == list(EXPECTED_TASK_IDS), "completed task identity mismatch")
    _require(response.get("completed_task_ids_scope") == COMPLETION_SCOPE, "completed task ID scope mismatch")
    _require(response.get("barrier_count") == EXPECTED_BARRIER_COUNT, "barrier count mismatch")
    barriers = response.get("barriers")
    _require(
        barriers == [
            {"barrier_index": 0, "wave_index": 0, "completed": True},
            {"barrier_index": 1, "wave_index": 1, "completed": True},
        ],
        "barrier evidence mismatch",
    )
    _require(response.get("observed_dpu_task_counts") == list(EXPECTED_DPU_TASK_COUNTS), "DPU task counts mismatch")


def _validate_transfer_invariant(response: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    transfer = _mapping(response, "transfer")
    _require(transfer.get("transfer_invariant") is True, "native transfer invariant flag is not passed")
    expected = manifest.get("expected_frontier_transfer")
    _require(isinstance(expected, Mapping), "manifest transfer expectations are missing")
    component_keys = (
        "descriptor_package_h2d_bytes", "initial_h2d_bytes", "operation_control_h2d_bytes",
        "inter_wave_d2h_bytes", "inter_wave_h2d_bytes", "final_d2h_bytes",
        "h2d_bytes", "d2h_bytes", "total_bytes",
    )
    for key in component_keys:
        value = transfer.get(key)
        _require(_is_nonnegative_int(value), f"native transfer component {key} is invalid")
        _require(value == expected.get(key), f"native transfer component {key} does not match manifest")
    for response_key, transfer_key in (
        ("actual_h2d_bytes", "h2d_bytes"),
        ("actual_d2h_bytes", "d2h_bytes"),
        ("actual_transfer_bytes", "total_bytes"),
    ):
        value = transfer.get(transfer_key)
        _require(_is_nonnegative_int(value), f"native transfer total {transfer_key} is invalid")
        _require(response.get(response_key) == value, f"top-level native transfer total {response_key} is inconsistent")
        _require(value == expected.get(transfer_key), f"native transfer total {transfer_key} does not match manifest")
    _require(
        transfer["h2d_bytes"] == transfer["initial_h2d_bytes"] + transfer["descriptor_package_h2d_bytes"] + transfer["operation_control_h2d_bytes"] + transfer["inter_wave_h2d_bytes"],
        "native transfer invariant failed",
    )
    _require(
        transfer["d2h_bytes"] == transfer["inter_wave_d2h_bytes"] + transfer["final_d2h_bytes"],
        "native transfer invariant failed",
    )
    _require(
        transfer["total_bytes"] == transfer["h2d_bytes"] + transfer["d2h_bytes"],
        "native transfer invariant failed",
    )
    _require(response.get("transfer_accounting_scope") == TRANSFER_SCOPE, "transfer scope is not native observed")


def _validate_response_tasks(response: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    tasks = response.get("tasks")
    _require(isinstance(tasks, list) and len(tasks) == 3, "native task completion count is invalid")
    seen = [item.get("task_id") if isinstance(item, Mapping) else None for item in tasks]
    _require(seen == list(EXPECTED_TASK_IDS), "native tasks are missing, duplicated, or out of order")
    expected = manifest.get("frontier_task_operations")
    _require(isinstance(expected, list) and len(expected) == 3, "manifest frontier operation plan is invalid")
    for item, expected_item in zip(tasks, expected):
        _require(isinstance(item, Mapping), "native task completion entry is invalid")
        for key in ("wave_index", "dpu_id", "operation_id"):
            _require(item.get(key) == expected_item.get(key), f"native task {key} binding mismatch")
        _require(item.get("completed") is True and item.get("completion_confirmed") is True, "task completion is unconfirmed")


def _validate_final_output(response: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    actual = response.get("final_output")
    expected = manifest.get("final_output_binding")
    _require(isinstance(actual, Mapping) and isinstance(expected, Mapping), "final output binding is missing")
    _require(actual.get("written") is True, "final output was not written")
    for key in ("component", "slot_id", "elements", "raw_bytes"):
        _require(actual.get(key) == expected.get(key), f"final output {key} binding mismatch")
    output_path = actual.get("output_path")
    path = actual.get("path")
    expected_path = expected.get("output_path")
    _require(_is_safe_relative_path(expected_path), "manifest final output path is invalid")
    _require(_is_safe_relative_path(output_path), "native final output_path must be relative")
    _require(output_path == expected_path, "native final output_path does not match manifest binding")
    _require(_is_safe_relative_path(path), "native final output path must be relative")
    _require(path == output_path, "final output path fields disagree")
    digest = actual.get("hash_fnv1a64")
    _require(isinstance(digest, str) and len(digest) == 16, "final output hash is missing")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError("response_evidence_invalid: final output hash is invalid") from exc
    hashes = _mapping(response, "hashes")
    for key in ("manifest_fnv1a64", "package_fnv1a64", "dpu_binary_fnv1a64", "host_source_fnv1a64"):
        value = hashes.get(key)
        _require(isinstance(value, str) and len(value) == 16, f"native hash {key} is missing")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"response_evidence_invalid: native hash {key} is invalid") from exc


def validate_frontier_package_against_manifest(
    package_bytes: bytes, manifest: Mapping[str, Any]
) -> None:
    """Validate resident binary descriptors against the augmented frontier manifest."""

    _validate_manifest_identity(manifest)
    metadata = validate_resident_graph_package_bytes(package_bytes)
    binding = manifest.get("resident_package_binding")
    _require(isinstance(binding, Mapping), "manifest resident package binding is missing")
    _require(binding.get("package_sha256") == hashlib.sha256(package_bytes).hexdigest(), "resident package hash mismatch")
    _require(binding.get("slot_descriptors") == metadata.get("slot_descriptors"), "resident package slot descriptors drifted")
    actual_operations = _resident_package_operations(package_bytes, metadata)
    _require(binding.get("operations") == actual_operations, "resident package operation descriptors drifted")
    operations = manifest.get("frontier_task_operations")
    _require(isinstance(operations, list), "manifest frontier operation plan is missing")
    _require_frontier_package_operations(actual_operations, operations)


def validate_frontier_output_file(
    response: Mapping[str, Any], manifest: Mapping[str, Any], session_root: Path
) -> None:
    """Validate the bound final output bytes and their native FNV-1a64 digest."""

    actual = response.get("final_output")
    expected = manifest.get("final_output_binding")
    _require(isinstance(actual, Mapping) and isinstance(expected, Mapping), "final output binding is missing")
    relative = expected.get("output_path")
    _require(_is_safe_relative_path(relative), "final output path is invalid")
    root = session_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("response_evidence_invalid: final output escapes session root") from exc
    actual = response.get("final_output")
    _require(isinstance(actual, Mapping), "final output binding is missing")
    for key in ("output_path", "path"):
        emitted = actual.get(key)
        _require(_is_safe_relative_path(emitted), f"native final output {key} is not relative")
        _require(emitted == relative, f"native final output {key} does not match manifest binding")
    _require(path.is_file(), "final output file is missing")
    elements = expected.get("elements")
    _require(_is_nonnegative_int(elements), "final output element count is invalid")
    raw = path.read_bytes()
    _require(len(raw) == elements * 4, "final output raw byte length mismatch")
    _require(_fnv1a64(raw) == actual.get("hash_fnv1a64"), "final output FNV-1a64 mismatch")


def _is_safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or Path(value).is_absolute():
        return False
    parts = Path(value).parts
    return bool(parts) and all(part not in (".", "..") for part in parts)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"response_evidence_invalid: {key} evidence is missing")
    return nested


def _resident_package_operations(
    package_bytes: bytes, metadata: Mapping[str, Any]
) -> list[JsonDict]:
    header = struct.unpack_from(RESIDENT_PACKAGE_HEADER_FORMAT, package_bytes, 0)
    operation_offset = int(header[8])
    operation_count = int(header[11])
    operation_format = _resident_operation_format()
    operations: list[JsonDict] = []
    for operation_id in range(operation_count):
        values = struct.unpack_from(
            operation_format,
            package_bytes,
            operation_offset + operation_id * RESIDENT_OPERATION_BYTES,
        )
        kind, mode, output_elements = values[:3]
        slots = values[3:9]
        operations.append(
            {
                "operation_id": operation_id,
                "kind": "contract" if kind == 1 else "complex_combine",
                "mode": "none" if mode == 0 else "per_task_resident_requantize",
                "output_elements": int(output_elements),
                "slot_a": int(slots[0]),
                "slot_b": int(slots[1]),
                "slot_c": int(slots[2]),
                "slot_d": int(slots[3]),
                "slot_out_real": int(slots[4]),
                "slot_out_imag": int(slots[5]),
            }
        )
    _require(len(operations) == int(metadata.get("operation_count", -1)), "resident package operation count mismatch")
    return operations


def _require_frontier_package_operations(
    package_operations: list[Mapping[str, Any]], manifest_operations: list[Mapping[str, Any]]
) -> None:
    _require(len(package_operations) == 3 and len(manifest_operations) == 3, "frontier package operation count mismatch")
    for index, (package_operation, manifest_operation) in enumerate(
        zip(package_operations, manifest_operations, strict=True)
    ):
        _require(package_operation.get("operation_id") == index, "resident package operation ID drifted")
        _require(package_operation.get("kind") == "contract", "frontier package operation kind drifted")
        _require(package_operation.get("mode") == NUMERIC_MODE, "frontier package operation mode drifted")
        _require(manifest_operation.get("operation_id") == index, "frontier manifest operation ID drifted")
        _require(manifest_operation.get("input_slot_ids") == [package_operation.get("slot_a"), package_operation.get("slot_b")], "frontier input slot binding drifted")
        _require(manifest_operation.get("output_slot_id") == package_operation.get("slot_out_real"), "frontier output slot binding drifted")
        _require(manifest_operation.get("output_elements") == package_operation.get("output_elements"), "frontier output element binding drifted")


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonnegative_int(value: Any, label: str) -> int:
    if not _is_nonnegative_int(value):
        raise ValueError(f"hardware_profile_violation: manifest {label} must be a nonnegative integer")
    return int(value)


def _align8(value: int) -> int:
    return (int(value) + 7) & ~7


def _fnv1a64(value: bytes) -> str:
    result = 14695981039346656037
    for byte in value:
        result ^= byte
        result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"response_evidence_invalid: {message}")


# Short aliases keep the route usable from preparation code without coupling it
# to the longer M3.1-specific names.
build_frontier_plan = build_hardware_frontier_plan
build_frontier_graph_package = build_hardware_frontier_graph_package
write_frontier_graph_package = write_hardware_frontier_graph_package
validate_frontier_response = validate_frontier_native_response


__all__ = [
    "BACKEND_ID",
    "COMPLETION_SCOPE",
    "EXPECTED_DPU_TASK_COUNTS",
    "EXPECTED_PATH",
    "EXPECTED_TASK_IDS",
    "HardwareFrontierPlan",
    "NATIVE_SCHEMA",
    "NUMERIC_MODE",
    "PROFILE_ID",
    "REQUEST_SCHEMA",
    "ROUTE_ID",
    "build_frontier_graph_package",
    "build_frontier_plan",
    "build_hardware_frontier_graph_package",
    "build_hardware_frontier_plan",
    "validate_frontier_native_response",
    "validate_frontier_output_file",
    "validate_frontier_package_against_manifest",
    "validate_frontier_package_validation_response",
    "validate_frontier_response",
    "validate_hardware_frontier_graph",
    "validate_hardware_frontier_response",
    "write_frontier_graph_package",
    "write_hardware_frontier_graph_package",
]
