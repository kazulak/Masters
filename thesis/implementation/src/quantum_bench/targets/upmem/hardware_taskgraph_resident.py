"""Contracts and lowering for the bounded one-DPU resident TaskGraph route.

The resident route is intentionally separate from ``hardware_taskgraph`` and
``hardware_session``'s legacy generic-loop protocols.  A graph is lowered to
float32 MRAM slots and an ordered descriptor stream.  The native host receives
one package, allocates one DPU set, and launches one synchronous DPU task per
descriptor.  It never writes an intermediate result file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from quantum_bench.bench.config import load_suite
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict, TaskGraph, TensorSpec, TensorValue, to_jsonable
from quantum_bench.routing.generic_prepare import (
    GENERIC_MODE_FLOAT32_NO_QUANT,
    GenericTaskPreparationCaps,
    generic_loop_reference_float32,
    generic_loop_reference_int32,
    generic_structural_feasibility,
)
from quantum_bench.tn.contract import contract_binary_task
from quantum_bench.tn.execution import order_final_tensor
from quantum_bench.tn.execution_bundle import canonical_hash


UPMEM_HARDWARE_TASKGRAPH_RESIDENT_SUITE_SCHEMA_VERSION = (
    "upmem_hardware_taskgraph_resident_v1"
)
RESIDENT_BACKEND_ID = "upmem_sdk_hardware_taskgraph_resident"
RESIDENT_ROUTE_ID = "upmem_tn_hardware_taskgraph_resident"
RESIDENT_PROFILE_VERSION = "hardware_taskgraph_single_dpu_mram_resident_v1"
RESIDENT_SESSION_PROTOCOL = "generic_loop_resident_graph_session_v1"
RESIDENT_TIMING_SCOPE = "one_dpu_mram_resident_full_taskgraph_v1"
RESIDENT_NUMERIC_MODES = ("none", "per_task_resident_requantize")
RESIDENT_COMPLEX_POLICY = "split_real_imag_float32_dpu_complex_combine"
RESIDENT_MAX_RANK = 16
RESIDENT_MAX_ELEMENTS = 256
RESIDENT_MAX_LOGICAL_TASKS = 32
RESIDENT_MAX_COMPONENT_OPS = 128
RESIDENT_MAX_SLOT_DESCRIPTORS = 128
RESIDENT_MRAM_POOL_BYTES = 512 * 1024
RESIDENT_MAX_CONTRACTED_COMBINATIONS = 256
RESIDENT_OUTPUT_TILE_ELEMENTS = 256
RESIDENT_TIMEOUT_S = 30.0

RESIDENT_PACKAGE_MAGIC = b"UPRGPCK1"
RESIDENT_PACKAGE_VERSION = 1
RESIDENT_PACKAGE_ENDIAN = 0x01020304
RESIDENT_PACKAGE_HEADER_FORMAT = "<8s4I5Q8I"
RESIDENT_PACKAGE_HEADER_BYTES = struct.calcsize(RESIDENT_PACKAGE_HEADER_FORMAT)
RESIDENT_SLOT_FORMAT = "<4I"
RESIDENT_SLOT_BYTES = struct.calcsize(RESIDENT_SLOT_FORMAT)
RESIDENT_INVALID_SLOT = 0xFFFFFFFF
RESIDENT_OPERATION_CONTRACT = 1
RESIDENT_OPERATION_COMPLEX_COMBINE = 2


def _resident_args_format(max_rank: int = RESIDENT_MAX_RANK) -> str:
    return "<" + "I" * (9 + 7 * max_rank) + "i" * (4 * max_rank)


def _resident_operation_format(max_rank: int = RESIDENT_MAX_RANK) -> str:
    # kind, mode, output elements, six slot references, two float scales,
    # followed by the unchanged generic-loop index metadata ABI.
    return "<" + "I" * 9 + "ff" + "I" * (9 + 7 * max_rank) + "i" * (4 * max_rank)


RESIDENT_OPERATION_BYTES = struct.calcsize(_resident_operation_format())


@dataclass(frozen=True)
class HardwareTaskGraphResidentProfile:
    version: str
    target: str
    backend_id: str
    route_id: str
    session_protocol: str
    timing_scope: str
    requested_dpu_count: int
    tasklets_per_dpu: int
    max_rank: int
    max_tensor_elements: int
    max_logical_tasks: int
    max_component_ops: int
    max_slot_descriptors: int
    mram_pool_bytes: int
    max_contracted_combinations: int
    output_tile_elements: int
    numeric_modes: tuple[str, ...]
    complex_policy: str
    synchronous_execution: bool
    timeout_s: float
    performance_claim_applicable: bool

    def to_json_dict(self) -> JsonDict:
        return {
            "hardware_profile_version": self.version,
            "target": self.target,
            "backend_id": self.backend_id,
            "route_id": self.route_id,
            "session_protocol": self.session_protocol,
            "timing_scope": self.timing_scope,
            "requested_dpu_count": self.requested_dpu_count,
            "tasklets_per_dpu": self.tasklets_per_dpu,
            "max_rank": self.max_rank,
            "max_tensor_elements": self.max_tensor_elements,
            "max_logical_tasks": self.max_logical_tasks,
            "max_component_ops": self.max_component_ops,
            "max_slot_descriptors": self.max_slot_descriptors,
            "mram_pool_bytes": self.mram_pool_bytes,
            "max_contracted_combinations": self.max_contracted_combinations,
            "output_tile_elements": self.output_tile_elements,
            "numeric_modes": list(self.numeric_modes),
            "complex_policy": self.complex_policy,
            "synchronous_execution": self.synchronous_execution,
            "timeout_s": self.timeout_s,
            "performance_claim_applicable": self.performance_claim_applicable,
        }


@dataclass(frozen=True)
class HardwareTaskGraphResidentVariant:
    variant_id: str
    label: str
    planner: JsonDict


@dataclass(frozen=True)
class HardwareTaskGraphResidentSuite:
    suite_path: Path
    suite: JsonDict
    profile: HardwareTaskGraphResidentProfile
    variants: tuple[HardwareTaskGraphResidentVariant, ...]


@dataclass(frozen=True)
class ResidentSlotLifetime:
    logical_id: str
    tensor_id: str
    component: str
    elements: int
    start_task: int
    end_task: int
    initial: bool
    final: bool
    sequence: int

    def to_json_dict(self) -> JsonDict:
        return {
            "logical_id": self.logical_id,
            "tensor_id": self.tensor_id,
            "component": self.component,
            "elements": self.elements,
            "start_task": self.start_task,
            "end_task": self.end_task,
            "initial": self.initial,
            "final": self.final,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class ResidentSlotDescriptor:
    slot_id: int
    offset_bytes: int
    capacity_elements: int
    element_count: int
    lifetimes: tuple[ResidentSlotLifetime, ...]

    @property
    def capacity_bytes(self) -> int:
        return _align8(self.capacity_elements * 4)

    @property
    def logical_ids(self) -> tuple[str, ...]:
        return tuple(item.logical_id for item in self.lifetimes)

    def to_json_dict(self) -> JsonDict:
        return {
            "slot_id": self.slot_id,
            "offset_bytes": self.offset_bytes,
            "capacity_elements": self.capacity_elements,
            "capacity_bytes": self.capacity_bytes,
            "element_count": self.element_count,
            "logical_ids": list(self.logical_ids),
            "lifetimes": [item.to_json_dict() for item in self.lifetimes],
        }


@dataclass(frozen=True)
class ResidentOperationDescriptor:
    operation_id: int
    task_id: str
    component: str
    kind: str
    mode: str
    output_elements: int
    slot_a: int
    slot_b: int
    slot_c: int
    slot_d: int
    slot_out_real: int
    slot_out_imag: int
    left_scale: float
    right_scale: float
    args: JsonDict = field(default_factory=dict)

    @property
    def native_kind(self) -> int:
        if self.kind == "contract":
            return RESIDENT_OPERATION_CONTRACT
        if self.kind == "complex_combine":
            return RESIDENT_OPERATION_COMPLEX_COMBINE
        raise ValueError(f"hardware_profile_violation: unknown resident operation kind {self.kind}")

    @property
    def native_mode(self) -> int:
        if self.mode == "none":
            return 0
        if self.mode == "per_task_resident_requantize":
            return 1
        raise ValueError(f"hardware_profile_violation: unknown resident numeric mode {self.mode}")

    def to_json_dict(self) -> JsonDict:
        return {
            "operation_id": self.operation_id,
            "task_id": self.task_id,
            "component": self.component,
            "kind": self.kind,
            "mode": self.mode,
            "output_elements": self.output_elements,
            "slot_a": self.slot_a,
            "slot_b": self.slot_b,
            "slot_c": self.slot_c,
            "slot_d": self.slot_d,
            "slot_out_real": self.slot_out_real,
            "slot_out_imag": self.slot_out_imag,
            "left_scale": self.left_scale,
            "right_scale": self.right_scale,
            "args": self.args,
            "intermediate_output_path": None,
        }

    def to_bytes(self, *, max_rank: int = RESIDENT_MAX_RANK) -> bytes:
        values = _pack_native_args(self.args, mode=GENERIC_MODE_FLOAT32_NO_QUANT)
        return struct.pack(
            _resident_operation_format(max_rank),
            self.native_kind,
            self.native_mode,
            int(self.output_elements),
            int(self.slot_a),
            int(self.slot_b),
            int(self.slot_c),
            int(self.slot_d),
            int(self.slot_out_real),
            int(self.slot_out_imag),
            float(self.left_scale),
            float(self.right_scale),
            *values,
        )


@dataclass(frozen=True)
class ResidentAllocation:
    slots: tuple[ResidentSlotDescriptor, ...]
    lifetimes: tuple[ResidentSlotLifetime, ...]
    logical_to_slot: Mapping[str, int]
    tensor_components: Mapping[str, tuple[str, ...]]
    initial_data: Mapping[int, np.ndarray]
    final_components: tuple[tuple[str, int, int], ...]
    logical_task_count: int
    component_operation_count: int
    mram_pool_bytes: int
    mram_used_bytes: int

    @property
    def slot_descriptor_count(self) -> int:
        return len(self.slots)

    @property
    def peak_resident_bytes(self) -> int:
        # The allocator's physical pool is the strict peak reservation.  It
        # may be larger than the live interval sum because slots are reused.
        return self.mram_used_bytes

    def slot_for(self, tensor_id: str, component: str = "real") -> int:
        return int(self.logical_to_slot[_logical_key(tensor_id, component)])

    def to_json_dict(self) -> JsonDict:
        return {
            "slot_descriptor_count": len(self.slots),
            "slots": [item.to_json_dict() for item in self.slots],
            "slot_lifetime_map": [item.to_json_dict() for item in self.lifetimes],
            "logical_to_slot": dict(self.logical_to_slot),
            "tensor_components": {
                key: list(value) for key, value in self.tensor_components.items()
            },
            "initial_slot_ids": sorted(int(value) for value in self.initial_data),
            "final_components": [
                {"component": name, "slot_id": slot, "elements": elements}
                for name, slot, elements in self.final_components
            ],
            "logical_task_count": self.logical_task_count,
            "component_operation_count": self.component_operation_count,
            "mram_pool_bytes": self.mram_pool_bytes,
            "mram_used_bytes": self.mram_used_bytes,
            "peak_resident_bytes": self.peak_resident_bytes,
        }


@dataclass(frozen=True)
class ResidentGraphPackage:
    graph: TaskGraph
    case_id: str
    suite_id: str
    quantization_mode: str
    allocation: ResidentAllocation
    operations: tuple[ResidentOperationDescriptor, ...]
    initial_data: Mapping[int, np.ndarray]
    full_precision_output: np.ndarray | None = None
    manifest_path: Path | None = None
    package_path: Path | None = None
    final_output_paths: Mapping[str, Path] = field(default_factory=dict)
    descriptor_sha256: str | None = None

    @property
    def graph_request_count(self) -> int:
        return 1

    @property
    def descriptor_count(self) -> int:
        return len(self.operations)

    @property
    def component_operation_count(self) -> int:
        return len(self.operations)

    def to_json_dict(self) -> JsonDict:
        return {
            "schema_version": RESIDENT_SESSION_PROTOCOL,
            "manifest_kind": "resident_graph_package",
            "case_id": self.case_id,
            "suite_id": self.suite_id,
            "route_id": RESIDENT_ROUTE_ID,
            "backend_id": RESIDENT_BACKEND_ID,
            "quantization_mode": self.quantization_mode,
            "graph_request_count": 1,
            "logical_task_count": len(self.graph.tasks),
            "component_operation_count": len(self.operations),
            "slot_descriptors": len(self.allocation.slots),
            "mram_pool_bytes": self.allocation.mram_pool_bytes,
            "mram_used_bytes": self.allocation.mram_used_bytes,
            "final_output_component_count": len(self.allocation.final_components),
            "native_operation_descriptors": [
                item.to_json_dict() for item in self.operations
            ],
            "allocation": self.allocation.to_json_dict(),
            "descriptor_sha256": self.descriptor_sha256,
            "no_host_intermediate_output_files": True,
            "intermediate_output_paths": [],
            "final_output_paths": {
                key: str(value) for key, value in self.final_output_paths.items()
            },
        }

    def write(
        self,
        session_root: Path,
        *,
        dpu_binary: Path,
        request_id: str,
    ) -> "ResidentGraphPackage":
        """Write one native request without creating intermediate output files."""

        root = session_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        safe = _safe_name(request_id)
        request_dir = root / "resident_requests" / safe
        request_dir.mkdir(parents=True, exist_ok=False)
        input_dir = request_dir / "inputs"
        output_dir = request_dir / "outputs"
        input_dir.mkdir()
        output_dir.mkdir()

        input_entries: list[JsonDict] = []
        initial_h2d_bytes = 0
        for slot_id in sorted(self.initial_data):
            array = np.asarray(self.initial_data[slot_id], dtype="<f4").ravel()
            path = input_dir / f"slot_{int(slot_id):04d}.bin"
            array.tofile(path)
            raw_bytes = int(array.nbytes)
            transfer_bytes = _align8(raw_bytes)
            initial_h2d_bytes += transfer_bytes
            input_entries.append(
                {
                    "slot_id": int(slot_id),
                    "elements": int(array.size),
                    "input_path": _relative(root, path),
                    "raw_bytes": raw_bytes,
                    "transfer_bytes": transfer_bytes,
                }
            )

        final_paths: dict[str, Path] = {}
        final_entries: list[JsonDict] = []
        for component, slot_id, elements in self.allocation.final_components:
            path = output_dir / f"final_{_safe_name(component)}.bin"
            final_paths[component] = path
            final_entries.append(
                {
                    "component": component,
                    "slot_id": int(slot_id),
                    "elements": int(elements),
                    "output_path": _relative(root, path),
                    "raw_bytes": int(elements) * 4,
                    "transfer_bytes": _align8(int(elements) * 4),
                }
            )

        package_bytes = _encode_package(self.allocation.slots, self.operations)
        package_path = request_dir / "resident_graph_package.bin"
        package_path.write_bytes(package_bytes)
        descriptor_sha256 = hashlib.sha256(package_bytes).hexdigest()
        descriptor_h2d_bytes = _descriptor_transfer_bytes(
            len(self.allocation.slots), len(self.operations)
        )
        final_d2h_bytes = sum(int(item["transfer_bytes"]) for item in final_entries)
        dpu_ref = _relative(root, dpu_binary)
        manifest_path = root / f"{safe}_resident_request.json"
        payload: JsonDict = {
            "schema_version": RESIDENT_SESSION_PROTOCOL,
            "manifest_kind": "resident_graph_request",
            "session_id": request_id,
            "route_id": RESIDENT_ROUTE_ID,
            "backend_id": RESIDENT_BACKEND_ID,
            "dpu_binary": dpu_ref,
            "package_path": _relative(root, package_path),
            "requested_dpus": 1,
            "tasklets": 1,
            "graph_request_count": 1,
            "logical_task_count": len(self.graph.tasks),
            "component_operation_count": len(self.operations),
            "slot_descriptor_count": len(self.allocation.slots),
            "mram_pool_bytes": self.allocation.mram_pool_bytes,
            "quantization_mode": self.quantization_mode,
            "numeric_policy": {
                "resident_slot_dtype": "float32",
                "mode": self.quantization_mode,
                "requantization": (
                    "dpu_local_per_task_max_abs_over_127_nearest_even_clip_127"
                    if self.quantization_mode == "per_task_resident_requantize"
                    else "none"
                ),
                "saturation_count_observed": False,
            },
            "initial_slots": input_entries,
            "final_outputs": final_entries,
            "initial_h2d_bytes": initial_h2d_bytes,
            "descriptor_h2d_bytes": descriptor_h2d_bytes,
            "descriptor_control_bytes": 16,
            "control_h2d_bytes_per_launch": 8,
            "final_d2h_bytes": final_d2h_bytes,
            "intermediate_h2d_bytes": 0,
            "intermediate_d2h_bytes": 0,
            "no_host_intermediate_output_files": True,
            "intermediate_output_paths": [],
            "package_parse_timing_boundary": (
                "native_process_start_through_validated_binary_package_before_dpu_alloc"
            ),
            "timing_scope": RESIDENT_TIMING_SCOPE,
        }
        write_json(manifest_path, payload)
        return ResidentGraphPackage(
            graph=self.graph,
            case_id=self.case_id,
            suite_id=self.suite_id,
            quantization_mode=self.quantization_mode,
            allocation=self.allocation,
            operations=self.operations,
            initial_data=self.initial_data,
            full_precision_output=self.full_precision_output,
            manifest_path=manifest_path,
            package_path=package_path,
            final_output_paths=final_paths,
            descriptor_sha256=descriptor_sha256,
        )


class ResidentCapacityError(ValueError):
    """Structured unsupported/profile failure for a resident capacity miss."""

    failure_stage = "hardware_profile_violation"


def load_hardware_taskgraph_resident_suite(path: Path) -> HardwareTaskGraphResidentSuite:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("hardware_profile_violation: resident suite must be a mapping")
    suite = load_suite(path)
    metadata = suite.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get(
        "hardware_taskgraph_resident_schema_version"
    ) != UPMEM_HARDWARE_TASKGRAPH_RESIDENT_SUITE_SCHEMA_VERSION:
        raise ValueError("hardware_profile_violation: unsupported resident suite schema")
    profile = _parse_profile(metadata.get("hardware_profile"))
    routes = tuple(str(item) for item in (suite.get("route_policy") or {}).get("routes", ()))
    if routes != (RESIDENT_ROUTE_ID,):
        raise ValueError("hardware_profile_violation: resident suite must contain only its route")
    if int(suite.get("warmups", 0)) != 2 or int(suite.get("repeats", 0)) != 7:
        raise ValueError("hardware_profile_violation: resident suite requires two warmups and seven repeats")
    if len(suite.get("cases", ())) != 13:
        raise ValueError("hardware_profile_violation: resident suite requires the current 13-case matrix")
    for case in suite.get("cases", ()):
        if case.get("hardware_numeric_coverage") not in {"real", "split_complex"}:
            raise ValueError("hardware_profile_violation: each resident workload needs numeric coverage")
    variants = _parse_variants(raw.get("path_variants"))
    return HardwareTaskGraphResidentSuite(path.resolve(), suite, profile, variants)


def hardware_taskgraph_resident_profile_metadata(
    profile: HardwareTaskGraphResidentProfile,
) -> JsonDict:
    return {
        **profile.to_json_dict(),
        "hardware_functionality_evidence": True,
        "hardware_timing_available": True,
        "timing_is_bringup_only": False,
        "hardware_speedup_applicable": False,
        "cross_backend_speedup_applicable": False,
        "native_build_reuse_required": True,
        "graph_request_count_per_record": 1,
        "resident_mram_slots": True,
        "host_rehydrated_equivalent_bytes_separate": True,
        "final_output_only_d2h": True,
        "intermediate_h2d_bytes": 0,
        "intermediate_d2h_bytes": 0,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "multi_dpu_execution": False,
        "claim_boundary": (
            "one-DPU MRAM-resident full-TaskGraph timing and policy-accuracy evidence; "
            "no CPU/GPU speedup, energy, scheduler, or multi-DPU claim"
        ),
    }


def validate_hardware_taskgraph_resident_execution_request(
    *, execute: bool, environment: Mapping[str, str] | None = None
) -> None:
    env = environment or {}
    if not execute:
        raise ValueError("hardware TaskGraph resident execution requires --execute")
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required for physical resident UPMEM execution")
    if env.get("DPU_BACKEND"):
        raise ValueError("DPU_BACKEND must be unset for physical resident TaskGraph execution")


def allocate_resident_slots(
    graph: TaskGraph,
    network: Any,
    *,
    profile: HardwareTaskGraphResidentProfile | None = None,
) -> ResidentAllocation:
    """Build a deterministic interval allocator for float32 resident slots."""

    selected = profile or _canonical_profile()
    tasks = tuple(graph.tasks)
    if not tasks:
        raise ResidentCapacityError("hardware_profile_violation: empty_task_graph_not_supported")
    if len(tasks) > selected.max_logical_tasks:
        raise ResidentCapacityError("hardware_profile_violation: logical_task_cap_exceeded")

    arrays = {tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    known_components: dict[str, tuple[str, ...]] = {
        tensor_id: _components_for(array) for tensor_id, array in arrays.items()
    }
    consumers: dict[str, list[int]] = {tensor_id: [] for tensor_id in arrays}
    task_outputs: set[str] = set()
    seen_tasks: set[str] = set()
    max_rank = 0
    component_ops = 0
    for task_index, task in enumerate(tasks):
        if task.id in seen_tasks:
            raise ResidentCapacityError("hardware_profile_violation: duplicate_task_id")
        seen_tasks.add(task.id)
        if any(dep not in seen_tasks for dep in task.dependencies):
            raise ResidentCapacityError("hardware_profile_violation: task_graph_not_topological")
        for tensor_id, shape in zip(task.input_tensor_ids, task.input_shapes):
            if tensor_id not in known_components:
                raise ResidentCapacityError(
                    f"hardware_profile_violation: missing_input_tensor:{tensor_id}"
                )
            consumers.setdefault(tensor_id, []).append(task_index)
            _check_shape_caps(shape, selected)
            max_rank = max(max_rank, len(shape))
        _check_shape_caps(task.output_shape, selected)
        max_rank = max(max_rank, len(task.output_shape), len(task.contracted_labels))
        structural = generic_structural_feasibility(
            task,
            GenericTaskPreparationCaps(
                max_rank=selected.max_rank,
                max_tensor_elements=selected.max_tensor_elements,
                max_contracted_combinations=selected.max_contracted_combinations,
            ),
            check_int32_accumulation=True,
        )
        if not structural.feasible:
            raise ResidentCapacityError(
                "hardware_profile_violation: "
                + (structural.reason or "generic_structural_rejection")
            )
        if task.output_tensor_id in known_components or task.output_tensor_id in task_outputs:
            raise ResidentCapacityError("hardware_profile_violation: duplicate_output_tensor_id")
        input_complex = any(len(known_components[item]) == 2 for item in task.input_tensor_ids)
        known_components[task.output_tensor_id] = ("real", "imag") if input_complex else ("real",)
        task_outputs.add(task.output_tensor_id)
        consumers.setdefault(task.output_tensor_id, [])
        component_ops += 5 if input_complex else 1

    if component_ops > selected.max_component_ops:
        raise ResidentCapacityError("hardware_profile_violation: component_operation_cap_exceeded")
    final_tensor_id = tasks[-1].output_tensor_id
    final_output_components = known_components[final_tensor_id]
    lifetimes: list[ResidentSlotLifetime] = []
    initial_data_logical: dict[str, np.ndarray] = {}
    sequence = 0
    for tensor in network.tensors:
        tensor_id = tensor.spec.id
        if tensor_id not in consumers or not consumers[tensor_id]:
            continue
        array = arrays[tensor_id]
        end = max(consumers[tensor_id])
        for component in known_components[tensor_id]:
            value = _component_array(array, component)
            logical_id = _logical_key(tensor_id, component)
            lifetimes.append(
                ResidentSlotLifetime(
                    logical_id, tensor_id, component, int(value.size), 0, end,
                    True, tensor_id == final_tensor_id, sequence,
                )
            )
            initial_data_logical[logical_id] = np.asarray(value, dtype=np.float32)
            sequence += 1

    for task_index, task in enumerate(tasks):
        output_components = known_components[task.output_tensor_id]
        end = max(consumers.get(task.output_tensor_id, [task_index]) or [task_index])
        if task.output_tensor_id == final_tensor_id:
            end = len(tasks) - 1
        for component in output_components:
            logical_id = _logical_key(task.output_tensor_id, component)
            lifetimes.append(
                ResidentSlotLifetime(
                    logical_id, task.output_tensor_id, component,
                    int(np.prod(task.output_shape)), task_index, end,
                    False, task.output_tensor_id == final_tensor_id, sequence,
                )
            )
            sequence += 1
        if len(output_components) == 2:
            for component in ("ar_br", "ai_bi", "ar_bi", "ai_br"):
                lifetimes.append(
                    ResidentSlotLifetime(
                        _logical_key(f"{task.id}__component_{component}", "real"),
                        f"{task.id}__component_{component}", "real",
                        int(np.prod(task.output_shape)), task_index, task_index,
                        False, False, sequence,
                    )
                )
                sequence += 1

    # Stable first-fit by topological start, then source sequence.  Inclusive
    # intervals mean a producer never aliases an input still needed by it.
    ordered = sorted(lifetimes, key=lambda item: (item.start_task, item.sequence))
    slot_lifetimes: list[list[ResidentSlotLifetime]] = []
    slot_capacity: list[int] = []
    logical_to_slot: dict[str, int] = {}
    for lifetime in ordered:
        chosen: int | None = None
        for slot_id, existing in enumerate(slot_lifetimes):
            if slot_capacity[slot_id] < lifetime.elements:
                continue
            if all(
                lifetime.end_task < other.start_task
                or lifetime.start_task > other.end_task
                for other in existing
            ):
                chosen = slot_id
                break
        if chosen is None:
            chosen = len(slot_lifetimes)
            slot_lifetimes.append([])
            slot_capacity.append(0)
        slot_lifetimes[chosen].append(lifetime)
        slot_capacity[chosen] = max(slot_capacity[chosen], lifetime.elements)
        logical_to_slot[lifetime.logical_id] = chosen

    offset = 0
    slots: list[ResidentSlotDescriptor] = []
    for slot_id, entries in enumerate(slot_lifetimes):
        offset = _align8(offset)
        capacity = slot_capacity[slot_id]
        slots.append(
            ResidentSlotDescriptor(
                slot_id, offset, capacity, capacity, tuple(sorted(entries, key=lambda x: x.sequence))
            )
        )
        offset += _align8(capacity * 4)
    used = offset
    if len(slots) > selected.max_slot_descriptors:
        raise ResidentCapacityError("hardware_profile_violation: slot_descriptor_cap_exceeded")
    if used > selected.mram_pool_bytes:
        raise ResidentCapacityError(
            f"hardware_profile_violation: resident_mram_capacity_exceeded:{used}>{selected.mram_pool_bytes}"
        )

    initial_data: dict[int, np.ndarray] = {}
    for logical_id, array in initial_data_logical.items():
        slot_id = logical_to_slot[logical_id]
        if slot_id in initial_data:
            raise ResidentCapacityError("hardware_profile_violation: overlapping_initial_slot_assignment")
        initial_data[slot_id] = np.asarray(array, dtype=np.float32)
    final_components = tuple(
        (
            component,
            logical_to_slot[_logical_key(final_tensor_id, component)],
            int(np.prod(tasks[-1].output_shape)),
        )
        for component in final_output_components
    )
    return ResidentAllocation(
        tuple(slots), tuple(lifetimes), logical_to_slot, known_components,
        initial_data, final_components, len(tasks), component_ops,
        selected.mram_pool_bytes, used,
    )


def build_resident_graph_package(
    graph: TaskGraph,
    network: Any,
    *,
    case_id: str,
    suite_id: str,
    quantization_mode: str,
    profile: HardwareTaskGraphResidentProfile | None = None,
    full_precision_output: np.ndarray | None = None,
) -> ResidentGraphPackage:
    selected = profile or _canonical_profile()
    if quantization_mode not in selected.numeric_modes:
        raise ResidentCapacityError("hardware_profile_violation: unsupported_numeric_mode")
    allocation = allocate_resident_slots(graph, network, profile=selected)
    operations: list[ResidentOperationDescriptor] = []
    component_index = 0
    component_counts = allocation.tensor_components
    for task_index, task in enumerate(graph.tasks):
        left_components = component_counts[task.input_tensor_ids[0]]
        right_components = component_counts[task.input_tensor_ids[1]]
        complex_task = len(left_components) == 2 or len(right_components) == 2
        if not complex_task:
            operations.append(
                _contract_operation(
                    task, component_index, "real", quantization_mode,
                    allocation.slot_for(task.input_tensor_ids[0]),
                    allocation.slot_for(task.input_tensor_ids[1]),
                    allocation.slot_for(task.output_tensor_id), selected,
                )
            )
            component_index += 1
            continue
        left_real = allocation.slot_for(task.input_tensor_ids[0], "real")
        right_real = allocation.slot_for(task.input_tensor_ids[1], "real")
        left_imag = allocation.logical_to_slot.get(_logical_key(task.input_tensor_ids[0], "imag"), RESIDENT_INVALID_SLOT)
        right_imag = allocation.logical_to_slot.get(_logical_key(task.input_tensor_ids[1], "imag"), RESIDENT_INVALID_SLOT)
        temp_slots: dict[str, int] = {}
        for name in ("ar_br", "ai_bi", "ar_bi", "ai_br"):
            temp_slots[name] = allocation.slot_for(f"{task.id}__component_{name}")
        pairs = {
            "ar_br": (left_real, right_real),
            "ai_bi": (left_imag, right_imag),
            "ar_bi": (left_real, right_imag),
            "ai_br": (left_imag, right_real),
        }
        for name in ("ar_br", "ai_bi", "ar_bi", "ai_br"):
            operations.append(
                _contract_operation(
                    task, component_index, name, quantization_mode,
                    pairs[name][0], pairs[name][1], temp_slots[name], selected,
                )
            )
            component_index += 1
        operations.append(
            ResidentOperationDescriptor(
                operation_id=component_index,
                task_id=task.id,
                component="complex_combine",
                kind="complex_combine",
                mode=quantization_mode,
                output_elements=int(np.prod(task.output_shape)),
                slot_a=temp_slots["ar_br"],
                slot_b=temp_slots["ai_bi"],
                slot_c=temp_slots["ar_bi"],
                slot_d=temp_slots["ai_br"],
                slot_out_real=allocation.slot_for(task.output_tensor_id, "real"),
                slot_out_imag=allocation.slot_for(task.output_tensor_id, "imag"),
                left_scale=1.0,
                right_scale=1.0,
                args={},
            )
        )
        component_index += 1
    if len(operations) != allocation.component_operation_count:
        raise ResidentCapacityError("hardware_profile_violation: component_operation_count_mismatch")
    return ResidentGraphPackage(
        graph=graph,
        case_id=case_id,
        suite_id=suite_id,
        quantization_mode=quantization_mode,
        allocation=allocation,
        operations=tuple(operations),
        initial_data=allocation.initial_data,
        full_precision_output=full_precision_output,
    )


def resident_round_nearest_even(values: Any) -> np.ndarray:
    """Return explicit ties-to-even values before int8 clipping."""

    array = np.asarray(values, dtype=np.float32)
    lower = np.floor(array)
    fraction = array - lower
    rounded = np.where(
        fraction < 0.5,
        lower,
        np.where(
            fraction > 0.5,
            lower + 1.0,
            np.where((np.mod(lower, 2.0) == 0.0), lower, lower + 1.0),
        ),
    )
    return np.asarray(rounded, dtype=np.float32)


def resident_requantize(values: Any) -> tuple[np.ndarray, float, int]:
    array = np.asarray(values, dtype=np.float32)
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    scale = 1.0 if max_abs == 0.0 else max_abs / 127.0
    scale32 = float(np.float32(scale))
    rounded = resident_round_nearest_even(array / np.float32(scale32))
    saturation = int(np.count_nonzero((rounded < -127.0) | (rounded > 127.0)))
    quantized = np.asarray(np.clip(rounded, -127.0, 127.0), dtype=np.int8)
    return quantized, scale32, saturation


def build_resident_policy_reference(
    graph: TaskGraph,
    network: Any,
    *,
    quantization_mode: str,
    profile: HardwareTaskGraphResidentProfile | None = None,
) -> JsonDict:
    """Mirror the DPU resident arithmetic using float32 canonical slots."""

    selected = profile or _canonical_profile()
    if quantization_mode not in selected.numeric_modes:
        raise ValueError("hardware_profile_violation: unsupported_numeric_mode")
    caps = GenericTaskPreparationCaps(
        max_rank=selected.max_rank,
        max_tensor_elements=selected.max_tensor_elements,
        max_contracted_combinations=selected.max_contracted_combinations,
    )
    tensors = {tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    task_metrics: list[JsonDict] = []
    for task_index, task in enumerate(graph.tasks):
        left = TensorValue(_spec_for(task.input_tensor_ids[0], labels, tensors, task.input_shapes[0]), tensors[task.input_tensor_ids[0]])
        right = TensorValue(_spec_for(task.input_tensor_ids[1], labels, tensors, task.input_shapes[1]), tensors[task.input_tensor_ids[1]])
        complex_task = _has_nonzero_imaginary(left.array) or _has_nonzero_imaginary(right.array)
        components = _split_components(left.array, right.array) if complex_task else {"real": (_real(left.array), _real(right.array))}
        outputs: dict[str, np.ndarray] = {}
        component_metrics: dict[str, JsonDict] = {}
        for component, (left_part, right_part) in components.items():
            output, metric = _resident_policy_component(
                task, left, right, left_part, right_part, quantization_mode, caps
            )
            outputs[component] = output
            component_metrics[component] = metric
        if complex_task:
            result = np.asarray(
                (outputs["ar_br"] - outputs["ai_bi"])
                + 1j * (outputs["ar_bi"] + outputs["ai_br"]),
                dtype=np.complex64,
            )
        else:
            result = np.asarray(outputs["real"], dtype=np.float32)
        tensors[task.output_tensor_id] = result
        labels[task.output_tensor_id] = task.output_labels
        task_metrics.append(
            {
                "task_id": task.id,
                "task_index": task_index,
                "complex_representation": "split_real_imag" if complex_task else "real",
                "split_complex_component_count": len(components) if complex_task else 0,
                "component_metrics": component_metrics,
                "output_dtype": str(result.dtype),
                "output_hash": _array_hash(result),
            }
        )
    if not graph.tasks:
        raise ValueError("hardware_profile_violation: empty_task_graph_not_supported")
    final = tensors[graph.tasks[-1].output_tensor_id]
    ordered = order_final_tensor(np.asarray(final), graph.tasks[-1].output_labels, graph.network.output_labels)[0]
    return {
        "status": "completed",
        "reference_kind": "cpu_resident_policy_reference",
        "quantization_mode": quantization_mode,
        "resident_slot_dtype": "float32",
        "dpu_local_requantization": quantization_mode == "per_task_resident_requantize",
        "rounding": "nearest_even",
        "clip_range": [-127, 127],
        "scale_formula": "max_abs/127_or_1_for_all_zero",
        "saturation_observed": True,
        "task_metrics": task_metrics,
        "output": np.asarray(ordered),
        "output_hash": _array_hash(ordered),
    }


def validate_resident_graph_package_bytes(
    payload: bytes,
    *,
    profile: HardwareTaskGraphResidentProfile | None = None,
) -> JsonDict:
    """Validate binary package framing and every slot interval before upload."""

    selected = profile or _canonical_profile()
    if len(payload) < RESIDENT_PACKAGE_HEADER_BYTES:
        raise ValueError("hardware_profile_violation: resident_package_truncated_header")
    header = struct.unpack_from(RESIDENT_PACKAGE_HEADER_FORMAT, payload, 0)
    (
        magic, version, endian, header_bytes, flags, file_bytes,
        slot_offset, slot_bytes, operation_offset, operation_bytes,
        slot_count, operation_count, pool_bytes, request_count,
        initial_count, final_count, max_rank, reserved,
    ) = header
    del flags, reserved
    if magic != RESIDENT_PACKAGE_MAGIC:
        raise ValueError("hardware_profile_violation: resident_package_bad_magic")
    if version != RESIDENT_PACKAGE_VERSION:
        raise ValueError("hardware_profile_violation: resident_package_bad_version")
    if endian != RESIDENT_PACKAGE_ENDIAN:
        raise ValueError("hardware_profile_violation: resident_package_bad_endian")
    if header_bytes != RESIDENT_PACKAGE_HEADER_BYTES or header_bytes % 8:
        raise ValueError("hardware_profile_violation: resident_package_bad_header_alignment")
    if file_bytes != len(payload):
        raise ValueError("hardware_profile_violation: resident_package_file_length_mismatch")
    if any(value % 8 for value in (slot_offset, slot_bytes, operation_offset, operation_bytes)):
        raise ValueError("hardware_profile_violation: resident_package_unaligned_section")
    if slot_offset < header_bytes or slot_offset + slot_bytes > file_bytes:
        raise ValueError("hardware_profile_violation: resident_package_slot_section_overflow")
    if operation_offset < slot_offset + slot_bytes or operation_offset + operation_bytes != file_bytes:
        raise ValueError("hardware_profile_violation: resident_package_operation_section_overlap")
    if slot_bytes != slot_count * RESIDENT_SLOT_BYTES:
        raise ValueError("hardware_profile_violation: resident_package_slot_length_mismatch")
    if operation_bytes != operation_count * RESIDENT_OPERATION_BYTES:
        raise ValueError("hardware_profile_violation: resident_package_operation_length_mismatch")
    if slot_count > selected.max_slot_descriptors:
        raise ValueError("hardware_profile_violation: slot_descriptor_cap_exceeded")
    if operation_count > selected.max_component_ops:
        raise ValueError("hardware_profile_violation: component_operation_cap_exceeded")
    if request_count != 1:
        raise ValueError("hardware_profile_violation: resident_package_graph_request_count_must_be_one")
    if pool_bytes != selected.mram_pool_bytes:
        raise ValueError("hardware_profile_violation: resident_package_mram_pool_mismatch")
    if max_rank != selected.max_rank:
        raise ValueError("hardware_profile_violation: resident_package_rank_profile_mismatch")
    if initial_count > slot_count or final_count == 0 or final_count > 2:
        raise ValueError("hardware_profile_violation: resident_package_output_count_invalid")
    slots: list[tuple[int, int, int, int]] = []
    for index in range(slot_count):
        slot = struct.unpack_from(RESIDENT_SLOT_FORMAT, payload, slot_offset + index * RESIDENT_SLOT_BYTES)
        slot_id, offset, capacity, elements = slot
        if slot_id != index:
            raise ValueError("hardware_profile_violation: resident_package_slot_ids_not_dense")
        if offset % 8 or capacity == 0 or elements == 0 or elements > capacity:
            raise ValueError("hardware_profile_violation: resident_package_slot_descriptor_invalid")
        if offset + _align8(capacity * 4) > pool_bytes:
            raise ValueError("hardware_profile_violation: resident_package_slot_overflow")
        slots.append(slot)
    for left, right in zip(sorted(slots, key=lambda item: item[1]), sorted(slots, key=lambda item: item[1])[1:]):
        if left[1] + _align8(left[2] * 4) > right[1]:
            raise ValueError("hardware_profile_violation: resident_package_slot_overlap")
    for index in range(operation_count):
        operation = struct.unpack_from(
            _resident_operation_format(selected.max_rank),
            payload,
            operation_offset + index * RESIDENT_OPERATION_BYTES,
        )
        kind, mode, output_elements = operation[:3]
        refs = operation[3:9]
        if kind not in {RESIDENT_OPERATION_CONTRACT, RESIDENT_OPERATION_COMPLEX_COMBINE}:
            raise ValueError("hardware_profile_violation: resident_package_operation_kind_invalid")
        if mode not in {0, 1}:
            raise ValueError("hardware_profile_violation: resident_package_operation_mode_invalid")
        if output_elements == 0:
            raise ValueError("hardware_profile_violation: resident_package_operation_empty_output")
        for slot_id in refs:
            if slot_id != RESIDENT_INVALID_SLOT and slot_id >= slot_count:
                raise ValueError("hardware_profile_violation: resident_package_operation_slot_reference_invalid")
    return {
        "magic": magic.decode("ascii"),
        "version": version,
        "endian": endian,
        "file_bytes": file_bytes,
        "slot_count": slot_count,
        "operation_count": operation_count,
        "pool_bytes": pool_bytes,
        "graph_request_count": request_count,
        "initial_slot_count": initial_count,
        "final_output_component_count": final_count,
        "max_rank": max_rank,
    }


def validate_resident_graph_package_file(path: Path, *, profile: HardwareTaskGraphResidentProfile | None = None) -> JsonDict:
    return validate_resident_graph_package_bytes(path.read_bytes(), profile=profile)


def resident_tile_ranges(element_count: int, tile_elements: int = RESIDENT_OUTPUT_TILE_ELEMENTS) -> tuple[tuple[int, int], ...]:
    if element_count < 0 or tile_elements <= 0:
        raise ValueError("hardware_profile_violation: invalid resident tile dimensions")
    return tuple(
        (start, min(element_count, start + tile_elements) - 1)
        for start in range(0, element_count, tile_elements)
    )


def _contract_operation(task, operation_id, component, mode, left_slot, right_slot, output_slot, profile):
    metadata = generic_structural_feasibility(
        task,
        GenericTaskPreparationCaps(
            max_rank=profile.max_rank,
            max_tensor_elements=profile.max_tensor_elements,
            max_contracted_combinations=profile.max_contracted_combinations,
        ),
        check_int32_accumulation=True,
    )
    if not metadata.feasible:
        raise ResidentCapacityError("hardware_profile_violation: " + (metadata.reason or "generic_structural_rejection"))
    return ResidentOperationDescriptor(
        operation_id=operation_id,
        task_id=task.id,
        component=component,
        kind="contract",
        mode=mode,
        output_elements=int(np.prod(task.output_shape)),
        slot_a=left_slot,
        slot_b=right_slot,
        slot_c=RESIDENT_INVALID_SLOT,
        slot_d=RESIDENT_INVALID_SLOT,
        slot_out_real=output_slot,
        slot_out_imag=RESIDENT_INVALID_SLOT,
        left_scale=0.0,
        right_scale=0.0,
        args={
            "left_rank": len(task.input_shapes[0]),
            "right_rank": len(task.input_shapes[1]),
            "output_rank": len(task.output_shape),
            "contracted_rank": len(task.contracted_labels),
            "left_shape": tuple(int(value) for value in task.input_shapes[0]),
            "right_shape": tuple(int(value) for value in task.input_shapes[1]),
            "output_shape": tuple(int(value) for value in task.output_shape),
            **metadata.metadata,
        },
    )


def _resident_policy_component(task, left, right, left_part, right_part, mode, caps):
    left_value = np.asarray(left_part, dtype=np.float32)
    right_value = np.asarray(right_part, dtype=np.float32)
    if mode == "none":
        output = generic_loop_reference_float32(
            left_value,
            right_value,
            output_shape=task.output_shape,
            **_index_args(task, caps),
        )
        return np.asarray(output, dtype=np.float32), {
            "mode": mode,
            "left_scale": None,
            "right_scale": None,
            "left_saturation_count": 0,
            "right_saturation_count": 0,
        }
    left_q, left_scale, left_sat = resident_requantize(left_value)
    right_q, right_scale, right_sat = resident_requantize(right_value)
    raw = generic_loop_reference_int32(
        left_q,
        right_q,
        output_shape=task.output_shape,
        **_index_args(task, caps),
    )
    output = np.asarray(raw.astype(np.float32) * np.float32(left_scale * right_scale), dtype=np.float32)
    return output, {
        "mode": mode,
        "left_scale": left_scale,
        "right_scale": right_scale,
        "left_saturation_count": left_sat,
        "right_saturation_count": right_sat,
    }


def _index_args(task, caps):
    result = generic_structural_feasibility(task, caps, check_int32_accumulation=True)
    if not result.feasible:
        raise ResidentCapacityError("hardware_profile_violation: " + (result.reason or "generic_structural_rejection"))
    return {
        key: result.metadata[key]
        for key in (
            "left_strides", "right_strides", "output_strides", "output_to_left_axes",
            "output_to_right_axes", "contracted_to_left_axes", "contracted_to_right_axes",
            "contracted_dims",
        )
    }


def _pack_native_args(args: Mapping[str, Any], *, mode: str, max_rank: int = RESIDENT_MAX_RANK) -> tuple[int, ...]:
    def unsigned(name: str) -> list[int]:
        return ([int(value) for value in args.get(name, ())] + [0] * max_rank)[:max_rank]

    def signed(name: str) -> list[int]:
        return ([int(value) for value in args.get(name, ())] + [-1] * max_rank)[:max_rank]

    fields = [
        int(args.get("left_rank", 0)), int(args.get("right_rank", 0)),
        int(args.get("output_rank", 0)), int(args.get("contracted_rank", 0)),
        _product(tuple(int(value) for value in args.get("left_shape", ()))),
        _product(tuple(int(value) for value in args.get("right_shape", ()))),
        int(args.get("output_element_count", 0)),
        int(args.get("contracted_combination_count", 0)),
        1 if mode == GENERIC_MODE_FLOAT32_NO_QUANT else 0,
    ]
    for name in ("left_shape", "right_shape", "output_shape", "contracted_dims", "left_strides", "right_strides", "output_strides"):
        fields.extend(unsigned(name))
    signed_fields: list[int] = []
    for name in ("output_to_left_axes", "output_to_right_axes", "contracted_to_left_axes", "contracted_to_right_axes"):
        signed_fields.extend(signed(name))
    return tuple(fields + signed_fields)


def _encode_package(slots: Sequence[ResidentSlotDescriptor], operations: Sequence[ResidentOperationDescriptor]) -> bytes:
    slot_payload = b"".join(
        struct.pack(RESIDENT_SLOT_FORMAT, item.slot_id, item.offset_bytes, item.capacity_elements, item.element_count)
        for item in slots
    )
    slot_offset = RESIDENT_PACKAGE_HEADER_BYTES
    operation_offset = _align8(slot_offset + len(slot_payload))
    operation_payload = b"".join(item.to_bytes() for item in operations)
    file_bytes = operation_offset + len(operation_payload)
    header = struct.pack(
        RESIDENT_PACKAGE_HEADER_FORMAT,
        RESIDENT_PACKAGE_MAGIC,
        RESIDENT_PACKAGE_VERSION,
        RESIDENT_PACKAGE_ENDIAN,
        RESIDENT_PACKAGE_HEADER_BYTES,
        0,
        file_bytes,
        slot_offset,
        len(slot_payload),
        operation_offset,
        len(operation_payload),
        len(slots),
        len(operations),
        _canonical_profile().mram_pool_bytes,
        1,
        sum(1 for item in slots if any(lifetime.initial for lifetime in item.lifetimes)),
        sum(1 for item in slots if any(lifetime.final for lifetime in item.lifetimes)),
        _canonical_profile().max_rank,
        0,
    )
    return header + slot_payload + (b"\0" * (operation_offset - slot_offset - len(slot_payload))) + operation_payload


def _descriptor_transfer_bytes(slot_count: int, operation_count: int) -> int:
    # The resident control block is 16 bytes and every active-operation
    # update is an 8-byte transfer.  No native control write is 4-byte sized.
    return (
        _align8(slot_count * RESIDENT_SLOT_BYTES)
        + _align8(operation_count * RESIDENT_OPERATION_BYTES)
        + 16
    )


def _canonical_profile() -> HardwareTaskGraphResidentProfile:
    return HardwareTaskGraphResidentProfile(
        version=RESIDENT_PROFILE_VERSION,
        target="hardware",
        backend_id=RESIDENT_BACKEND_ID,
        route_id=RESIDENT_ROUTE_ID,
        session_protocol=RESIDENT_SESSION_PROTOCOL,
        timing_scope=RESIDENT_TIMING_SCOPE,
        requested_dpu_count=1,
        tasklets_per_dpu=1,
        max_rank=RESIDENT_MAX_RANK,
        max_tensor_elements=RESIDENT_MAX_ELEMENTS,
        max_logical_tasks=RESIDENT_MAX_LOGICAL_TASKS,
        max_component_ops=RESIDENT_MAX_COMPONENT_OPS,
        max_slot_descriptors=RESIDENT_MAX_SLOT_DESCRIPTORS,
        mram_pool_bytes=RESIDENT_MRAM_POOL_BYTES,
        max_contracted_combinations=RESIDENT_MAX_CONTRACTED_COMBINATIONS,
        output_tile_elements=RESIDENT_OUTPUT_TILE_ELEMENTS,
        numeric_modes=RESIDENT_NUMERIC_MODES,
        complex_policy=RESIDENT_COMPLEX_POLICY,
        synchronous_execution=True,
        timeout_s=RESIDENT_TIMEOUT_S,
        performance_claim_applicable=False,
    )


def _parse_profile(value: object) -> HardwareTaskGraphResidentProfile:
    if not isinstance(value, Mapping):
        raise ValueError("hardware_profile_violation: resident hardware_profile must be a mapping")
    expected = _canonical_profile().to_json_dict()
    if set(value) != set(expected):
        raise ValueError("hardware_profile_violation: resident profile keys differ")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"hardware_profile_violation: {key} must be {expected_value!r}")
    return _canonical_profile()


def _parse_variants(value: object) -> tuple[HardwareTaskGraphResidentVariant, ...]:
    expected_ids = ("opt_einsum_greedy", "custom_upmem_v2_balanced")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("hardware_profile_violation: resident suite requires exactly two path variants")
    parsed = []
    expected_planners = {
        "opt_einsum_greedy": {"engine": "opt_einsum", "optimize": "greedy"},
        "custom_upmem_v2_balanced": {
            "engine": "custom_upmem", "algorithm": "greedy", "objective_version": "upmem_path_cost_v2",
            "selection_scope": "projected_prefix", "weight_profile": "balanced_literature_informed",
            "normalization": "fixed_log1p_generic_budgets_v2", "execution_policy": "generic_single_dpu_split_complex_v2",
        },
    }
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"id", "label", "planner"}:
            raise ValueError("hardware_profile_violation: invalid resident path variant fields")
        variant_id = str(item["id"])
        if variant_id not in expected_ids or item["planner"] != expected_planners[variant_id]:
            raise ValueError(f"hardware_profile_violation: planner for {variant_id} is not fixed")
        parsed.append(HardwareTaskGraphResidentVariant(variant_id, str(item["label"]), dict(item["planner"])))
    if tuple(item.variant_id for item in parsed) != expected_ids:
        raise ValueError("hardware_profile_violation: unsupported resident path variant IDs")
    return tuple(parsed)


def _check_shape_caps(shape: Sequence[int], profile: HardwareTaskGraphResidentProfile) -> None:
    if len(shape) > profile.max_rank:
        raise ResidentCapacityError("hardware_profile_violation: rank_cap_exceeded")
    if _product(tuple(int(value) for value in shape)) > profile.max_tensor_elements:
        raise ResidentCapacityError("hardware_profile_violation: element_count_cap_exceeded")


def _components_for(array: np.ndarray) -> tuple[str, ...]:
    return ("real", "imag") if _has_nonzero_imaginary(array) else ("real",)


def _has_nonzero_imaginary(value: Any) -> bool:
    array = np.asarray(value)
    return bool(np.iscomplexobj(array) and np.any(np.asarray(array.imag) != 0.0))


def _component_array(array: np.ndarray, component: str) -> np.ndarray:
    if component == "real":
        return np.asarray(array.real if np.iscomplexobj(array) else array, dtype=np.float32)
    if component == "imag":
        return np.asarray(array.imag if np.iscomplexobj(array) else np.zeros_like(array), dtype=np.float32)
    raise ValueError(f"unsupported component {component}")


def _split_components(left: Any, right: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    left_real = _real(left_array)
    right_real = _real(right_array)
    left_imag = np.asarray(left_array.imag if np.iscomplexobj(left_array) else np.zeros_like(left_real), dtype=np.float32)
    right_imag = np.asarray(right_array.imag if np.iscomplexobj(right_array) else np.zeros_like(right_real), dtype=np.float32)
    return {
        "ar_br": (left_real, right_real), "ai_bi": (left_imag, right_imag),
        "ar_bi": (left_real, right_imag), "ai_br": (left_imag, right_real),
    }


def _real(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return np.asarray(array.real if np.iscomplexobj(array) else array, dtype=np.float32)


def _spec_for(tensor_id: str, labels: Mapping[str, tuple[int, ...]], tensors: Mapping[str, np.ndarray], shape: Sequence[int]) -> TensorSpec:
    array = tensors[tensor_id]
    return TensorSpec(tensor_id, labels[tensor_id], tuple(int(item) for item in shape), "dense", dtype=str(array.dtype))


def _logical_key(tensor_id: str, component: str) -> str:
    return f"{tensor_id}::{component}"


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _align8(value: int) -> int:
    if value < 0:
        raise ValueError("hardware_profile_violation: negative resident byte count")
    return (int(value) + 7) & ~7


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("hardware_profile_violation: resident package path must be under session root") from exc


def _array_hash_json(value: Any) -> str:
    return canonical_hash({"array_hash": _array_hash(value)})


# Public aliases make the allocator/package boundary easy to test without
# coupling tests to the route's CLI entry point.
build_resident_slot_lifetime_map = allocate_resident_slots
build_resident_package = build_resident_graph_package
validate_resident_package = validate_resident_graph_package_bytes
