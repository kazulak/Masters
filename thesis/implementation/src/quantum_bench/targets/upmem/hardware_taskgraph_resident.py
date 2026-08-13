"""Contracts and lowering for the bounded one-DPU resident TaskGraph route.

The resident route is intentionally separate from ``hardware_taskgraph`` and
``hardware_session``'s legacy generic-loop protocols.  A graph is lowered to
float32 MRAM slots and an ordered descriptor stream.  The native host receives
one package, allocates one DPU set, and launches one synchronous DPU task per
descriptor.  It never writes an intermediate result file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from quantum_bench.bench.config import load_suite
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict, TaskGraph, TensorSpec, TensorValue
from quantum_bench.routing.generic_prepare import (
    GENERIC_MODE_FLOAT32_NO_QUANT,
    GenericTaskPreparationCaps,
    generic_loop_reference_float32,
    generic_loop_reference_int32,
    generic_structural_feasibility,
)
from quantum_bench.tn.execution import order_final_tensor
from quantum_bench.tn.execution_bundle import canonical_hash


UPMEM_HARDWARE_TASKGRAPH_RESIDENT_SUITE_SCHEMA_VERSION = (
    "upmem_hardware_taskgraph_resident_v1"
)
RESIDENT_BACKEND_ID = "upmem_sdk_hardware_taskgraph_resident"
RESIDENT_ROUTE_ID = "upmem_tn_hardware_taskgraph_resident"
RESIDENT_PROFILE_VERSION = "hardware_taskgraph_single_dpu_mram_resident_v1"
RESIDENT_EXECUTION_PLAN_BACKEND_ID = "upmem_sdk_hardware_execution_plan_resident"
RESIDENT_EXECUTION_PLAN_ROUTE_ID = "upmem_tn_hardware_execution_plan_resident"
RESIDENT_EXECUTION_PLAN_PROFILE_VERSION = "hardware_taskgraph_execution_plan_resident_v1"
RESIDENT_M46_PROFILE_VERSION = "hardware_taskgraph_single_dpu_mram_resident_m4_6_v1"
RESIDENT_V3_PROFILE_VERSION = "hardware_taskgraph_distributed_single_contraction_m5_v3"
RESIDENT_M5_V3_PROFILE_VERSION = RESIDENT_V3_PROFILE_VERSION
RESIDENT_SESSION_PROTOCOL = "generic_loop_resident_graph_session_v1"
RESIDENT_ALLOCATION_PROFILE = "backend=hw"
RESIDENT_TIMING_SCOPE = "one_dpu_mram_resident_full_taskgraph_v1"
RESIDENT_NUMERIC_MODES = ("none", "per_task_resident_requantize")
RESIDENT_V3_NUMERIC_MODES = (*RESIDENT_NUMERIC_MODES, "host_packed_int8")
RESIDENT_COMPLEX_POLICY = "split_real_imag_float32_dpu_complex_combine"
RESIDENT_MAX_RANK = 16
RESIDENT_MAX_ELEMENTS = 256
RESIDENT_MAX_LOGICAL_TASKS = 32
RESIDENT_MAX_COMPONENT_OPS = 128
RESIDENT_MAX_SLOT_DESCRIPTORS = 128
RESIDENT_MRAM_POOL_BYTES = 512 * 1024
RESIDENT_MAX_CONTRACTED_COMBINATIONS = 256
RESIDENT_OUTPUT_TILE_ELEMENTS = 256
RESIDENT_M46_OUTPUT_TILE_ELEMENTS = 2
RESIDENT_SUPPORTED_TASKLETS = (1, 2, 4, 8, 16)
RESIDENT_V3_SUPPORTED_TASKLETS = tuple(range(1, 25))
RESIDENT_V3_MAX_ELEMENTS = 65536
RESIDENT_V3_MAX_LOGICAL_TASKS = 1
RESIDENT_V3_MAX_COMPONENT_OPS = 1
RESIDENT_V3_OUTPUT_TILE_ELEMENTS = 2
RESIDENT_V3_MAX_DPUS = 64
RESIDENT_TIMEOUT_S = 30.0

RESIDENT_OPERATION_ABI_V1 = 1
RESIDENT_OPERATION_ABI_V2 = 2
RESIDENT_V3_OPERATION_ABI_VERSION = RESIDENT_OPERATION_ABI_V2
RESIDENT_PACKAGE_MAGIC_V1 = b"UPRGPCK1"
RESIDENT_PACKAGE_MAGIC_V2 = b"UPRGPCK2"
RESIDENT_PACKAGE_MAGIC_V3 = b"UPRGPCK3"
RESIDENT_PACKAGE_VERSION_V1 = 1
RESIDENT_PACKAGE_VERSION_V2 = 2
RESIDENT_PACKAGE_VERSION_V3 = 3
RESIDENT_PACKAGE_ENDIAN = 0x01020304
RESIDENT_PACKAGE_HEADER_FORMAT = "<8s4I5Q8I"
RESIDENT_PACKAGE_HEADER_BYTES = struct.calcsize(RESIDENT_PACKAGE_HEADER_FORMAT)
RESIDENT_SLOT_FORMAT = "<4I"
RESIDENT_SLOT_BYTES = struct.calcsize(RESIDENT_SLOT_FORMAT)
RESIDENT_V3_SLOT_FORMAT = "<8I"
RESIDENT_V3_SLOT_BYTES = struct.calcsize(RESIDENT_V3_SLOT_FORMAT)
RESIDENT_V3_FLAG_FLOAT32 = 0
RESIDENT_V3_FLAG_HOST_PACKED_INT8 = 1
RESIDENT_V3_STORAGE_F32 = 1
RESIDENT_V3_STORAGE_PACKED_I8 = 2
RESIDENT_V3_STORAGE_I32 = 3
RESIDENT_V3_OPERATION_ABI_VERSION = RESIDENT_OPERATION_ABI_V2
RESIDENT_INVALID_SLOT = 0xFFFFFFFF
RESIDENT_SLOT_ID_MASK = 0x3FFFFFFF
RESIDENT_SLOT_INITIAL_FLAG = 0x40000000
RESIDENT_SLOT_FINAL_FLAG = 0x80000000
RESIDENT_DESCRIPTOR_CONTROL_BYTES = 16
RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH = 8
RESIDENT_OPERATION_CONTRACT = 1
RESIDENT_OPERATION_COMPLEX_COMBINE = 2


def _resident_args_format(max_rank: int = RESIDENT_MAX_RANK) -> str:
    return "<" + "I" * (9 + 7 * max_rank) + "i" * (4 * max_rank) + "I" * 4


def _resident_operation_format(
    max_rank: int = RESIDENT_MAX_RANK,
    operation_abi_version: int = RESIDENT_OPERATION_ABI_V1,
) -> str:
    # kind, mode, output elements, six slot references, two float scales,
    # followed by the unchanged generic-loop index metadata ABI.
    if operation_abi_version not in {RESIDENT_OPERATION_ABI_V1, RESIDENT_OPERATION_ABI_V2}:
        raise ValueError("hardware_profile_violation: unsupported resident operation ABI")
    fmt = "<" + "I" * 9 + "ff" + "I" * (9 + 7 * max_rank) + "i" * (4 * max_rank)
    if operation_abi_version == RESIDENT_OPERATION_ABI_V2:
        fmt += "I" * 4
    return fmt


RESIDENT_OPERATION_BYTES_V1 = struct.calcsize(
    _resident_operation_format(operation_abi_version=RESIDENT_OPERATION_ABI_V1)
)
RESIDENT_OPERATION_BYTES_V2 = struct.calcsize(
    _resident_operation_format(operation_abi_version=RESIDENT_OPERATION_ABI_V2)
)
# Frozen M4.x callers intentionally retain the v1 defaults.
RESIDENT_PACKAGE_MAGIC = RESIDENT_PACKAGE_MAGIC_V1
RESIDENT_PACKAGE_VERSION = RESIDENT_PACKAGE_VERSION_V1
RESIDENT_OPERATION_BYTES = RESIDENT_OPERATION_BYTES_V1


def _resident_abi_metadata(operation_abi_version: int) -> tuple[bytes, int, int, str]:
    if operation_abi_version == RESIDENT_OPERATION_ABI_V1:
        return (
            RESIDENT_PACKAGE_MAGIC_V1,
            RESIDENT_PACKAGE_VERSION_V1,
            RESIDENT_OPERATION_BYTES_V1,
            "dpu_resident",
        )
    if operation_abi_version == RESIDENT_OPERATION_ABI_V2:
        return (
            RESIDENT_PACKAGE_MAGIC_V2,
            RESIDENT_PACKAGE_VERSION_V2,
            RESIDENT_OPERATION_BYTES_V2,
            "dpu_resident_v2",
        )
    raise ValueError("hardware_profile_violation: unsupported resident operation ABI")


def _resident_v3_package_metadata(package_flags: int) -> tuple[bytes, int, int, str]:
    if package_flags not in {RESIDENT_V3_FLAG_FLOAT32, RESIDENT_V3_FLAG_HOST_PACKED_INT8}:
        raise ValueError("hardware_profile_violation: unsupported resident v3 package flags")
    return (
        RESIDENT_PACKAGE_MAGIC_V3,
        RESIDENT_PACKAGE_VERSION_V3,
        RESIDENT_OPERATION_BYTES_V2,
        "dpu_resident_v3",
    )


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
        flags = _slot_flags(self)
        return {
            "slot_id": self.slot_id,
            "offset_bytes": self.offset_bytes,
            "capacity_elements": self.capacity_elements,
            "capacity_bytes": self.capacity_bytes,
            "element_count": self.element_count,
            "initial": bool(flags & RESIDENT_SLOT_INITIAL_FLAG),
            "final": bool(flags & RESIDENT_SLOT_FINAL_FLAG),
            "logical_ids": list(self.logical_ids),
            "lifetimes": [item.to_json_dict() for item in self.lifetimes],
        }


@dataclass(frozen=True)
class ResidentV3SlotDescriptor:
    """Typed UPRGPCK3 slot record.

    ``logical_bytes`` excludes MRAM/file padding.  ``transfer_bytes`` is the
    host-visible 8-byte-aligned transfer size.
    """

    slot_id: int
    offset_bytes: int
    capacity_elements: int
    element_count: int
    element_bytes: int
    storage_kind: int
    logical_bytes: int
    transfer_bytes: int
    initial: bool
    final: bool

    @property
    def capacity_bytes(self) -> int:
        return _align8(self.capacity_elements * self.element_bytes)

    @property
    def storage_dtype(self) -> str:
        return {
            RESIDENT_V3_STORAGE_F32: "float32",
            RESIDENT_V3_STORAGE_PACKED_I8: "int8",
            RESIDENT_V3_STORAGE_I32: "int32",
        }[self.storage_kind]

    def to_json_dict(self) -> JsonDict:
        return {
            "slot_id": self.slot_id,
            "offset_bytes": self.offset_bytes,
            "capacity_elements": self.capacity_elements,
            "element_count": self.element_count,
            "element_bytes": self.element_bytes,
            "storage_kind": self.storage_kind,
            "storage_dtype": self.storage_dtype,
            "logical_bytes": self.logical_bytes,
            "transfer_bytes": self.transfer_bytes,
            "initial": self.initial,
            "final": self.final,
        }


# Explicit aliases make the additive record easy to discover without
# changing the frozen ResidentSlotDescriptor used by V1/V2 callers.
RESIDENT_SLOT_V3 = ResidentV3SlotDescriptor
ResidentSlotDescriptorV3 = ResidentV3SlotDescriptor


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
        if self.mode == "host_packed_int8":
            return 2
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

    def to_bytes(
        self,
        *,
        max_rank: int = RESIDENT_MAX_RANK,
        operation_abi_version: int = RESIDENT_OPERATION_ABI_V1,
    ) -> bytes:
        if not math.isfinite(float(self.left_scale)) or not math.isfinite(float(self.right_scale)):
            raise ValueError("hardware_profile_violation: resident operation scale must be finite")
        values = _pack_native_args(
            self.args,
            mode=(
                "host_packed_int8"
                if self.mode == "host_packed_int8"
                else GENERIC_MODE_FLOAT32_NO_QUANT
            ),
            output_elements=self.output_elements,
            operation_abi_version=operation_abi_version,
        )
        return struct.pack(
            _resident_operation_format(max_rank, operation_abi_version),
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
    profile: HardwareTaskGraphResidentProfile | None = None
    operation_abi_version: int = RESIDENT_OPERATION_ABI_V1
    package_magic: bytes | None = None
    package_version: int | None = None
    package_flags: int = RESIDENT_V3_FLAG_FLOAT32
    typed_slots: tuple[ResidentV3SlotDescriptor, ...] = ()
    storage_initial_data: Mapping[int, np.ndarray] | None = None
    input_scales: Mapping[int, float] = field(default_factory=dict)
    raw_final_output_paths: Mapping[str, Path] = field(default_factory=dict)
    dequant_final_output_paths: Mapping[str, Path] = field(default_factory=dict)

    @property
    def graph_request_count(self) -> int:
        return 1

    @property
    def descriptor_count(self) -> int:
        return len(self.operations)

    @property
    def component_operation_count(self) -> int:
        return len(self.operations)

    @property
    def is_v3(self) -> bool:
        return self.package_version == RESIDENT_PACKAGE_VERSION_V3

    def to_json_dict(self) -> JsonDict:
        selected_profile = self.profile or _canonical_profile()
        if self.is_v3:
            package_magic, package_version, operation_bytes, dpu_binary_abi = (
                _resident_v3_package_metadata(self.package_flags)
            )
        else:
            package_magic, package_version, operation_bytes, dpu_binary_abi = (
                _resident_abi_metadata(self.operation_abi_version)
            )
        return {
            "schema_version": RESIDENT_SESSION_PROTOCOL,
            "manifest_kind": "resident_graph_package",
            "case_id": self.case_id,
            "suite_id": self.suite_id,
            "route_id": selected_profile.route_id,
            "backend_id": selected_profile.backend_id,
            "quantization_mode": self.quantization_mode,
            "package_magic": package_magic.decode("ascii"),
            "package_version": package_version,
            "operation_abi_version": self.operation_abi_version,
            "operation_bytes": operation_bytes,
            "dpu_binary_abi": dpu_binary_abi,
            "hardware_profile_version": selected_profile.version,
            "tasklets_per_dpu": selected_profile.tasklets_per_dpu,
            "requested_dpu_count": selected_profile.requested_dpu_count,
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
            "raw_output_paths": {
                key: str(value) for key, value in self.raw_final_output_paths.items()
            },
            "package_flags": self.package_flags,
            "typed_slots": [item.to_json_dict() for item in self.typed_slots],
            "input_scales": {str(key): float(value) for key, value in self.input_scales.items()},
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
        if not request_id or not request_id.isascii():
            raise ValueError(
                "hardware_profile_violation: resident request identifiers must be non-empty ASCII"
            )
        root.mkdir(parents=True, exist_ok=True)
        safe = _safe_name(request_id)
        request_dir = root / "resident_requests" / safe
        request_dir.mkdir(parents=True, exist_ok=False)
        input_dir = request_dir / "inputs"
        output_dir = request_dir / "outputs"
        input_dir.mkdir()
        output_dir.mkdir()

        if self.is_v3:
            return _write_v3_package_request(
                self,
                root=root,
                request_dir=request_dir,
                input_dir=input_dir,
                output_dir=output_dir,
                dpu_binary=dpu_binary,
                request_id=request_id,
            )

        input_entries: list[JsonDict] = []
        initial_h2d_bytes = 0
        for slot_id in sorted(self.initial_data):
            array = np.asarray(self.initial_data[slot_id], dtype="<f4").ravel()
            path = input_dir / f"slot_{int(slot_id):04d}.bin"
            _require_ascii(_relative(root, path), "resident input path")
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
            _require_ascii(_relative(root, path), "resident output path")
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

        # Select the profile before encoding so the package header and the
        # Python validator use the same additive v3 limits.
        selected_profile = self.profile or _canonical_profile()
        package_bytes = _encode_package(
            self.allocation.slots,
            self.operations,
            profile=selected_profile,
            operation_abi_version=self.operation_abi_version,
        )
        # Apply the Python-side ABI validator before emitting any request
        # manifest. This keeps preparation fail-closed with the native parser.
        validate_resident_graph_package_bytes(
            package_bytes,
            profile=selected_profile,
            operation_abi_version=self.operation_abi_version,
        )
        package_path = request_dir / "resident_graph_package.bin"
        package_path.write_bytes(package_bytes)
        descriptor_sha256 = hashlib.sha256(package_bytes).hexdigest()
        descriptor_h2d_bytes = _descriptor_transfer_bytes(
            len(self.allocation.slots),
            len(self.operations),
            self.operation_abi_version,
        )
        final_d2h_bytes = sum(int(item["transfer_bytes"]) for item in final_entries)
        dpu_ref = _relative(root, dpu_binary)
        _require_ascii(dpu_ref, "resident DPU binary path")
        package_magic, package_version, operation_bytes, dpu_binary_abi = _resident_abi_metadata(
            self.operation_abi_version
        )
        manifest_path = root / f"{safe}_resident_request.json"
        payload: JsonDict = {
            "schema_version": RESIDENT_SESSION_PROTOCOL,
            "manifest_kind": "resident_graph_request",
            "session_id": request_id,
            "route_id": selected_profile.route_id,
            "backend_id": selected_profile.backend_id,
            "hardware_profile_version": selected_profile.version,
            "target": "hardware",
            "sdk_allocation_profile": RESIDENT_ALLOCATION_PROFILE,
            "session_protocol": selected_profile.session_protocol,
            "package_magic": package_magic.decode("ascii"),
            "package_version": package_version,
            "operation_abi_version": self.operation_abi_version,
            "operation_bytes": operation_bytes,
            "dpu_binary_abi": dpu_binary_abi,
            "dpu_binary": dpu_ref,
            "package_path": _relative(root, package_path),
            "requested_dpus": selected_profile.requested_dpu_count,
            "requested_dpu_count": selected_profile.requested_dpu_count,
            "tasklets": selected_profile.tasklets_per_dpu,
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
            "descriptor_control_bytes": RESIDENT_DESCRIPTOR_CONTROL_BYTES,
            "control_h2d_bytes_per_launch": RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
            "control_h2d_bytes": RESIDENT_DESCRIPTOR_CONTROL_BYTES + len(self.operations) * RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
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
            profile=selected_profile,
            operation_abi_version=self.operation_abi_version,
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
        "completion_abi_version": 2 if profile.version == RESIDENT_M46_PROFILE_VERSION else 1,
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
    allow_slot_reuse: bool = True,
) -> ResidentAllocation:
    """Build a deterministic interval allocator for float32 resident slots."""

    selected = profile or _canonical_profile()
    tasks = tuple(graph.tasks)
    if not tasks:
        raise ResidentCapacityError("hardware_profile_violation: empty_task_graph_not_supported")
    if len(tasks) > selected.max_logical_tasks:
        raise ResidentCapacityError("hardware_profile_violation: logical_task_cap_exceeded")

    arrays = {tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors}
    known_components: dict[str, tuple[str, ...]] = {
        tensor_id: _components_for(array) for tensor_id, array in arrays.items()
    }
    tensor_complexity: dict[str, bool] = {
        tensor_id: _has_nonzero_imaginary(array) for tensor_id, array in arrays.items()
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
        input_complex = any(tensor_complexity[item] for item in task.input_tensor_ids)
        if input_complex:
            for tensor_id in task.input_tensor_ids:
                if "imag" not in known_components[tensor_id]:
                    known_components[tensor_id] = ("real", "imag")
        known_components[task.output_tensor_id] = ("real", "imag") if input_complex else ("real",)
        tensor_complexity[task.output_tensor_id] = input_complex
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
        if allow_slot_reuse:
            for slot_id, existing in enumerate(slot_lifetimes):
                if slot_capacity[slot_id] < lifetime.elements:
                    continue
                # The native file-backed ABI gives initial and final slots
                # disjoint roles. Keep final outputs out of every slot that has
                # ever carried an initial input, even when live intervals no
                # longer overlap. Ordinary intermediate reuse remains allowed.
                if lifetime.final and any(item.initial for item in existing):
                    continue
                if lifetime.initial and any(item.final for item in existing):
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
    allow_slot_reuse: bool = True,
    operation_abi_version: int = RESIDENT_OPERATION_ABI_V1,
) -> ResidentGraphPackage:
    selected = profile or _canonical_profile()
    _require_canonical_profile(selected)
    _resident_abi_metadata(operation_abi_version)
    if selected.version == RESIDENT_V3_PROFILE_VERSION and operation_abi_version != RESIDENT_OPERATION_ABI_V2:
        raise ResidentCapacityError("hardware_profile_violation: resident v3 requires operation ABI v2")
    if quantization_mode not in selected.numeric_modes:
        raise ResidentCapacityError("hardware_profile_violation: unsupported_numeric_mode")
    _validate_finite_inputs(network)
    allocation = allocate_resident_slots(
        graph,
        network,
        profile=selected,
        allow_slot_reuse=allow_slot_reuse,
    )
    operations: list[ResidentOperationDescriptor] = []
    component_index = 0
    tensor_complexity = {
        tensor.spec.id: _has_nonzero_imaginary(tensor.array) for tensor in network.tensors
    }
    if quantization_mode == "host_packed_int8" and len(graph.tasks) != 1:
        raise ValueError("hardware_profile_violation: host_packed_int8 requires one contraction")
    for task_index, task in enumerate(graph.tasks):
        complex_task = any(tensor_complexity[item] for item in task.input_tensor_ids)
        if quantization_mode == "host_packed_int8" and complex_task:
            raise ValueError("hardware_profile_violation: host_packed_int8 requires real inputs")
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
            tensor_complexity[task.output_tensor_id] = False
            continue
        left_real = allocation.slot_for(task.input_tensor_ids[0], "real")
        right_real = allocation.slot_for(task.input_tensor_ids[1], "real")
        left_imag = allocation.slot_for(task.input_tensor_ids[0], "imag")
        right_imag = allocation.slot_for(task.input_tensor_ids[1], "imag")
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
        tensor_complexity[task.output_tensor_id] = True
    if len(operations) != allocation.component_operation_count:
        raise ResidentCapacityError("hardware_profile_violation: component_operation_count_mismatch")
    if selected.version == RESIDENT_V3_PROFILE_VERSION:
        if len(graph.tasks) != 1 or len(operations) != 1 or operations[0].kind != "contract":
            raise ResidentCapacityError(
                "hardware_profile_violation: resident v3 requires one real contraction"
            )
        input_scales: dict[int, float] = {}
        if quantization_mode == "host_packed_int8":
            storage_initial_data: dict[int, np.ndarray] = {}
            for slot_id, values in allocation.initial_data.items():
                quantized, scale, _saturation = resident_requantize(values)
                if not math.isfinite(scale) or scale <= 0.0:
                    raise ResidentCapacityError(
                        "hardware_profile_violation: packed int8 scale must be finite and positive"
                    )
                storage_initial_data[int(slot_id)] = np.ascontiguousarray(quantized, dtype=np.int8)
                input_scales[int(slot_id)] = float(np.float32(scale))
            operation = replace(
                operations[0],
                left_scale=input_scales[int(operations[0].slot_a)],
                right_scale=input_scales[int(operations[0].slot_b)],
            )
            package_flags = RESIDENT_V3_FLAG_HOST_PACKED_INT8
        else:
            operation = operations[0]
            storage_initial_data = {
                int(slot_id): np.ascontiguousarray(values, dtype=np.float32)
                for slot_id, values in allocation.initial_data.items()
            }
            package_flags = RESIDENT_V3_FLAG_FLOAT32
        typed_slots = _build_v3_slot_descriptors(allocation, package_flags)
        if len(typed_slots) != 3 or len(storage_initial_data) != 2:
            raise ResidentCapacityError(
                "hardware_profile_violation: resident v3 requires two initial slots and one final slot"
            )
        return ResidentGraphPackage(
            graph=graph,
            case_id=case_id,
            suite_id=suite_id,
            quantization_mode=quantization_mode,
            allocation=allocation,
            operations=(operation,),
            initial_data=allocation.initial_data,
            storage_initial_data=storage_initial_data,
            input_scales=input_scales,
            full_precision_output=full_precision_output,
            profile=selected,
            operation_abi_version=RESIDENT_OPERATION_ABI_V2,
            package_magic=RESIDENT_PACKAGE_MAGIC_V3,
            package_version=RESIDENT_PACKAGE_VERSION_V3,
            package_flags=package_flags,
            typed_slots=typed_slots,
        )
    return ResidentGraphPackage(
        graph=graph,
        case_id=case_id,
        suite_id=suite_id,
        quantization_mode=quantization_mode,
        allocation=allocation,
        operations=tuple(operations),
        initial_data=allocation.initial_data,
        full_precision_output=full_precision_output,
        profile=selected,
        operation_abi_version=operation_abi_version,
    )


def resident_round_nearest_even(values: Any) -> np.ndarray:
    """Return explicit ties-to-even values before int8 clipping."""

    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("hardware_profile_violation: resident rounding input must be finite")
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
    if not np.all(np.isfinite(array)):
        raise ValueError("hardware_profile_violation: resident quantization input must be finite")
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    scale32 = 1.0 if max_abs == 0.0 else float(np.float32(max_abs) / np.float32(127.0))
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
    _require_canonical_profile(selected)
    if quantization_mode not in selected.numeric_modes:
        raise ValueError("hardware_profile_violation: unsupported_numeric_mode")
    _validate_finite_inputs(network)
    caps = GenericTaskPreparationCaps(
        max_rank=selected.max_rank,
        max_tensor_elements=selected.max_tensor_elements,
        max_contracted_combinations=selected.max_contracted_combinations,
    )
    tensors = {tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    tensor_complexity = {
        tensor.spec.id: _has_nonzero_imaginary(tensor.array) for tensor in network.tensors
    }
    task_metrics: list[JsonDict] = []
    for task_index, task in enumerate(graph.tasks):
        left = TensorValue(_spec_for(task.input_tensor_ids[0], labels, tensors, task.input_shapes[0]), tensors[task.input_tensor_ids[0]])
        right = TensorValue(_spec_for(task.input_tensor_ids[1], labels, tensors, task.input_shapes[1]), tensors[task.input_tensor_ids[1]])
        complex_task = any(tensor_complexity[item] for item in task.input_tensor_ids)
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
        tensor_complexity[task.output_tensor_id] = complex_task
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
    result: JsonDict = {
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
    if quantization_mode == "host_packed_int8":
        metric = task_metrics[0]["component_metrics"]["real"]
        raw_output = np.asarray(metric["raw_output"], dtype=np.int32)
        raw_ordered = order_final_tensor(
            raw_output,
            graph.tasks[-1].output_labels,
            graph.network.output_labels,
        )[0]
        if np.iscomplexobj(raw_ordered):
            if np.any(np.asarray(raw_ordered).imag != 0):
                raise ValueError(
                    "hardware_profile_violation: packed int32 reference became complex"
                )
            raw_ordered = np.asarray(raw_ordered).real
        result.update(
            {
                "reference_kind": "cpu_host_packed_int8_reference",
                "resident_slot_dtype": "int8_inputs_int32_output",
                "dpu_local_requantization": False,
                "raw_output": np.asarray(raw_ordered, dtype=np.int32),
                "raw_output_hash": _array_hash(raw_ordered),
                "input_scales": metric["input_scales"],
                "scale_metadata": metric["scale_metadata"],
            }
        )
    return result


def validate_resident_graph_package_bytes(
    payload: bytes,
    *,
    profile: HardwareTaskGraphResidentProfile | None = None,
    operation_abi_version: int | None = None,
) -> JsonDict:
    """Validate binary package framing and every slot interval before upload."""

    selected = profile or _canonical_profile()
    _require_canonical_profile(selected)
    if selected.version == RESIDENT_V3_PROFILE_VERSION and operation_abi_version not in {None, RESIDENT_OPERATION_ABI_V2}:
        raise ValueError("hardware_profile_violation: resident v3 requires operation ABI v2")
    if len(payload) < RESIDENT_PACKAGE_HEADER_BYTES:
        raise ValueError("hardware_profile_violation: resident_package_truncated_header")
    header = struct.unpack_from(RESIDENT_PACKAGE_HEADER_FORMAT, payload, 0)
    (
        magic, version, endian, header_bytes, flags, file_bytes,
        slot_offset, slot_bytes, operation_offset, operation_bytes,
        slot_count, operation_count, pool_bytes, request_count,
        initial_count, final_count, max_rank, reserved,
    ) = header
    package_abi_by_header = {
        RESIDENT_PACKAGE_MAGIC_V1: RESIDENT_OPERATION_ABI_V1,
        RESIDENT_PACKAGE_MAGIC_V2: RESIDENT_OPERATION_ABI_V2,
        RESIDENT_PACKAGE_MAGIC_V3: RESIDENT_OPERATION_ABI_V2,
    }
    header_operation_abi = package_abi_by_header.get(magic)
    if header_operation_abi is None:
        raise ValueError("hardware_profile_violation: resident_package_bad_magic")
    if operation_abi_version is None:
        operation_abi_version = header_operation_abi
    if header_operation_abi != operation_abi_version:
        raise ValueError("hardware_profile_violation: resident_package_abi_mismatch")
    is_v3 = magic == RESIDENT_PACKAGE_MAGIC_V3
    if is_v3:
        if selected.version != RESIDENT_V3_PROFILE_VERSION:
            raise ValueError("hardware_profile_violation: resident_package_profile_mismatch")
        _, expected_package_version, expected_operation_bytes, _ = (
            _resident_v3_package_metadata(flags)
        )
        expected_slot_bytes = RESIDENT_V3_SLOT_BYTES
    else:
        if flags != 0:
            raise ValueError("hardware_profile_violation: resident_package_header_flags_invalid")
        _, expected_package_version, expected_operation_bytes, _ = _resident_abi_metadata(
            operation_abi_version
        )
        expected_slot_bytes = RESIDENT_SLOT_BYTES
    if version != expected_package_version:
        raise ValueError("hardware_profile_violation: resident_package_bad_version")
    if endian != RESIDENT_PACKAGE_ENDIAN:
        raise ValueError("hardware_profile_violation: resident_package_bad_endian")
    if reserved != 0:
        raise ValueError("hardware_profile_violation: resident_package_header_flags_invalid")
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
    if slot_bytes != slot_count * expected_slot_bytes:
        raise ValueError("hardware_profile_violation: resident_package_slot_length_mismatch")
    if operation_bytes != operation_count * expected_operation_bytes:
        raise ValueError("hardware_profile_violation: resident_package_operation_length_mismatch")
    if slot_count == 0 or operation_count == 0:
        raise ValueError("hardware_profile_violation: resident_package_descriptor_count_invalid")
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
    if initial_count == 0 or initial_count > slot_count or final_count == 0 or final_count > 2:
        raise ValueError("hardware_profile_violation: resident_package_output_count_invalid")
    slots: list[JsonDict] = []
    observed_initial = 0
    observed_final = 0
    for index in range(slot_count):
        record_offset = slot_offset + index * expected_slot_bytes
        if is_v3:
            (
                encoded_id,
                offset,
                capacity,
                elements,
                element_bytes,
                storage_kind,
                logical_bytes,
                transfer_bytes,
            ) = struct.unpack_from(RESIDENT_V3_SLOT_FORMAT, payload, record_offset)
        else:
            encoded_id, offset, capacity, elements = struct.unpack_from(
                RESIDENT_SLOT_FORMAT, payload, record_offset
            )
            element_bytes = 4
            storage_kind = RESIDENT_V3_STORAGE_F32
            logical_bytes = elements * element_bytes
            transfer_bytes = _align8(capacity * element_bytes)
        slot_flags = encoded_id & ~RESIDENT_SLOT_ID_MASK
        slot_id = encoded_id & RESIDENT_SLOT_ID_MASK
        if slot_id != index or slot_flags & ~(
            RESIDENT_SLOT_INITIAL_FLAG | RESIDENT_SLOT_FINAL_FLAG
        ):
            raise ValueError("hardware_profile_violation: resident_package_slot_ids_not_dense")
        initial = bool(slot_flags & RESIDENT_SLOT_INITIAL_FLAG)
        final = bool(slot_flags & RESIDENT_SLOT_FINAL_FLAG)
        observed_initial += initial
        observed_final += final
        if offset % 8 or capacity == 0 or elements == 0 or elements > capacity:
            raise ValueError("hardware_profile_violation: resident_package_slot_descriptor_invalid")
        if is_v3:
            expected_width = {
                RESIDENT_V3_STORAGE_F32: 4,
                RESIDENT_V3_STORAGE_PACKED_I8: 1,
                RESIDENT_V3_STORAGE_I32: 4,
            }.get(storage_kind)
            if expected_width is None or element_bytes != expected_width:
                raise ValueError(
                    "hardware_profile_violation: resident_package_slot_storage_invalid"
                )
            if logical_bytes != elements * element_bytes:
                raise ValueError(
                    "hardware_profile_violation: resident_package_slot_logical_bytes_invalid"
                )
            if transfer_bytes != _align8(capacity * element_bytes):
                raise ValueError(
                    "hardware_profile_violation: resident_package_slot_transfer_bytes_invalid"
                )
            if transfer_bytes % 8:
                raise ValueError(
                    "hardware_profile_violation: resident_package_slot_transfer_unaligned"
                )
            if flags == RESIDENT_V3_FLAG_HOST_PACKED_INT8:
                expected_storage = (
                    RESIDENT_V3_STORAGE_PACKED_I8
                    if initial
                    else RESIDENT_V3_STORAGE_I32 if final else None
                )
                if storage_kind != expected_storage:
                    raise ValueError(
                        "hardware_profile_violation: resident_package_packed_slot_storage_invalid"
                    )
            elif storage_kind != RESIDENT_V3_STORAGE_F32:
                raise ValueError(
                    "hardware_profile_violation: resident_package_float_slot_storage_invalid"
                )
        if offset + transfer_bytes > pool_bytes:
            raise ValueError("hardware_profile_violation: resident_package_slot_overflow")
        slots.append(
            {
                "slot_id": slot_id,
                "offset_bytes": offset,
                "capacity_elements": capacity,
                "element_count": elements,
                "element_bytes": element_bytes,
                "storage_kind": storage_kind,
                "logical_bytes": logical_bytes,
                "transfer_bytes": transfer_bytes,
                "initial": initial,
                "final": final,
            }
        )
    if observed_initial != initial_count or observed_final != final_count:
        raise ValueError("hardware_profile_violation: resident_package_slot_flag_count_mismatch")
    initial_slot_ids = {int(slot["slot_id"]) for slot in slots if slot["initial"]}
    final_slot_ids = {int(slot["slot_id"]) for slot in slots if slot["final"]}
    if initial_slot_ids & final_slot_ids:
        raise ValueError(
            "hardware_profile_violation: resident_package_initial_final_slot_alias"
        )
    sorted_slots = sorted(slots, key=lambda item: int(item["offset_bytes"]))
    for left, right in zip(sorted_slots, sorted_slots[1:]):
        if int(left["offset_bytes"]) + int(left["transfer_bytes"]) > int(
            right["offset_bytes"]
        ):
            raise ValueError("hardware_profile_violation: resident_package_slot_overlap")
    produced_slots: set[int] = set()
    referenced_initial_slots: set[int] = set()
    operation_modes: list[int] = []
    for index in range(operation_count):
        operation = struct.unpack_from(
            _resident_operation_format(selected.max_rank, operation_abi_version),
            payload,
            operation_offset + index * expected_operation_bytes,
        )
        kind, mode, output_elements = operation[:3]
        refs = operation[3:9]
        left_scale, right_scale = operation[9:11]
        if kind not in {RESIDENT_OPERATION_CONTRACT, RESIDENT_OPERATION_COMPLEX_COMBINE}:
            raise ValueError("hardware_profile_violation: resident_package_operation_kind_invalid")
        allowed_modes = {0, 1, 2} if is_v3 else {0, 1}
        if mode not in allowed_modes:
            raise ValueError("hardware_profile_violation: resident_package_operation_mode_invalid")
        if output_elements == 0 or output_elements > selected.max_tensor_elements:
            raise ValueError("hardware_profile_violation: resident_package_operation_empty_output")
        if not math.isfinite(left_scale) or not math.isfinite(right_scale):
            raise ValueError("hardware_profile_violation: resident_package_operation_scale_invalid")
        args = operation[11:]
        if operation_abi_version == RESIDENT_OPERATION_ABI_V2:
            if args[-4:] != (0, args[6], 0, args[7]):
                raise ValueError(
                    "hardware_profile_violation: resident_package_operation_v2_range_invalid"
                )
        operation_modes.append(mode)
        for slot_id in refs:
            if slot_id != RESIDENT_INVALID_SLOT and slot_id >= slot_count:
                raise ValueError("hardware_profile_violation: resident_package_operation_slot_reference_invalid")
        if kind == RESIDENT_OPERATION_CONTRACT:
            if refs[0] == RESIDENT_INVALID_SLOT or refs[1] == RESIDENT_INVALID_SLOT or refs[4] == RESIDENT_INVALID_SLOT:
                raise ValueError("hardware_profile_violation: resident_package_contract_slot_reference_invalid")
            if refs[2] != RESIDENT_INVALID_SLOT or refs[3] != RESIDENT_INVALID_SLOT or refs[5] != RESIDENT_INVALID_SLOT:
                raise ValueError("hardware_profile_violation: resident_package_contract_unused_slot_reference_invalid")
            read_slots = refs[:2]
            write_slots = (refs[4],)
        elif any(slot_id == RESIDENT_INVALID_SLOT for slot_id in refs):
            raise ValueError("hardware_profile_violation: resident_package_complex_slot_reference_invalid")
        else:
            read_slots = refs[:4]
            write_slots = refs[4:6]
        referenced_initial_slots.update(read_slots)
        if any(slot_id not in initial_slot_ids | produced_slots for slot_id in read_slots):
            raise ValueError("hardware_profile_violation: resident_package_slot_read_before_initialization")
        produced_slots.update(write_slots)
        if is_v3 and flags == RESIDENT_V3_FLAG_HOST_PACKED_INT8:
            if kind != RESIDENT_OPERATION_CONTRACT or mode != 2:
                raise ValueError(
                    "hardware_profile_violation: resident_package_packed_operation_invalid"
                )
            if left_scale <= 0.0 or right_scale <= 0.0:
                raise ValueError(
                    "hardware_profile_violation: resident_package_packed_scale_invalid"
                )
            contracted_elements = int(args[7])
            if contracted_elements > RESIDENT_V3_MAX_ELEMENTS:
                raise ValueError(
                    "hardware_profile_violation: resident_package_int32_bound_exceeded"
                )
            if contracted_elements * 127 * 127 > np.iinfo(np.int32).max:
                raise ValueError(
                    "hardware_profile_violation: resident_package_int32_bound_exceeded"
                )
            slot_by_id = {int(slot["slot_id"]): slot for slot in slots}
            if any(
                int(slot_by_id[int(slot_id)]["storage_kind"])
                != RESIDENT_V3_STORAGE_PACKED_I8
                for slot_id in read_slots
            ) or any(
                int(slot_by_id[int(slot_id)]["storage_kind"])
                != RESIDENT_V3_STORAGE_I32
                for slot_id in write_slots
            ):
                raise ValueError(
                    "hardware_profile_violation: resident_package_packed_operation_storage_mismatch"
                )
    if initial_slot_ids - referenced_initial_slots:
        raise ValueError(
            "hardware_profile_violation: resident_package_initial_slot_not_referenced"
        )
    if not set(final_slot_ids).issubset(produced_slots):
        raise ValueError("hardware_profile_violation: resident_package_final_slot_not_produced")
    if is_v3 and flags == RESIDENT_V3_FLAG_HOST_PACKED_INT8:
        if slot_count != 3 or initial_count != 2 or final_count != 1 or operation_count != 1:
            raise ValueError(
                "hardware_profile_violation: resident_package_packed_shape_invalid"
            )
    if is_v3 and flags == RESIDENT_V3_FLAG_FLOAT32 and 2 in operation_modes:
        raise ValueError(
            "hardware_profile_violation: resident_package_float_mode_storage_mismatch"
        )
    return {
        "magic": magic.decode("ascii"),
        "version": version,
        "package_magic": magic.decode("ascii"),
        "package_version": version,
        "package_flags": flags,
        "operation_abi_version": operation_abi_version,
        "operation_bytes": expected_operation_bytes,
        "endian": endian,
        "file_bytes": file_bytes,
        "slot_count": slot_count,
        "operation_count": operation_count,
        "pool_bytes": pool_bytes,
        "graph_request_count": request_count,
        "initial_slot_count": initial_count,
        "final_output_component_count": final_count,
        "max_rank": max_rank,
        "operation_modes": operation_modes,
        "initial_slot_ids": sorted(initial_slot_ids),
        "final_slot_ids": sorted(final_slot_ids),
        "slot_descriptors": slots,
    }


def validate_resident_graph_package_file(
    path: Path,
    *,
    profile: HardwareTaskGraphResidentProfile | None = None,
    operation_abi_version: int | None = None,
) -> JsonDict:
    return validate_resident_graph_package_bytes(
        path.read_bytes(),
        profile=profile,
        operation_abi_version=operation_abi_version,
    )


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
    if mode == "host_packed_int8":
        left_q, left_scale, left_sat = resident_requantize(left_value)
        right_q, right_scale, right_sat = resident_requantize(right_value)
        raw = np.asarray(
            generic_loop_reference_int32(
                left_q,
                right_q,
                output_shape=task.output_shape,
                **_index_args(task, caps),
            ),
            dtype=np.int32,
        )
        output_scale = np.float32(np.float32(left_scale) * np.float32(right_scale))
        if not np.isfinite(output_scale) or output_scale <= 0.0:
            raise ValueError(
                "hardware_profile_violation: packed int8 output scale must be finite and positive"
            )
        output = np.asarray(raw.astype(np.float32) * output_scale, dtype=np.float32)
        return output, {
            "mode": mode,
            "left_scale": float(np.float32(left_scale)),
            "right_scale": float(np.float32(right_scale)),
            "output_scale": float(output_scale),
            "left_saturation_count": left_sat,
            "right_saturation_count": right_sat,
            "raw_output": raw,
            "input_scales": {
                "left": float(np.float32(left_scale)),
                "right": float(np.float32(right_scale)),
            },
            "scale_metadata": {
                "left": _host_scale_metadata(left_value, left_scale, left_sat),
                "right": _host_scale_metadata(right_value, right_scale, right_sat),
            },
        }
    left_q, left_scale, left_sat = resident_requantize(left_value)
    right_q, right_scale, right_sat = resident_requantize(right_value)
    raw = generic_loop_reference_int32(
        left_q,
        right_q,
        output_shape=task.output_shape,
        **_index_args(task, caps),
    )
    scaled = np.asarray(raw, dtype=np.int32).astype(np.float32)
    scaled = np.asarray(scaled * np.float32(left_scale), dtype=np.float32)
    output = np.asarray(scaled * np.float32(right_scale), dtype=np.float32)
    return output, {
        "mode": mode,
        "left_scale": left_scale,
        "right_scale": right_scale,
        "left_saturation_count": left_sat,
        "right_saturation_count": right_sat,
    }


def _host_scale_metadata(values: np.ndarray, scale: float, saturation_count: int) -> JsonDict:
    array = np.asarray(values, dtype=np.float32)
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    return {
        "max_abs": max_abs,
        "scale": float(np.float32(scale)),
        "scale_formula": "max_abs/127_or_1_for_exact_zero",
        "rounding": "nearest_even",
        "clip_min": -127,
        "clip_max": 127,
        "saturation_count": int(saturation_count),
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


def _pack_native_args(
    args: Mapping[str, Any],
    *,
    mode: str,
    output_elements: int | None = None,
    max_rank: int = RESIDENT_MAX_RANK,
    operation_abi_version: int = RESIDENT_OPERATION_ABI_V1,
) -> tuple[int, ...]:
    _resident_abi_metadata(operation_abi_version)
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
        (
            2
            if mode == "host_packed_int8"
            else 1 if mode == GENERIC_MODE_FLOAT32_NO_QUANT else 0
        ),
    ]
    for name in ("left_shape", "right_shape", "output_shape", "contracted_dims", "left_strides", "right_strides", "output_strides"):
        fields.extend(unsigned(name))
    signed_fields: list[int] = []
    for name in ("output_to_left_axes", "output_to_right_axes", "contracted_to_left_axes", "contracted_to_right_axes"):
        signed_fields.extend(signed(name))
    if operation_abi_version == RESIDENT_OPERATION_ABI_V1:
        return tuple(fields + signed_fields)
    # v2 descriptors always carry the complete range. Distributed execution
    # replaces these values in a per-DPU descriptor copy after validation.
    slice_fields = [
        int(args.get("dpu_slice_offset", 0)),
        int(args.get("dpu_slice_elements", args.get("output_element_count", output_elements or 0))),
        int(args.get("contracted_offset", 0)),
        int(args.get("contracted_elements_slice", args.get("contracted_combination_count", 0))),
    ]
    return tuple(fields + signed_fields + slice_fields)


def _build_v3_slot_descriptors(
    allocation: ResidentAllocation,
    package_flags: int,
) -> tuple[ResidentV3SlotDescriptor, ...]:
    if package_flags not in {RESIDENT_V3_FLAG_FLOAT32, RESIDENT_V3_FLAG_HOST_PACKED_INT8}:
        raise ResidentCapacityError("hardware_profile_violation: unsupported resident v3 flags")
    descriptors: list[ResidentV3SlotDescriptor] = []
    offset = 0
    for slot in sorted(allocation.slots, key=lambda item: item.slot_id):
        flags = _slot_flags(slot)
        initial = bool(flags & RESIDENT_SLOT_INITIAL_FLAG)
        final = bool(flags & RESIDENT_SLOT_FINAL_FLAG)
        if initial and final:
            raise ResidentCapacityError(
                "hardware_profile_violation: resident v3 initial/final slot alias"
            )
        if package_flags == RESIDENT_V3_FLAG_HOST_PACKED_INT8:
            if initial:
                element_bytes, storage_kind = 1, RESIDENT_V3_STORAGE_PACKED_I8
            elif final:
                element_bytes, storage_kind = 4, RESIDENT_V3_STORAGE_I32
            else:
                raise ResidentCapacityError(
                    "hardware_profile_violation: packed v3 does not admit intermediate slots"
                )
        else:
            element_bytes, storage_kind = 4, RESIDENT_V3_STORAGE_F32
        offset = _align8(offset)
        logical_bytes = int(slot.element_count) * element_bytes
        transfer_bytes = _align8(int(slot.capacity_elements) * element_bytes)
        descriptors.append(
            ResidentV3SlotDescriptor(
                slot_id=int(slot.slot_id),
                offset_bytes=offset,
                capacity_elements=int(slot.capacity_elements),
                element_count=int(slot.element_count),
                element_bytes=element_bytes,
                storage_kind=storage_kind,
                logical_bytes=logical_bytes,
                transfer_bytes=transfer_bytes,
                initial=initial,
                final=final,
            )
        )
        offset += transfer_bytes
    if offset > allocation.mram_pool_bytes:
        raise ResidentCapacityError(
            f"hardware_profile_violation: resident_v3_mram_capacity_exceeded:{offset}>{allocation.mram_pool_bytes}"
        )
    return tuple(descriptors)


def _encoded_v3_slot_id(slot: ResidentV3SlotDescriptor) -> int:
    flags = 0
    if slot.initial:
        flags |= RESIDENT_SLOT_INITIAL_FLAG
    if slot.final:
        flags |= RESIDENT_SLOT_FINAL_FLAG
    return int(slot.slot_id) | flags


def _encode_package_v3(
    slots: Sequence[ResidentV3SlotDescriptor],
    operations: Sequence[ResidentOperationDescriptor],
    *,
    profile: HardwareTaskGraphResidentProfile,
    package_flags: int,
) -> bytes:
    _resident_v3_package_metadata(package_flags)
    slot_payload = b"".join(
        struct.pack(
            RESIDENT_V3_SLOT_FORMAT,
            _encoded_v3_slot_id(item),
            item.offset_bytes,
            item.capacity_elements,
            item.element_count,
            item.element_bytes,
            item.storage_kind,
            item.logical_bytes,
            item.transfer_bytes,
        )
        for item in slots
    )
    operation_payload = b"".join(
        item.to_bytes(operation_abi_version=RESIDENT_OPERATION_ABI_V2)
        for item in operations
    )
    slot_offset = RESIDENT_PACKAGE_HEADER_BYTES
    operation_offset = _align8(slot_offset + len(slot_payload))
    file_bytes = operation_offset + len(operation_payload)
    header = struct.pack(
        RESIDENT_PACKAGE_HEADER_FORMAT,
        RESIDENT_PACKAGE_MAGIC_V3,
        RESIDENT_PACKAGE_VERSION_V3,
        RESIDENT_PACKAGE_ENDIAN,
        RESIDENT_PACKAGE_HEADER_BYTES,
        package_flags,
        file_bytes,
        slot_offset,
        len(slot_payload),
        operation_offset,
        len(operation_payload),
        len(slots),
        len(operations),
        profile.mram_pool_bytes,
        1,
        sum(item.initial for item in slots),
        sum(item.final for item in slots),
        profile.max_rank,
        0,
    )
    return (
        header
        + slot_payload
        + (b"\0" * (operation_offset - slot_offset - len(slot_payload)))
        + operation_payload
    )


def _write_v3_package_request(
    package: ResidentGraphPackage,
    *,
    root: Path,
    request_dir: Path,
    input_dir: Path,
    output_dir: Path,
    dpu_binary: Path,
    request_id: str,
) -> ResidentGraphPackage:
    profile = package.profile or _canonical_profile()
    if profile.version != RESIDENT_V3_PROFILE_VERSION or len(package.operations) != 1:
        raise ValueError("hardware_profile_violation: typed v3 package requires the v3 profile")
    slots = {item.slot_id: item for item in package.typed_slots}
    storage_data = package.storage_initial_data or package.initial_data
    input_entries: list[JsonDict] = []
    initial_h2d_bytes = 0
    for slot_id in sorted(storage_data):
        slot = slots[int(slot_id)]
        if not slot.initial:
            raise ValueError("hardware_profile_violation: v3 input is not an initial slot")
        dtype = np.dtype("i1") if slot.storage_kind == RESIDENT_V3_STORAGE_PACKED_I8 else np.dtype("<f4")
        array = np.ascontiguousarray(np.asarray(storage_data[slot_id], dtype=dtype).ravel())
        payload = array.tobytes(order="C")
        if len(payload) != slot.logical_bytes:
            raise ValueError("hardware_profile_violation: v3 input logical byte count mismatch")
        path = input_dir / f"slot_{int(slot_id):04d}.bin"
        _require_ascii(_relative(root, path), "resident v3 input path")
        path.write_bytes(payload)
        logical_sha256 = hashlib.sha256(payload).hexdigest()
        source = np.ascontiguousarray(
            np.asarray(package.initial_data[int(slot_id)], dtype="<f4").ravel()
        ).tobytes(order="C")
        initial_h2d_bytes += slot.transfer_bytes
        input_entries.append(
            {
                "slot_id": int(slot_id),
                "elements": int(slot.element_count),
                "storage_kind": int(slot.storage_kind),
                "storage_dtype": slot.storage_dtype,
                "element_bytes": int(slot.element_bytes),
                "input_path": _relative(root, path),
                "raw_bytes": int(slot.logical_bytes),
                "transfer_bytes": int(slot.transfer_bytes),
                "logical_sha256": logical_sha256,
                "source_float32_sha256": hashlib.sha256(source).hexdigest(),
                "packed_input_sha256": (
                    logical_sha256
                    if slot.storage_kind == RESIDENT_V3_STORAGE_PACKED_I8
                    else None
                ),
                "scale": package.input_scales.get(int(slot_id)),
            }
        )

    final_paths: dict[str, Path] = {}
    raw_paths: dict[str, Path] = {}
    final_entries: list[JsonDict] = []
    for component, slot_id, elements in package.allocation.final_components:
        slot = slots[int(slot_id)]
        dequant_path = output_dir / f"final_{_safe_name(component)}_f32.bin"
        _require_ascii(_relative(root, dequant_path), "resident v3 output path")
        final_paths[component] = dequant_path
        entry: JsonDict = {
            "component": component,
            "slot_id": int(slot_id),
            "elements": int(elements),
            "storage_kind": int(slot.storage_kind),
            "storage_dtype": slot.storage_dtype,
            "element_bytes": int(slot.element_bytes),
            "output_path": _relative(root, dequant_path),
            "raw_bytes": int(slot.logical_bytes),
            "transfer_bytes": int(slot.transfer_bytes),
        }
        if slot.storage_kind == RESIDENT_V3_STORAGE_I32:
            raw_path = output_dir / f"final_{_safe_name(component)}_i32.bin"
            _require_ascii(_relative(root, raw_path), "resident v3 raw output path")
            raw_paths[component] = raw_path
            entry["raw_output_path"] = _relative(root, raw_path)
        final_entries.append(entry)

    package_bytes = _encode_package_v3(
        package.typed_slots,
        package.operations,
        profile=profile,
        package_flags=package.package_flags,
    )
    validate_resident_graph_package_bytes(
        package_bytes,
        profile=profile,
        operation_abi_version=RESIDENT_OPERATION_ABI_V2,
    )
    package_path = request_dir / "resident_graph_package.bin"
    package_path.write_bytes(package_bytes)
    descriptor_sha256 = hashlib.sha256(package_bytes).hexdigest()
    descriptor_h2d_bytes = _align8(len(package.typed_slots) * RESIDENT_V3_SLOT_BYTES) + _align8(
        len(package.operations) * RESIDENT_OPERATION_BYTES_V2
    )
    final_d2h_bytes = sum(int(item["transfer_bytes"]) for item in final_entries)
    dpu_ref = _relative(root, dpu_binary)
    _require_ascii(dpu_ref, "resident v3 DPU binary path")
    scale_payload = {
        str(slot_id): float(np.float32(value))
        for slot_id, value in sorted(package.input_scales.items())
    }
    scale_metadata_sha256 = canonical_hash(scale_payload)
    manifest_path = root / f"{_safe_name(request_id)}_resident_request.json"
    payload: JsonDict = {
        "schema_version": RESIDENT_SESSION_PROTOCOL,
        "manifest_kind": "resident_graph_request",
        "session_id": request_id,
        "route_id": profile.route_id,
        "backend_id": profile.backend_id,
        "hardware_profile_version": profile.version,
        "target": "hardware",
        "sdk_allocation_profile": RESIDENT_ALLOCATION_PROFILE,
        "session_protocol": profile.session_protocol,
        "package_magic": RESIDENT_PACKAGE_MAGIC_V3.decode("ascii"),
        "package_version": RESIDENT_PACKAGE_VERSION_V3,
        "package_flags": package.package_flags,
        "operation_abi_version": RESIDENT_OPERATION_ABI_V2,
        "operation_bytes": RESIDENT_OPERATION_BYTES_V2,
        "dpu_binary_abi": "dpu_resident_v3",
        "dpu_binary": dpu_ref,
        "package_path": _relative(root, package_path),
        "requested_dpus": profile.requested_dpu_count,
        "requested_dpu_count": profile.requested_dpu_count,
        "tasklets": profile.tasklets_per_dpu,
        "graph_request_count": 1,
        "logical_task_count": 1,
        "component_operation_count": 1,
        "slot_descriptor_count": len(package.typed_slots),
        "mram_pool_bytes": profile.mram_pool_bytes,
        "quantization_mode": package.quantization_mode,
        "numeric_policy": {
            "mode": package.quantization_mode,
            "initial_storage_dtype": (
                "int8" if package.package_flags == RESIDENT_V3_FLAG_HOST_PACKED_INT8 else "float32"
            ),
            "output_storage_dtype": (
                "int32" if package.package_flags == RESIDENT_V3_FLAG_HOST_PACKED_INT8 else "float32"
            ),
            "host_quantization": package.package_flags == RESIDENT_V3_FLAG_HOST_PACKED_INT8,
            "dpu_intermediate_requantization": False,
            "rounding": "nearest_even",
            "clip_range": [-127, 127],
            "scale_formula": "max_abs/127_or_1_for_exact_zero",
            "input_scales": scale_payload,
            "scale_metadata_sha256": scale_metadata_sha256,
        },
        "typed_slots": [item.to_json_dict() for item in package.typed_slots],
        "initial_slots": input_entries,
        "final_outputs": final_entries,
        "initial_h2d_bytes": initial_h2d_bytes,
        "descriptor_h2d_bytes": descriptor_h2d_bytes,
        "descriptor_control_bytes": RESIDENT_DESCRIPTOR_CONTROL_BYTES,
        "control_h2d_bytes_per_launch": RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
        "control_h2d_bytes": RESIDENT_DESCRIPTOR_CONTROL_BYTES + RESIDENT_CONTROL_H2D_BYTES_PER_LAUNCH,
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
    return replace(
        package,
        manifest_path=manifest_path,
        package_path=package_path,
        final_output_paths=final_paths,
        raw_final_output_paths=raw_paths,
        dequant_final_output_paths=final_paths,
        descriptor_sha256=descriptor_sha256,
    )


def _encode_package(
    slots: Sequence[ResidentSlotDescriptor],
    operations: Sequence[ResidentOperationDescriptor],
    *,
    profile: HardwareTaskGraphResidentProfile | None = None,
    operation_abi_version: int = RESIDENT_OPERATION_ABI_V1,
) -> bytes:
    selected = profile or _canonical_profile()
    _require_canonical_profile(selected)
    package_magic, package_version, _operation_bytes, _dpu_binary_abi = _resident_abi_metadata(
        operation_abi_version
    )
    slot_payload = b"".join(
        struct.pack(
            RESIDENT_SLOT_FORMAT,
            _encoded_slot_id(item),
            item.offset_bytes,
            item.capacity_elements,
            item.element_count,
        )
        for item in slots
    )
    slot_offset = RESIDENT_PACKAGE_HEADER_BYTES
    operation_offset = _align8(slot_offset + len(slot_payload))
    operation_payload = b"".join(
        item.to_bytes(operation_abi_version=operation_abi_version) for item in operations
    )
    file_bytes = operation_offset + len(operation_payload)
    header = struct.pack(
        RESIDENT_PACKAGE_HEADER_FORMAT,
        package_magic,
        package_version,
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
        selected.mram_pool_bytes,
        1,
        sum(1 for item in slots if any(lifetime.initial for lifetime in item.lifetimes)),
        sum(1 for item in slots if any(lifetime.final for lifetime in item.lifetimes)),
        selected.max_rank,
        0,
    )
    return header + slot_payload + (b"\0" * (operation_offset - slot_offset - len(slot_payload))) + operation_payload


def _descriptor_transfer_bytes(
    slot_count: int,
    operation_count: int,
    operation_abi_version: int = RESIDENT_OPERATION_ABI_V1,
) -> int:
    # Descriptor bytes are reported separately from the 16-byte control block
    # and the 8-byte active-operation writes so application-visible H2D is
    # exactly initial + descriptor + control.
    return (
        _align8(slot_count * RESIDENT_SLOT_BYTES)
        + _align8(
            operation_count
            * _resident_abi_metadata(operation_abi_version)[2]
        )
    )


def _canonical_profile(
    tasklets_per_dpu: int = 1,
    *,
    version: str = RESIDENT_PROFILE_VERSION,
    requested_dpu_count: int = 1,
) -> HardwareTaskGraphResidentProfile:
    if version in {RESIDENT_PROFILE_VERSION, RESIDENT_EXECUTION_PLAN_PROFILE_VERSION}:
        output_tile_elements = RESIDENT_OUTPUT_TILE_ELEMENTS
        max_elements = RESIDENT_MAX_ELEMENTS
        max_logical_tasks = RESIDENT_MAX_LOGICAL_TASKS
        max_component_ops = RESIDENT_MAX_COMPONENT_OPS
        max_contracted_combinations = RESIDENT_MAX_CONTRACTED_COMBINATIONS
        supported_tasklets = (
            (1,)
            if version == RESIDENT_EXECUTION_PLAN_PROFILE_VERSION
            else RESIDENT_SUPPORTED_TASKLETS
        )
    elif version == RESIDENT_M46_PROFILE_VERSION:
        output_tile_elements = RESIDENT_M46_OUTPUT_TILE_ELEMENTS
        max_elements = RESIDENT_MAX_ELEMENTS
        max_logical_tasks = RESIDENT_MAX_LOGICAL_TASKS
        max_component_ops = RESIDENT_MAX_COMPONENT_OPS
        max_contracted_combinations = RESIDENT_MAX_CONTRACTED_COMBINATIONS
        supported_tasklets = RESIDENT_SUPPORTED_TASKLETS
    elif version == RESIDENT_V3_PROFILE_VERSION:
        output_tile_elements = RESIDENT_V3_OUTPUT_TILE_ELEMENTS
        max_elements = RESIDENT_V3_MAX_ELEMENTS
        max_logical_tasks = RESIDENT_V3_MAX_LOGICAL_TASKS
        max_component_ops = RESIDENT_V3_MAX_COMPONENT_OPS
        max_contracted_combinations = RESIDENT_V3_MAX_ELEMENTS
        supported_tasklets = RESIDENT_V3_SUPPORTED_TASKLETS
    else:
        raise ValueError(f"hardware_profile_violation: unsupported resident profile {version}")
    if tasklets_per_dpu not in supported_tasklets:
        if version != RESIDENT_V3_PROFILE_VERSION:
            raise ValueError(
                "hardware_profile_violation: tasklets_per_dpu must be one of 1, 2, 4, 8, 16"
            )
        raise ValueError("hardware_profile_violation: unsupported resident tasklet count")
    if version in {RESIDENT_PROFILE_VERSION, RESIDENT_M46_PROFILE_VERSION} and int(requested_dpu_count) != 1:
        raise ValueError(
            "hardware_profile_violation: legacy resident profiles require requested_dpu_count=1"
        )
    if not 1 <= int(requested_dpu_count) <= RESIDENT_V3_MAX_DPUS:
        raise ValueError("hardware_profile_violation: requested DPU count is outside the v3 cap")
    return HardwareTaskGraphResidentProfile(
        version=version,
        target="hardware",
        backend_id=(
            RESIDENT_EXECUTION_PLAN_BACKEND_ID
            if version == RESIDENT_EXECUTION_PLAN_PROFILE_VERSION
            else RESIDENT_BACKEND_ID
        ),
        route_id=(
            RESIDENT_EXECUTION_PLAN_ROUTE_ID
            if version == RESIDENT_EXECUTION_PLAN_PROFILE_VERSION
            else RESIDENT_ROUTE_ID
        ),
        session_protocol=RESIDENT_SESSION_PROTOCOL,
        timing_scope=RESIDENT_TIMING_SCOPE,
        requested_dpu_count=int(requested_dpu_count),
        tasklets_per_dpu=tasklets_per_dpu,
        max_rank=RESIDENT_MAX_RANK,
        max_tensor_elements=max_elements,
        max_logical_tasks=max_logical_tasks,
        max_component_ops=max_component_ops,
        max_slot_descriptors=RESIDENT_MAX_SLOT_DESCRIPTORS,
        mram_pool_bytes=RESIDENT_MRAM_POOL_BYTES,
        max_contracted_combinations=max_contracted_combinations,
        output_tile_elements=output_tile_elements,
        numeric_modes=(
            RESIDENT_V3_NUMERIC_MODES
            if version == RESIDENT_V3_PROFILE_VERSION
            else RESIDENT_NUMERIC_MODES
        ),
        complex_policy=RESIDENT_COMPLEX_POLICY,
        synchronous_execution=True,
        timeout_s=RESIDENT_TIMEOUT_S,
        performance_claim_applicable=False,
    )


def canonical_execution_plan_resident_profile(
    requested_dpu_count: int,
) -> HardwareTaskGraphResidentProfile:
    """Return the exact-count profile used by execution-plan packages."""

    return _canonical_profile(
        1,
        version=RESIDENT_EXECUTION_PLAN_PROFILE_VERSION,
        requested_dpu_count=requested_dpu_count,
    )


def _parse_profile(value: object) -> HardwareTaskGraphResidentProfile:
    if not isinstance(value, Mapping):
        raise ValueError("hardware_profile_violation: resident hardware_profile must be a mapping")
    version = value.get("hardware_profile_version")
    if version not in {
        RESIDENT_PROFILE_VERSION,
        RESIDENT_M46_PROFILE_VERSION,
        RESIDENT_EXECUTION_PLAN_PROFILE_VERSION,
        RESIDENT_V3_PROFILE_VERSION,
    }:
        raise ValueError(f"hardware_profile_violation: unsupported resident profile {version}")
    tasklets = int(value.get("tasklets_per_dpu", 1))
    requested_dpus = int(value.get("requested_dpu_count", 1))
    supported_tasklets = (
        RESIDENT_V3_SUPPORTED_TASKLETS
        if version == RESIDENT_V3_PROFILE_VERSION
        else (1,) if version == RESIDENT_EXECUTION_PLAN_PROFILE_VERSION else RESIDENT_SUPPORTED_TASKLETS
    )
    if tasklets not in supported_tasklets:
        message = (
            "hardware_profile_violation: tasklets_per_dpu must be one of 1, 2, 4, 8, 16"
            if version != RESIDENT_V3_PROFILE_VERSION
            else "hardware_profile_violation: unsupported resident tasklet count"
        )
        raise ValueError(
            message
        )
    if version in {RESIDENT_PROFILE_VERSION, RESIDENT_EXECUTION_PLAN_PROFILE_VERSION} and tasklets != 1:
        raise ValueError("hardware_profile_violation: resident v1 profile is one-tasklet only")
    if version in {RESIDENT_PROFILE_VERSION, RESIDENT_M46_PROFILE_VERSION} and requested_dpus != 1:
        raise ValueError(
            "hardware_profile_violation: legacy resident profiles require requested_dpu_count=1"
        )
    expected = _canonical_profile(tasklets, version=version, requested_dpu_count=requested_dpus).to_json_dict()
    if set(value) != set(expected):
        raise ValueError("hardware_profile_violation: resident profile keys differ")
    for key, expected_value in expected.items():
        actual_value = value.get(key)
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError(f"hardware_profile_violation: {key} must be {expected_value!r}")
    return _canonical_profile(
        tasklets, version=version, requested_dpu_count=requested_dpus
    )


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
    # Complex storage with an exactly zero imaginary part follows the real
    # execution path. Split components are allocated only when a complex
    # operation actually requires them, avoiding dead initial input slots in
    # the native manifest ABI.
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


def _slot_flags(slot: ResidentSlotDescriptor) -> int:
    flags = 0
    if any(lifetime.initial for lifetime in slot.lifetimes):
        flags |= RESIDENT_SLOT_INITIAL_FLAG
    if any(lifetime.final for lifetime in slot.lifetimes):
        flags |= RESIDENT_SLOT_FINAL_FLAG
    return flags


def _encoded_slot_id(slot: ResidentSlotDescriptor) -> int:
    if slot.slot_id < 0 or slot.slot_id > RESIDENT_SLOT_ID_MASK:
        raise ResidentCapacityError("hardware_profile_violation: resident slot id overflow")
    return int(slot.slot_id) | _slot_flags(slot)


def _require_canonical_profile(profile: HardwareTaskGraphResidentProfile) -> None:
    supported_tasklets = (
        RESIDENT_V3_SUPPORTED_TASKLETS
        if profile.version == RESIDENT_V3_PROFILE_VERSION
        else (1,)
        if profile.version == RESIDENT_EXECUTION_PLAN_PROFILE_VERSION
        else RESIDENT_SUPPORTED_TASKLETS
    )
    if profile.tasklets_per_dpu not in supported_tasklets:
        message = (
            "hardware_profile_violation: tasklets_per_dpu must be one of 1, 2, 4, 8, 16"
            if profile.version != RESIDENT_V3_PROFILE_VERSION
            else "hardware_profile_violation: unsupported resident tasklet count"
        )
        raise ResidentCapacityError(
            message
        )
    if profile.version in {RESIDENT_PROFILE_VERSION, RESIDENT_EXECUTION_PLAN_PROFILE_VERSION} and profile.tasklets_per_dpu != 1:
        raise ResidentCapacityError(
            "hardware_profile_violation: resident v1 profile is one-tasklet only"
        )
    if profile.version not in {
        RESIDENT_PROFILE_VERSION,
        RESIDENT_M46_PROFILE_VERSION,
        RESIDENT_EXECUTION_PLAN_PROFILE_VERSION,
        RESIDENT_V3_PROFILE_VERSION,
    }:
        raise ResidentCapacityError(
            "hardware_profile_violation: unsupported resident profile version"
        )
    if profile.version in {RESIDENT_PROFILE_VERSION, RESIDENT_M46_PROFILE_VERSION} and profile.requested_dpu_count != 1:
        raise ResidentCapacityError(
            "hardware_profile_violation: legacy resident profiles require requested_dpu_count=1"
        )
    if profile.to_json_dict() != _canonical_profile(
        profile.tasklets_per_dpu,
        version=profile.version,
        requested_dpu_count=profile.requested_dpu_count,
    ).to_json_dict():
        raise ResidentCapacityError(
            "hardware_profile_violation: resident package ABI requires the canonical frozen profile"
        )


def _validate_finite_inputs(network: Any) -> None:
    for tensor in network.tensors:
        try:
            array = np.asarray(tensor.array)
            if np.iscomplexobj(array):
                converted_parts = (
                    np.asarray(array.real, dtype=np.float32),
                    np.asarray(array.imag, dtype=np.float32),
                )
                finite = np.all(np.isfinite(array)) and all(
                    np.all(np.isfinite(part)) for part in converted_parts
                )
            else:
                converted = np.asarray(array, dtype=np.float32)
                finite = np.all(np.isfinite(array)) and np.all(np.isfinite(converted))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ResidentCapacityError(
                f"hardware_profile_violation: resident_non_numeric_input:{tensor.spec.id}"
            ) from exc
        if not finite:
            raise ResidentCapacityError(
                f"hardware_profile_violation: resident_non_finite_input:{tensor.spec.id}"
            )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("hardware_profile_violation: resident package path must be under session root") from exc


def _require_ascii(value: str, label: str) -> None:
    if not value.isascii():
        raise ValueError(f"hardware_profile_violation: {label} must be ASCII")


def _array_hash_json(value: Any) -> str:
    return canonical_hash({"array_hash": _array_hash(value)})


# Public aliases make the allocator/package boundary easy to test without
# coupling tests to the route's CLI entry point.
build_resident_slot_lifetime_map = allocate_resident_slots
build_resident_package = build_resident_graph_package
validate_resident_package = validate_resident_graph_package_bytes
