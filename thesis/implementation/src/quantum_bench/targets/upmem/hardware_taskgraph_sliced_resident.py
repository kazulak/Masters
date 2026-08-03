"""Planning and package bridge for the bounded M2 two-slice resident route.

The scientific TaskGraph remains intact while each resident package receives a
materialized single-value restriction of the selected contracted dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import (
    JsonDict,
    TaskGraph,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
    to_jsonable,
)
from quantum_bench.targets.upmem.execution_plan import (
    DpuResourceContext,
    UpmemCommunicationPlan,
    UpmemExecutionPlan,
    UpmemKernelPlan,
    UpmemNumericPlan,
    UpmemPlacementPlan,
    UpmemSchedulePlan,
    validate_execution_plan_graph_identity,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    HardwareTaskGraphResidentProfile,
    ResidentGraphPackage,
    build_resident_graph_package,
)
from quantum_bench.tn.execution_bundle import with_execution_identity
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.tn.slicing import (
    SliceAwareTaskGraphModel,
    SliceInputRestriction,
    SliceModelTask,
    SliceReconstructionStep,
    build_slice_aware_taskgraph_model,
    validate_slice_aware_taskgraph_model,
)


SLICED_RESIDENT_PLAN_SCHEMA_VERSION = "upmem_sliced_resident_plan_v1"
SLICED_RESIDENT_EXECUTION_SCHEMA_VERSION = "upmem_sliced_resident_execution_v1"
SLICED_RESIDENT_OUTER_ROUTE_ID = "upmem_tn_hardware_sliced_resident_two_dpu"
SLICED_RESIDENT_OUTER_BACKEND_ID = "upmem_sdk_hardware_sliced_resident_two_dpu"
SLICED_RESIDENT_OUTER_PROFILE_VERSION = "hardware_sliced_resident_two_dpu_m2_v1"
SLICED_RESIDENT_OUTER_TIMING_SCOPE = (
    "two_dpu_sliced_resident_host_observed_bringup_v1"
)


@dataclass(frozen=True)
class SlicedResidentGraphPackage:
    """A resident package bound to exactly one validated source slice."""

    slice_plan: SlicedResidentSlicePlan
    package: ResidentGraphPackage
    network: TensorNetworkValue
    restrictions: tuple[SliceInputRestriction, ...]
    restricted_input_sha256: Mapping[str, str]
    restricted_input_fnv1a64: Mapping[str, str]
    source_hashes: Mapping[str, str]
    resident_descriptor_fnv1a64: str | None = None

    @property
    def slice_id(self) -> int:
        return self.slice_plan.slice_id

    @property
    def dpu_id(self) -> int:
        return self.slice_plan.dpu_id


@dataclass(frozen=True)
class SlicedResidentSlicePlan:
    """One source slice with its deterministic DPU ownership."""

    slice_task: SliceModelTask
    dpu_id: int

    @property
    def slice_id(self) -> int:
        return self.slice_task.slice_id

    def to_json_dict(self) -> JsonDict:
        return self.slice_task.to_json() | {"dpu_id": self.dpu_id}


@dataclass(frozen=True)
class SlicedResidentPlan:
    """Immutable two-slice plan plus host-side reconstruction contract."""

    graph: TaskGraph
    network: TensorNetworkValue
    model: SliceAwareTaskGraphModel
    execution_plan: UpmemExecutionPlan
    slice_plans: tuple[SlicedResidentSlicePlan, ...]
    reconstruction_step: SliceReconstructionStep

    @property
    def circuit_semantics_hash(self) -> str:
        return self.execution_plan.circuit_semantics_hash

    @property
    def tensor_network_hash(self) -> str:
        return self.execution_plan.tensor_network_hash

    @property
    def contraction_plan_hash(self) -> str:
        return self.execution_plan.contraction_plan_hash

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(
            {
                "schema_version": SLICED_RESIDENT_PLAN_SCHEMA_VERSION,
                "slice_count": len(self.slice_plans),
                "slices": self.slice_plans,
                "reconstruction": self.reconstruction_step,
                "execution_plan": self.execution_plan,
                "circuit_semantics_hash": self.circuit_semantics_hash,
                "tensor_network_hash": self.tensor_network_hash,
                "contraction_plan_hash": self.contraction_plan_hash,
            }
        )


def build_two_slice_resident_plan(
    graph: TaskGraph,
    network: TensorNetworkValue,
    *,
    model: SliceAwareTaskGraphModel | None = None,
) -> SlicedResidentPlan:
    """Build exactly two resident slices under the bounded M2 contract.

    The historical M2 fixture slices a dependency-free terminal task.  M2.1
    additionally permits one terminal task with a dependent prefix when the
    model explicitly declares ``dependent_prefix_replicated``.  Each DPU then
    receives and executes the complete restricted graph; no prefix result is
    injected from the CPU.
    """

    if network.spec != graph.network:
        raise ValueError("sliced_resident_network_identity_mismatch")
    selected_model = model or build_slice_aware_taskgraph_model(
        graph, max_slice_count=2
    )
    valid, reason = validate_slice_aware_taskgraph_model(selected_model)
    if not valid:
        raise ValueError(
            f"sliced_resident_model_unsupported:{reason or 'invalid_model'}"
        )
    if (
        selected_model.slice_model_slice_count != 2
        or len(selected_model.slice_tasks) != 2
    ):
        raise ValueError("sliced_resident_requires_exactly_two_slices")
    if (
        selected_model.sliced_task_id is None
        or selected_model.reconstruction_step is None
    ):
        raise ValueError("sliced_resident_model_missing_selection")
    selected_index = next(
        (
            index
            for index, task in enumerate(graph.tasks)
            if task.id == selected_model.sliced_task_id
        ),
        None,
    )
    if selected_index is None:
        raise ValueError("sliced_resident_model_source_task_mismatch")
    if selected_index != len(graph.tasks) - 1:
        if selected_model.slice_model_kind != "dependent_prefix_replicated":
            raise ValueError("sliced_resident_terminal_single_task_graph")
        raise ValueError("sliced_resident_selected_task_must_be_terminal")
    if len(graph.tasks) > 1 and selected_model.slice_model_kind != "dependent_prefix_replicated":
        raise ValueError("sliced_resident_dependent_prefix_contract_missing")
    if len(graph.tasks) > 2:
        raise ValueError("sliced_resident_prefix_task_cap_exceeded")
    _validate_model_binding(graph, selected_model)

    slice_tasks = tuple(
        sorted(selected_model.slice_tasks, key=lambda task: task.slice_id)
    )
    if tuple(task.slice_id for task in slice_tasks) != (0, 1):
        raise ValueError("sliced_resident_slice_ids_must_be_zero_and_one")
    if tuple(task.assignment.value for task in slice_tasks) != (0, 1):
        raise ValueError("sliced_resident_assignment_values_must_match_slice_ids")
    slice_plans = tuple(
        SlicedResidentSlicePlan(slice_task=task, dpu_id=task.slice_id)
        for task in slice_tasks
    )
    if tuple(plan.dpu_id for plan in slice_plans) != (0, 1):
        raise ValueError("sliced_resident_dpu_assignment_mismatch")

    execution_plan = _build_execution_plan(graph)
    validate_execution_plan_graph_identity(execution_plan, graph)
    result = SlicedResidentPlan(
        graph=graph,
        network=network,
        model=selected_model,
        execution_plan=execution_plan,
        slice_plans=slice_plans,
        reconstruction_step=selected_model.reconstruction_step,
    )
    valid, reason = validate_two_slice_resident_plan(result)
    if not valid:
        raise ValueError(f"sliced_resident_plan_invalid:{reason or 'invalid_plan'}")
    return result


def build_two_slice_resident_graph_packages(
    plan: SlicedResidentPlan,
    *,
    case_id: str,
    suite_id: str,
    quantization_mode: str,
    profile: HardwareTaskGraphResidentProfile | None = None,
) -> tuple[SlicedResidentGraphPackage, ...]:
    """Materialize the two validated slices as independent resident packages."""

    valid, reason = validate_two_slice_resident_plan(plan)
    if not valid:
        raise ValueError(f"sliced_resident_plan_invalid:{reason or 'invalid_plan'}")
    _validate_m2_source_inputs(plan)
    source_hashes = {
        "circuit_semantics_hash": plan.circuit_semantics_hash,
        "tensor_network_hash": plan.tensor_network_hash,
        "contraction_plan_hash": plan.contraction_plan_hash,
    }
    packages: list[SlicedResidentGraphPackage] = []
    for slice_plan in plan.slice_plans:
        graph, network, restricted_hashes, restricted_fnv = (
            _materialize_restricted_slice(plan, slice_plan)
        )
        package = build_resident_graph_package(
            graph,
            network,
            case_id=case_id,
            suite_id=suite_id,
            quantization_mode=quantization_mode,
            profile=profile,
        )
        packages.append(
            SlicedResidentGraphPackage(
                slice_plan=slice_plan,
                package=package,
                network=network,
                restrictions=slice_plan.slice_task.input_restrictions,
                restricted_input_sha256=restricted_hashes,
                restricted_input_fnv1a64=restricted_fnv,
                source_hashes=source_hashes,
            )
        )
    return tuple(packages)


def write_two_slice_resident_graph_packages(
    packages: tuple[SlicedResidentGraphPackage, ...],
    session_root: Path,
    *,
    dpu_binary: Path,
    request_id_prefix: str = "sliced-resident",
) -> tuple[SlicedResidentGraphPackage, ...]:
    """Write each slice request and bind it to its source-slice contract."""

    if len(packages) != 2 or tuple(item.slice_id for item in packages) != (0, 1):
        raise ValueError("sliced_resident_packages_must_have_slice_ids_0_and_1")
    if len({item.dpu_id for item in packages}) != 2:
        raise ValueError("sliced_resident_packages_must_have_distinct_dpu_ids")

    written: list[SlicedResidentGraphPackage] = []
    for item in packages:
        package = item.package.write(
            session_root,
            dpu_binary=dpu_binary,
            request_id=f"{request_id_prefix}-slice-{item.slice_id}",
        )
        if package.manifest_path is None or package.descriptor_sha256 is None:
            raise ValueError("sliced_resident_package_write_incomplete")
        fingerprints = _written_input_fingerprints(
            package,
            package.manifest_path.parent,
        )
        descriptor = _file_fingerprints(package.package_path)
        manifest = _slice_execution_manifest(
            item,
            package,
            restricted_input_sha256=fingerprints["sha256"],
            restricted_input_fnv1a64=fingerprints["fnv1a64"],
            resident_descriptor_sha256=descriptor["sha256"],
            resident_descriptor_fnv1a64=descriptor["fnv1a64"],
        )
        write_json(package.manifest_path, manifest)
        written.append(
            replace(
                item,
                package=package,
                restricted_input_sha256=fingerprints["sha256"],
                restricted_input_fnv1a64=fingerprints["fnv1a64"],
                resident_descriptor_fnv1a64=descriptor["fnv1a64"],
            )
        )

    if len({item.package.package_path for item in written}) != 2:
        raise ValueError("sliced_resident_packages_must_be_distinct")
    if len({item.package.manifest_path for item in written}) != 2:
        raise ValueError("sliced_resident_manifests_must_be_distinct")
    if len(
        {path for item in written for path in item.package.final_output_paths.values()}
    ) != sum(len(item.package.final_output_paths) for item in written):
        raise ValueError("sliced_resident_output_paths_must_be_distinct")
    return tuple(written)


def validate_written_two_slice_packages(
    plan: SlicedResidentPlan,
    packages: tuple[SlicedResidentGraphPackage, ...],
) -> JsonDict:
    """Fail closed unless both written packages exactly match ``plan`` and disk."""

    valid, reason = validate_two_slice_resident_plan(plan)
    if not valid:
        raise ValueError(f"sliced_resident_plan_invalid:{reason or 'invalid_plan'}")
    _validate_m2_source_inputs(plan)
    _validate_written_package_assignments(plan, packages)

    validated: dict[str, JsonDict] = {}
    for item in packages:
        package = item.package
        if package.manifest_path is None or package.package_path is None:
            raise ValueError("sliced_resident_package_write_incomplete")
        if not package.manifest_path.is_file() or not package.package_path.is_file():
            raise ValueError("sliced_resident_written_package_missing")
        manifest = _load_json_object(
            package.manifest_path, "sliced_resident_manifest_invalid"
        )
        root = package.manifest_path.parent.resolve()
        if (
            _manifest_file_path(root, manifest.get("package_path"), "package_path")
            != package.package_path.resolve()
        ):
            raise ValueError("sliced_resident_manifest_package_path_mismatch")

        inputs = _written_input_fingerprints(
            package,
            root,
            manifest=manifest,
        )
        descriptor = _file_fingerprints(package.package_path)
        if package.descriptor_sha256 != descriptor["sha256"]:
            raise ValueError("sliced_resident_descriptor_sha256_mismatch")
        binding = manifest.get("slice_execution")
        if not isinstance(binding, Mapping):
            raise ValueError("sliced_resident_manifest_slice_execution_missing")
        _validate_slice_execution_binding(
            plan,
            item,
            binding,
            restricted_input_sha256=inputs["sha256"],
            restricted_input_fnv1a64=inputs["fnv1a64"],
            resident_descriptor_sha256=descriptor["sha256"],
            resident_descriptor_fnv1a64=descriptor["fnv1a64"],
        )
        if (
            dict(item.restricted_input_sha256) != inputs["sha256"]
            or dict(item.restricted_input_fnv1a64) != inputs["fnv1a64"]
            or item.resident_descriptor_fnv1a64 != descriptor["fnv1a64"]
        ):
            raise ValueError("sliced_resident_written_provenance_mismatch")
        validated[str(item.slice_id)] = {
            "manifest_path": str(package.manifest_path.resolve()),
            "package_path": str(package.package_path.resolve()),
            "restricted_input_sha256": inputs["sha256"],
            "restricted_input_fnv1a64": inputs["fnv1a64"],
            "resident_descriptor_sha256": descriptor["sha256"],
            "resident_descriptor_fnv1a64": descriptor["fnv1a64"],
        }
    return {
        "validated": True,
        "source_hashes": dict(packages[0].source_hashes),
        "slices": validated,
    }


def load_and_reconstruct_two_slice_native_outputs(
    plan: SlicedResidentPlan,
    packages: tuple[SlicedResidentGraphPackage, ...],
    native_response: Mapping[str, Any] | Path,
    *,
    reference_partials: Mapping[int, Any] | None = None,
) -> tuple[np.ndarray, JsonDict]:
    """Load two verified native float32 partials and reconstruct on the host."""

    package_validation = validate_written_two_slice_packages(plan, packages)
    response, response_path = _load_native_response(native_response)
    _validate_native_two_slice_response(response, packages, response_path)
    requires_slice_references = (
        plan.model.slice_model_kind == "dependent_prefix_replicated"
    )
    if requires_slice_references and reference_partials is None:
        raise ValueError("sliced_resident_cpu_partial_reference_missing")

    partials: dict[int, np.ndarray] = {}
    output_hashes: dict[str, JsonDict] = {}
    response_slices = {entry["slice_id"]: entry for entry in response["slices"]}
    for item in packages:
        expected_path = _single_final_output_path(item.package)
        entry = response_slices[item.slice_id]
        if not _response_path_matches(
            entry["partial_output_path"], expected_path, response_path
        ):
            raise ValueError("sliced_resident_native_partial_output_path_mismatch")
        expected_bytes = int(np.prod(item.slice_plan.slice_task.output_shape)) * 4
        if expected_path.stat().st_size != expected_bytes:
            raise ValueError("sliced_resident_native_partial_output_size_mismatch")
        raw = np.fromfile(expected_path, dtype="<f4")
        try:
            partial = raw.reshape(item.slice_plan.slice_task.output_shape)
        except ValueError as exc:
            raise ValueError(
                "sliced_resident_native_partial_output_shape_mismatch"
            ) from exc
        if not np.all(np.isfinite(partial)):
            raise ValueError("sliced_resident_native_partial_output_non_finite")
        partials[item.slice_id] = partial
        if reference_partials is None:
            reference = None
        else:
            try:
                reference = np.asarray(
                    reference_partials[item.slice_id], dtype=np.float32
                )
            except KeyError as exc:
                raise ValueError("sliced_resident_cpu_partial_reference_missing") from exc
        if reference is not None and reference.shape != partial.shape:
            raise ValueError("sliced_resident_cpu_partial_shape_mismatch")
        observed_operation_count = entry.get(
            "observed_operation_completion_count"
        )
        if observed_operation_count is None:
            observed_operation_count = (
                entry.get("completed_operation_count")
                or entry.get("operation_count")
                or 1
            )
        completion_confirmed = (
            entry.get(
                "operation_completion_confirmed", entry.get("completion_confirmed")
            )
            is True
        )
        sentinel = entry.get("dpu_completion_sentinel")
        sentinel_verified = (
            isinstance(sentinel, Mapping)
            and sentinel.get("verified") is True
        )
        max_abs_error = (
            None
            if reference is None
            else float(np.max(np.abs(partial - reference)))
        )
        output_hashes[str(item.slice_id)] = {
            "path": str(expected_path.resolve()),
            "sha256": _file_fingerprints(expected_path)["sha256"],
            "fnv1a64": _file_fingerprints(expected_path)["fnv1a64"],
            "bytes": expected_bytes,
            "raw_bytes": expected_bytes,
            "transfer_bytes": (expected_bytes + 7) & ~7,
            "norm": float(np.linalg.norm(partial)),
            "max_abs": float(np.max(np.abs(partial))) if partial.size else 0.0,
            "nonzero_element_count": int(
                np.count_nonzero(np.abs(partial) > 1.0e-7)
            ),
            "nonzero_threshold": 1.0e-7,
            "operation_count": int(item.package.component_operation_count),
            "planned_operation_count": int(item.package.component_operation_count),
            "observed_operation_completion_count": int(observed_operation_count),
            "completion_confirmed": completion_confirmed,
            "dpu_completion_sentinel": sentinel,
            "dpu_completion_sentinel_verified": sentinel_verified,
            "cpu_reference_max_abs_error": max_abs_error,
            "cpu_reference_l2_error": (
                None
                if reference is None
                else float(np.linalg.norm(partial - reference))
            ),
        }
    output = reconstruct_host_slice_outputs(plan, partials)
    useful = all(
        output_hashes[str(slice_id)]["norm"] > 1.0e-7
        and output_hashes[str(slice_id)]["nonzero_element_count"] > 0
        and output_hashes[str(slice_id)]["completion_confirmed"]
        and (
            plan.model.slice_model_kind != "dependent_prefix_replicated"
            or output_hashes[str(slice_id)]["dpu_completion_sentinel_verified"]
        )
        and output_hashes[str(slice_id)]["observed_operation_completion_count"] > 0
        for slice_id in (0, 1)
    )
    return output, {
        "package_validation": package_validation,
        "native_response_sha256": _response_fingerprint(
            response, response_path, "sha256"
        ),
        "native_response_fnv1a64": _response_fingerprint(
            response, response_path, "fnv1a64"
        ),
        "partial_outputs": output_hashes,
        "per_slice_output_validation_status": (
            "passed"
            if all(
                item["cpu_reference_max_abs_error"] is None
                or item["cpu_reference_max_abs_error"] <= 1.0e-5
                for item in output_hashes.values()
            )
            else "failed"
        ),
        "slice_useful_work": {
            "status": "passed" if useful else "failed",
            "required": True,
            "threshold": 1.0e-7,
            "slice_count": 2,
            "completed_slice_count": sum(
                int(
                    item["completion_confirmed"]
                    and item["observed_operation_completion_count"] > 0
                )
                for item in output_hashes.values()
            ),
            "nonzero_slice_count": sum(
                int(item["norm"] > 1.0e-7) for item in output_hashes.values()
            ),
        },
    }


def reconstruct_host_slice_outputs(
    plan: SlicedResidentPlan,
    partial_outputs: Mapping[int, Any],
) -> np.ndarray:
    """Sum the two DPU partials on the host in slice order."""

    valid, reason = validate_two_slice_resident_plan(plan)
    if not valid:
        raise ValueError(f"sliced_resident_plan_invalid:{reason or 'invalid_plan'}")
    if set(partial_outputs) != {0, 1}:
        raise ValueError("sliced_resident_reconstruction_requires_slice_ids_0_and_1")
    arrays = [np.asarray(partial_outputs[slice_id]) for slice_id in (0, 1)]
    expected_shape = plan.slice_plans[0].slice_task.output_shape
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError("sliced_resident_partial_shape_mismatch")
    return np.sum(np.stack(arrays, axis=0), axis=0)


def validate_two_slice_resident_plan(
    plan: SlicedResidentPlan,
) -> tuple[bool, str | None]:
    """Validate ownership, graph identity, and host reconstruction metadata."""

    try:
        validate_execution_plan_graph_identity(plan.execution_plan, plan.graph)
    except ValueError:
        return False, "execution_plan_graph_identity_mismatch"
    if plan.network.spec != plan.graph.network:
        return False, "network_identity_mismatch"
    valid, reason = validate_slice_aware_taskgraph_model(plan.model)
    if not valid:
        return False, reason or "invalid_slice_model"
    if tuple(item.slice_task for item in plan.slice_plans) != plan.model.slice_tasks:
        return False, "slice_task_model_mismatch"
    if plan.reconstruction_step != plan.model.reconstruction_step:
        return False, "reconstruction_step_model_mismatch"
    try:
        _validate_model_binding(plan.graph, plan.model)
    except ValueError as exc:
        return False, str(exc)
    if len(plan.slice_plans) != 2:
        return False, "slice_plan_count_mismatch"
    if tuple(item.slice_id for item in plan.slice_plans) != (0, 1):
        return False, "slice_id_order_mismatch"
    if tuple(item.slice_task.assignment.value for item in plan.slice_plans) != (0, 1):
        return False, "assignment_values_mismatch"
    if tuple(item.dpu_id for item in plan.slice_plans) != (0, 1):
        return False, "dpu_assignment_mismatch"
    selected_source = next(
        task for task in plan.graph.tasks if task.id == plan.model.sliced_task_id
    )
    if any(item.slice_task.dependencies != selected_source.dependencies for item in plan.slice_plans):
        return False, "slice_task_dependency_mismatch"
    if selected_source.dependencies and plan.model.slice_model_kind != "dependent_prefix_replicated":
        return False, "dependent_prefix_contract_missing"
    if plan.reconstruction_step.dependencies != tuple(
        item.slice_task.id for item in plan.slice_plans
    ):
        return False, "reconstruction_dependency_mismatch"
    if plan.reconstruction_step.input_tensor_ids != tuple(
        item.slice_task.partial_output_tensor_id for item in plan.slice_plans
    ):
        return False, "reconstruction_input_mismatch"
    if plan.execution_plan.placement.resources.requested_dpu_count != 2:
        return False, "requested_dpu_count_mismatch"
    return True, None


def _materialize_restricted_slice(
    plan: SlicedResidentPlan,
    slice_plan: SlicedResidentSlicePlan,
) -> tuple[TaskGraph, TensorNetworkValue, dict[str, str], dict[str, str]]:
    source_task = next(
        task for task in plan.graph.tasks if task.id == plan.model.sliced_task_id
    )
    slice_task = slice_plan.slice_task
    tensors = {tensor.spec.id: tensor for tensor in plan.network.tensors}
    restricted_values: dict[str, np.ndarray] = {}
    restricted_specs: dict[str, TensorSpec] = {}
    hashes: dict[str, str] = {}
    fnv_hashes: dict[str, str] = {}
    seen_tensor_ids: set[str] = set()

    for restriction in slice_task.input_restrictions:
        if restriction.tensor_id in seen_tensor_ids:
            raise ValueError("sliced_resident_overlapping_input_restrictions")
        seen_tensor_ids.add(restriction.tensor_id)
        tensor = tensors.get(restriction.tensor_id)
        if tensor is None:
            raise ValueError("sliced_resident_missing_restricted_input")
        if (
            restriction.axis < 0
            or restriction.axis >= len(tensor.spec.labels)
            or tensor.spec.labels[restriction.axis] != restriction.label
            or restriction.value < 0
            or restriction.value >= tensor.spec.shape[restriction.axis]
        ):
            raise ValueError("sliced_resident_invalid_input_restriction")
        value = np.take(
            _m2_real_float32(tensor.array), [restriction.value], axis=restriction.axis
        )
        spec = replace(
            tensor.spec, shape=tuple(int(dim) for dim in value.shape), dtype="float32"
        )
        restricted_values[restriction.tensor_id] = value
        restricted_specs[restriction.tensor_id] = spec
        hashes[restriction.tensor_id] = _array_sha256(value)
        fnv_hashes[restriction.tensor_id] = _array_fnv1a64(value)

    if len(restricted_values) != 2:
        raise ValueError("sliced_resident_restrictions_must_cover_two_initial_inputs")

    network_tensors: list[TensorValue] = []
    network_specs: list[TensorSpec] = []
    for tensor in plan.network.tensors:
        spec = restricted_specs.get(tensor.spec.id, tensor.spec)
        value = restricted_values.get(tensor.spec.id, np.asarray(tensor.array))
        network_specs.append(spec)
        network_tensors.append(TensorValue(spec, value))
    network_spec = TensorNetworkSpec(
        circuit=plan.network.spec.circuit,
        tensors=tuple(network_specs),
        output_labels=plan.network.spec.output_labels,
        einsum_expression=plan.network.spec.einsum_expression,
    )
    shape_by_tensor = {tensor.spec.id: tensor.spec.shape for tensor in network_tensors}
    restricted_tasks = []
    for task in plan.graph.tasks:
        task_input_tensor_ids = tuple(
            slice_task.partial_output_tensor_id
            if tensor_id == source_task.output_tensor_id
            else tensor_id
            for tensor_id in task.input_tensor_ids
        )
        task_dependencies = tuple(
            slice_task.id if dependency == source_task.id else dependency
            for dependency in task.dependencies
        )
        try:
            input_shapes = tuple(
                shape_by_tensor[tensor_id] for tensor_id in task_input_tensor_ids
            )
        except KeyError as exc:
            raise ValueError("sliced_resident_restricted_graph_missing_input_shape") from exc
        output_shape, gemm_m, gemm_k, gemm_n = _restricted_task_geometry(
            task, input_shapes
        )
        output_tensor_id = task.output_tensor_id
        task_id = task.id
        if task.id == source_task.id:
            task_id = slice_task.id
            output_tensor_id = slice_task.partial_output_tensor_id
        restricted_task = replace(
            task,
            id=task_id,
            input_tensor_ids=task_input_tensor_ids,
            dependencies=task_dependencies,
            output_tensor_id=output_tensor_id,
            input_shapes=input_shapes,
            output_shape=output_shape,
            gemm_m=gemm_m,
            gemm_k=gemm_k,
            gemm_n=gemm_n,
            estimated_flops=2 * gemm_m * gemm_k * gemm_n,
            estimated_bytes=sum(int(np.prod(shape)) for shape in input_shapes)
            + int(np.prod(output_shape)),
        )
        restricted_tasks.append(restricted_task)
        shape_by_tensor[output_tensor_id] = output_shape
    graph = with_execution_identity(
        replace(
            plan.graph,
            network=network_spec,
            tasks=tuple(restricted_tasks),
            circuit_semantics_hash="",
            tensor_network_hash="",
            contraction_plan_hash="",
        )
    )
    return graph, TensorNetworkValue(network_spec, network_tensors), hashes, fnv_hashes


def _restricted_task_geometry(
    task: Any,
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[tuple[int, ...], int, int, int]:
    dimensions: dict[int, int] = {}
    for labels, shape in zip(
        (task.left_labels, task.right_labels), input_shapes, strict=True
    ):
        if len(labels) != len(shape):
            raise ValueError("sliced_resident_restricted_input_rank_mismatch")
        for label, dimension in zip(labels, shape, strict=True):
            previous = dimensions.setdefault(int(label), int(dimension))
            if previous != int(dimension):
                raise ValueError("sliced_resident_restricted_label_dimension_mismatch")
    try:
        output_shape = tuple(dimensions[int(label)] for label in task.output_labels)
        gemm_m = int(np.prod([dimensions[int(label)] for label in task.left_labels if label not in task.contracted_labels], dtype=np.int64))
        gemm_k = int(np.prod([dimensions[int(label)] for label in task.contracted_labels], dtype=np.int64))
        gemm_n = int(np.prod([dimensions[int(label)] for label in task.right_labels if label not in task.contracted_labels], dtype=np.int64))
    except KeyError as exc:
        raise ValueError("sliced_resident_restricted_task_label_missing") from exc
    return output_shape, gemm_m, gemm_k, gemm_n


def _slice_execution_manifest(
    item: SlicedResidentGraphPackage,
    package: ResidentGraphPackage,
    *,
    restricted_input_sha256: Mapping[str, str],
    restricted_input_fnv1a64: Mapping[str, str],
    resident_descriptor_sha256: str,
    resident_descriptor_fnv1a64: str,
) -> JsonDict:
    if package.manifest_path is None or package.descriptor_sha256 is None:
        raise ValueError("sliced_resident_package_write_incomplete")
    manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    slice_task = item.slice_plan.slice_task
    source_hashes = dict(item.source_hashes)
    manifest["slice_execution"] = {
        "schema_version": SLICED_RESIDENT_EXECUTION_SCHEMA_VERSION,
        "version": 1,
        "slice_id": item.slice_id,
        "dpu_id": item.dpu_id,
        "source_task_id": slice_task.source_task_id,
        "sliced_label": slice_task.assignment.label,
        "assignment_value": slice_task.assignment.value,
        "restrictions": [item.to_json() for item in item.restrictions],
        "source_hashes": source_hashes,
        "source_circuit_semantics_hash": source_hashes["circuit_semantics_hash"],
        "source_tensor_network_hash": source_hashes["tensor_network_hash"],
        "source_contraction_plan_hash": source_hashes["contraction_plan_hash"],
        "restricted_input_sha256": dict(restricted_input_sha256),
        "restricted_input_fnv1a64": dict(restricted_input_fnv1a64),
        "resident_descriptor_sha256": resident_descriptor_sha256,
        "resident_descriptor_fnv1a64": resident_descriptor_fnv1a64,
        "reconstruction_contract": "python_sum_partials",
        "source_task_count": plan_source_task_count(item),
        "expanded_task_count": int(package.component_operation_count),
        "outer_execution_identity": {
            "route_id": SLICED_RESIDENT_OUTER_ROUTE_ID,
            "backend_id": SLICED_RESIDENT_OUTER_BACKEND_ID,
            "hardware_profile_version": SLICED_RESIDENT_OUTER_PROFILE_VERSION,
            "timing_scope": SLICED_RESIDENT_OUTER_TIMING_SCOPE,
        },
        "nested_package_execution_identity": {
            "route_id": "upmem_tn_hardware_taskgraph_resident",
            "backend_id": "upmem_sdk_hardware_taskgraph_resident",
            "profile_version": "hardware_taskgraph_single_dpu_mram_resident_v1",
        },
    }
    return manifest


def plan_source_task_count(item: SlicedResidentGraphPackage) -> int:
    """Return the source graph count without duplicating it in package metadata."""

    return int(item.package.graph.path_summary.task_count)


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(_float32_bytes(value))


def _array_fnv1a64(value: np.ndarray) -> str:
    return _fnv1a64_bytes(_float32_bytes(value))


def _validate_m2_source_inputs(plan: SlicedResidentPlan) -> None:
    initial_tensors = [tensor for tensor in plan.network.tensors if tensor.spec.produced_by is None]
    if not initial_tensors:
        raise ValueError("sliced_resident_missing_source_input")
    for tensor in initial_tensors:
        tensor_id = tensor.spec.id
        if _has_nonzero_imaginary(tensor.array):
            raise ValueError(
                f"sliced_resident_m2_nonzero_imaginary_source_input:{tensor_id}"
            )


def _m2_real_float32(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if _has_nonzero_imaginary(array):
        raise ValueError("sliced_resident_m2_nonzero_imaginary_source_input")
    return np.asarray(array.real if np.iscomplexobj(array) else array, dtype=np.float32)


def _has_nonzero_imaginary(value: Any) -> bool:
    array = np.asarray(value)
    return bool(np.iscomplexobj(array) and np.any(array.imag != 0.0))


def _float32_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(value, dtype="<f4")).tobytes(order="C")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fnv1a64_bytes(value: bytes) -> str:
    """Match the byte-at-a-time FNV-1a loop used by the two-DPU C host."""

    result = 14695981039346656037
    for byte in value:
        result ^= byte
        result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def _file_fingerprints(path: Path) -> JsonDict:
    if not path.is_file():
        raise ValueError("sliced_resident_provenance_file_missing")
    raw = path.read_bytes()
    return {"sha256": _sha256_bytes(raw), "fnv1a64": _fnv1a64_bytes(raw)}


def _load_json_object(path: Path, error: str) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(value, dict):
        raise ValueError(error)
    return value


def _manifest_file_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"sliced_resident_manifest_{field}_invalid")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"sliced_resident_manifest_{field}_must_be_relative")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"sliced_resident_manifest_{field}_outside_session") from exc
    return resolved


def _written_input_fingerprints(
    package: ResidentGraphPackage,
    root: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    source = manifest or _load_json_object(
        package.manifest_path, "sliced_resident_manifest_invalid"
    )
    entries = source.get("initial_slots")
    if not isinstance(entries, list) or len(entries) != len(package.initial_data):
        raise ValueError("sliced_resident_manifest_restricted_inputs_invalid")
    expected_slots = set(package.initial_data)
    paths: set[str] = set()
    sha256: dict[str, str] = {}
    fnv1a64: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("sliced_resident_manifest_restricted_inputs_invalid")
        slot_id = entry.get("slot_id")
        path_text = entry.get("input_path")
        if (
            not isinstance(slot_id, int)
            or slot_id not in expected_slots
            or not isinstance(path_text, str)
            or path_text in paths
            or entry.get("raw_bytes")
            != int(np.asarray(package.initial_data[slot_id]).size) * 4
        ):
            raise ValueError("sliced_resident_manifest_restricted_inputs_invalid")
        paths.add(path_text)
        fingerprints = _file_fingerprints(
            _manifest_file_path(root, path_text, "input_path")
        )
        sha256[path_text] = fingerprints["sha256"]
        fnv1a64[path_text] = fingerprints["fnv1a64"]
    if len(paths) != len(expected_slots):
        raise ValueError("sliced_resident_manifest_restricted_inputs_invalid")
    return {"sha256": sha256, "fnv1a64": fnv1a64}


def _validate_written_package_assignments(
    plan: SlicedResidentPlan, packages: tuple[SlicedResidentGraphPackage, ...]
) -> None:
    if len(packages) != 2 or tuple(item.slice_id for item in packages) != (0, 1):
        raise ValueError("sliced_resident_packages_must_have_slice_ids_0_and_1")
    if (
        tuple(item.dpu_id for item in packages) != (0, 1)
        or len({item.dpu_id for item in packages}) != 2
    ):
        raise ValueError("sliced_resident_packages_must_have_distinct_dpu_ids")
    if tuple(item.slice_plan for item in packages) != plan.slice_plans:
        raise ValueError("sliced_resident_written_slice_plan_mismatch")
    expected_source_hashes = {
        "circuit_semantics_hash": plan.circuit_semantics_hash,
        "tensor_network_hash": plan.tensor_network_hash,
        "contraction_plan_hash": plan.contraction_plan_hash,
    }
    if any(
        item.restrictions != item.slice_plan.slice_task.input_restrictions
        or dict(item.source_hashes) != expected_source_hashes
        for item in packages
    ):
        raise ValueError("sliced_resident_written_slice_metadata_mismatch")
    if (
        len({item.package.manifest_path for item in packages}) != 2
        or len({item.package.package_path for item in packages}) != 2
    ):
        raise ValueError("sliced_resident_written_paths_must_be_distinct")
    outputs = [
        path for item in packages for path in item.package.final_output_paths.values()
    ]
    if len(outputs) != 2 or len(set(outputs)) != 2:
        raise ValueError("sliced_resident_output_paths_must_be_distinct")


def _validate_slice_execution_binding(
    plan: SlicedResidentPlan,
    item: SlicedResidentGraphPackage,
    binding: Mapping[str, Any],
    *,
    restricted_input_sha256: Mapping[str, str],
    restricted_input_fnv1a64: Mapping[str, str],
    resident_descriptor_sha256: str,
    resident_descriptor_fnv1a64: str,
) -> None:
    task = item.slice_plan.slice_task
    source_hashes = {
        "circuit_semantics_hash": plan.circuit_semantics_hash,
        "tensor_network_hash": plan.tensor_network_hash,
        "contraction_plan_hash": plan.contraction_plan_hash,
    }
    expected = {
        "schema_version": SLICED_RESIDENT_EXECUTION_SCHEMA_VERSION,
        "version": 1,
        "slice_id": item.slice_id,
        "dpu_id": item.dpu_id,
        "source_task_id": task.source_task_id,
        "sliced_label": task.assignment.label,
        "assignment_value": task.assignment.value,
        "restrictions": to_jsonable(list(item.restrictions)),
        "source_hashes": source_hashes,
        "source_circuit_semantics_hash": source_hashes["circuit_semantics_hash"],
        "source_tensor_network_hash": source_hashes["tensor_network_hash"],
        "source_contraction_plan_hash": source_hashes["contraction_plan_hash"],
        "restricted_input_sha256": dict(restricted_input_sha256),
        "restricted_input_fnv1a64": dict(restricted_input_fnv1a64),
        "resident_descriptor_sha256": resident_descriptor_sha256,
        "resident_descriptor_fnv1a64": resident_descriptor_fnv1a64,
        "reconstruction_contract": "python_sum_partials",
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("sliced_resident_manifest_slice_execution_mismatch")


def _single_final_output_path(package: ResidentGraphPackage) -> Path:
    if len(package.final_output_paths) != 1 or set(package.final_output_paths) != {
        "real"
    }:
        raise ValueError("sliced_resident_requires_one_real_partial_output")
    path = package.final_output_paths["real"]
    if not path.is_file():
        raise ValueError("sliced_resident_native_partial_output_missing")
    return path


def _load_native_response(
    native_response: Mapping[str, Any] | Path,
) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(native_response, Path):
        return _load_json_object(
            native_response, "sliced_resident_native_response_invalid"
        ), native_response
    if not isinstance(native_response, Mapping):
        raise ValueError("sliced_resident_native_response_invalid")
    return native_response, None


def _validate_native_two_slice_response(
    response: Mapping[str, Any],
    packages: tuple[SlicedResidentGraphPackage, ...],
    response_path: Path | None,
) -> None:
    if (
        response.get("schema_version")
        != "generic_loop_resident_two_dpu_contraction_slice_v1"
        or response.get("manifest_kind") != "resident_two_slice_response"
        or response.get("status") != "completed"
        or response.get("failure_stage") is not None
        or response.get("cpu_fallback_used") is not False
        or response.get("topology") != "two_dpu_allocation"
        or response.get("hardware_execution") is not True
        or response.get("native_reconstruction_performed") is not False
        or response.get("reconstruction_contract") != "python_sum_partials"
    ):
        raise ValueError("sliced_resident_native_response_status_invalid")
    allocation = response.get("allocation")
    launch = response.get("launch")
    release = response.get("release")
    if (
        not isinstance(allocation, Mapping)
        or allocation.get("requested_dpus") != 2
        or allocation.get("allocated_dpus") != 2
        or allocation.get("profile") != "backend=hw"
        or allocation.get("verified") is not True
        or not isinstance(launch, Mapping)
        or launch.get("mode") != "asynchronous"
        or launch.get("completed") is not True
        or not isinstance(release, Mapping)
        or release.get("attempted") is not True
        or release.get("confirmed") is not True
    ):
        raise ValueError("sliced_resident_native_hardware_evidence_invalid")
    entries = response.get("slices")
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("sliced_resident_native_slice_count_invalid")
    by_id: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("slice_id"), int):
            raise ValueError("sliced_resident_native_slice_invalid")
        by_id[entry["slice_id"]] = entry
    if set(by_id) != {0, 1}:
        raise ValueError("sliced_resident_native_slice_assignments_invalid")
    operation_count = int(packages[0].package.component_operation_count)
    if operation_count < 1 or any(
        int(item.package.component_operation_count) != operation_count
        for item in packages
    ):
        raise ValueError("sliced_resident_native_operation_count_invalid")
    if (
        launch.get("async_launch_count") != operation_count
        or launch.get("synchronize_count") != operation_count
        or launch.get("operation_count", 1) != operation_count
    ):
        raise ValueError("sliced_resident_native_operation_execution_invalid")
    observed_counts = response.get("observed_operation_completion_counts")
    if operation_count > 1 and observed_counts != [operation_count, operation_count]:
        raise ValueError("sliced_resident_native_observed_operation_count_invalid")
    if operation_count > 1 and response.get("completion_sentinel_read_counts") != [
        operation_count,
        operation_count,
    ]:
        raise ValueError("sliced_resident_native_completion_sentinel_read_count_invalid")
    if operation_count > 1 and (
        response.get("native_execution_sentinel_available") is not True
        or response.get("completion_evidence")
        != "dpu_written_completion_sentinel_read_after_each_sync"
        or response.get("device_completion_confirmed") is not True
    ):
        raise ValueError("sliced_resident_native_completion_sentinel_invalid")
    if operation_count > 1:
        observed_bytes = tuple(response.get(name) for name in (
            "actual_h2d_bytes", "actual_d2h_bytes", "actual_transfer_bytes"
        ))
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in observed_bytes
        ) or observed_bytes[2] != observed_bytes[0] + observed_bytes[1]:
            raise ValueError("sliced_resident_native_transfer_accounting_invalid")
        planned_h2d = planned_d2h = 0
        for item in packages:
            payload = json.loads(item.package.manifest_path.read_text(encoding="utf-8"))
            planned_h2d += sum(
                int(payload.get(key, 0))
                for key in ("initial_h2d_bytes", "descriptor_h2d_bytes", "control_h2d_bytes")
            )
            planned_d2h += int(payload.get("final_d2h_bytes", 0))
        if observed_bytes != (planned_h2d, planned_d2h, planned_h2d + planned_d2h):
            raise ValueError("sliced_resident_native_transfer_plan_mismatch")
    for item in packages:
        entry = by_id[item.slice_id]
        package = item.package
        if package.manifest_path is None:
            raise ValueError("sliced_resident_package_write_incomplete")
        expected_bytes = int(np.prod(item.slice_plan.slice_task.output_shape)) * 4
        expected_transfer_bytes = (expected_bytes + 7) & ~7
        if (
            entry.get("dpu_index") != item.dpu_id
            or entry.get("allocated") is not True
            or entry.get("release_confirmed") is not True
            or entry.get("package_transferred") is not True
            or entry.get("input_count") != len(package.initial_data)
            or entry.get("operation_count", 1) != operation_count
            or entry.get("completion_confirmed") is not True
            or entry.get("operation_completion_confirmed", entry.get("completion_confirmed")) is not True
            or entry.get("inputs_transferred") is not True
            or entry.get("partial_output_elements")
            != int(np.prod(item.slice_plan.slice_task.output_shape))
            or entry.get("partial_output_raw_bytes", entry.get("partial_output_bytes")) != expected_bytes
            or entry.get("partial_output_transfer_bytes", entry.get("partial_output_bytes")) != expected_transfer_bytes
            or entry.get("partial_output_read") is not True
            or entry.get("partial_output_written") is not True
            or not _response_path_matches(
                entry.get("manifest_path"), package.manifest_path, response_path
            )
            or entry.get("manifest_fnv1a64")
            != _file_fingerprints(package.manifest_path)["fnv1a64"]
        ):
            raise ValueError("sliced_resident_native_slice_evidence_invalid")
        if operation_count > 1:
            sentinel = entry.get("dpu_completion_sentinel")
            if not isinstance(sentinel, Mapping) or sentinel.get("verified") is not True:
                raise ValueError("sliced_resident_native_slice_completion_sentinel_invalid")
            if entry.get("completion_sentinel_read_count") != operation_count:
                raise ValueError("sliced_resident_native_slice_completion_sentinel_invalid")
            if (
                sentinel.get("active_operation_index") != operation_count - 1
                or sentinel.get("completion_status") != 1
                or sentinel.get("completed_operation_count") != operation_count
                or sentinel.get("output_elements_processed")
                != int(np.prod(item.slice_plan.slice_task.output_shape))
            ):
                raise ValueError("sliced_resident_native_slice_completion_sentinel_invalid")
        observed_count = entry.get("observed_operation_completion_count")
        if operation_count > 1 and observed_count != operation_count:
            raise ValueError("sliced_resident_native_observed_slice_count_invalid")


def _response_path_matches(
    value: Any, expected: Path, response_path: Path | None
) -> bool:
    if not isinstance(value, str) or not value:
        return False
    candidate = Path(value)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(expected.parent / candidate)
        if response_path is not None:
            candidates.append(response_path.parent / candidate)
    return any(path.resolve() == expected.resolve() for path in candidates)


def _response_fingerprint(
    response: Mapping[str, Any], response_path: Path | None, algorithm: str
) -> str:
    if response_path is not None:
        return _file_fingerprints(response_path)[algorithm]
    raw = json.dumps(
        to_jsonable(response), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(raw) if algorithm == "sha256" else _fnv1a64_bytes(raw)


def _validate_model_binding(graph: TaskGraph, model: SliceAwareTaskGraphModel) -> None:
    if not graph.tasks:
        raise ValueError("sliced_resident_requires_task_graph")
    source = next(
        (task for task in graph.tasks if task.id == model.sliced_task_id), None
    )
    if source is None or source is not graph.tasks[-1]:
        raise ValueError("sliced_resident_selected_task_must_be_terminal")
    if model.sliced_task_id != source.id or model.source_task_count != len(graph.tasks):
        raise ValueError("sliced_resident_model_source_task_mismatch")
    if model.sliced_indices != (model.slice_tasks[0].assignment.label,):
        raise ValueError("sliced_resident_model_sliced_index_mismatch")
    label = model.sliced_indices[0]
    if label not in source.contracted_labels:
        raise ValueError("sliced_resident_model_sliced_index_mismatch")
    if model.downstream_dependency_rewrites:
        raise ValueError("sliced_resident_model_has_downstream_rewrites")

    for slice_task in model.slice_tasks:
        slice_id = slice_task.slice_id
        if slice_task.source_task_id != source.id:
            raise ValueError("sliced_resident_model_source_task_mismatch")
        if slice_task.id != f"{source.id}__slice_{slice_id}":
            raise ValueError("sliced_resident_model_slice_id_mismatch")
        if (
            slice_task.assignment.label != label
            or slice_task.assignment.value != slice_id
        ):
            raise ValueError("sliced_resident_assignment_values_must_match_slice_ids")
        if slice_task.dependencies != source.dependencies:
            if not source.dependencies and slice_task.dependencies:
                raise ValueError("sliced_resident_dependent_selected_task")
            raise ValueError("sliced_resident_model_dependencies_mismatch")
        if slice_task.input_tensor_ids != source.input_tensor_ids:
            raise ValueError("sliced_resident_model_input_binding_mismatch")
        if (
            slice_task.partial_output_tensor_id
            != f"{source.output_tensor_id}__slice_{slice_id}"
        ):
            raise ValueError("sliced_resident_model_output_binding_mismatch")
        if (
            slice_task.output_shape != source.output_shape
            or slice_task.output_labels != source.output_labels
        ):
            raise ValueError("sliced_resident_model_output_binding_mismatch")
        tensors = {tensor.id: tensor for tensor in graph.network.tensors}
        if model.slice_model_kind == "dependent_prefix_replicated":
            if len(graph.tasks) != 2 or not source.dependencies:
                raise ValueError("sliced_resident_dependent_prefix_contract_invalid")
            if len(slice_task.input_restrictions) != 2:
                raise ValueError("sliced_resident_model_restrictions_mismatch")
            source_tensor_ids = {tensor.id for tensor in graph.network.tensors}
            intermediate_tensor_ids = {
                task.output_tensor_id for task in graph.tasks
            }
            for restriction in slice_task.input_restrictions:
                tensor = tensors.get(restriction.tensor_id)
                if (
                    tensor is None
                    or restriction.tensor_id not in source_tensor_ids
                    or restriction.tensor_id in intermediate_tensor_ids
                    or restriction.label != label
                    or restriction.axis >= len(tensor.labels)
                    or tensor.labels[restriction.axis] != label
                    or restriction.value != slice_id
                ):
                    raise ValueError("sliced_resident_model_restrictions_mismatch")
        else:
            expected_restrictions = tuple(
                SliceInputRestriction(
                    tensor_id=tensor_id,
                    label=label,
                    axis=labels.index(label),
                    value=slice_id,
                )
                for tensor_id, labels in zip(
                    source.input_tensor_ids,
                    (source.left_labels, source.right_labels),
                    strict=True,
                )
                if label in labels
            )
            if slice_task.input_restrictions != expected_restrictions:
                raise ValueError("sliced_resident_model_restrictions_mismatch")

    reconstruction = model.reconstruction_step
    if (
        reconstruction.source_task_id != source.id
        or reconstruction.id != f"{source.id}__slice_reconstruct"
        or reconstruction.output_tensor_id != source.output_tensor_id
        or reconstruction.operation != "sum_partials"
    ):
        raise ValueError("sliced_resident_model_reconstruction_output_mismatch")


def _build_execution_plan(graph: TaskGraph) -> UpmemExecutionPlan:
    return UpmemExecutionPlan.for_task_graph(
        graph,
        kernel=UpmemKernelPlan(
            provider_id="upmem_resident_hardware",
            kernel_id="generic_loop_resident_graph",
            kernel_version="generic_loop_resident_graph_v1",
            implementation="explicit_sdk_resident",
            resident=True,
        ),
        placement=UpmemPlacementPlan(
            resources=DpuResourceContext(requested_dpu_count=2),
            assignment_policy="slice_id_to_dpu_id",
            topology="two_dpu_allocation",
        ),
        communication=UpmemCommunicationPlan(
            host_to_dpu="explicit_sdk",
            dpu_to_host="explicit_sdk",
            intermediate_transport="host_partial_output",
            reduction="host",
        ),
        numeric=UpmemNumericPlan(),
        schedule=UpmemSchedulePlan(
            ordering="slice_id",
            dependency_policy="strict",
            parallelism="independent_slices",
            resident_lifetime="slice",
        ),
    )


__all__ = [
    "SLICED_RESIDENT_EXECUTION_SCHEMA_VERSION",
    "SLICED_RESIDENT_PLAN_SCHEMA_VERSION",
    "SlicedResidentGraphPackage",
    "SlicedResidentPlan",
    "SlicedResidentSlicePlan",
    "build_two_slice_resident_graph_packages",
    "build_two_slice_resident_plan",
    "load_and_reconstruct_two_slice_native_outputs",
    "reconstruct_host_slice_outputs",
    "validate_two_slice_resident_plan",
    "validate_written_two_slice_packages",
    "write_two_slice_resident_graph_packages",
]
