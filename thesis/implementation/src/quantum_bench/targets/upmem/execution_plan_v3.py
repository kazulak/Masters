"""Python adapter for the additive one-task UPMEM execution-plan v3 ABI."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import struct
import sys
from typing import Any, Mapping

import numpy as np

from quantum_bench.core.records import (
    CircuitSpec,
    ContractionTask,
    JsonDict,
    PathSummary,
    TaskGraph,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
    to_jsonable,
)
from quantum_bench.targets.upmem import distributed_plan_v3
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_OPERATION_ABI_V2,
    RESIDENT_PACKAGE_HEADER_FORMAT,
    RESIDENT_V3_PROFILE_VERSION,
    _canonical_profile,
    build_resident_graph_package,
    build_resident_policy_reference,
)
from quantum_bench.tn.execution_bundle import canonical_hash, with_execution_identity
from quantum_bench.tn.network import TensorNetworkValue


SCHEMA_VERSION = "upmem_execution_plan_v3"
NATIVE_RESPONSE_SCHEMA = "upmem_execution_plan_native_v3"
MAX_DPUS = 64
MIN_TASKLETS = 1
MAX_TASKLETS = 24
NUMERIC_FLOAT32 = "float32"
NUMERIC_INT8 = "host_packed_int8"
NUMERIC_INT8_REQUANTIZE = "per_task_resident_requantize"
PARTITION_OUTPUT = "output"
PARTITION_CONTRACTED = "contracted"
TRANSPORT_FLOAT32_MRAM = "float32_mram"
TRANSPORT_PACKED_INT8_MRAM = "host_packed_int8_mram"
TIMING_SCOPE = "one_task_resident_v3_full_execution"
DEFAULT_TIMEOUT_S = 900.0
NATIVE_V3_MANIFEST_PROFILE_VERSION = RESIDENT_V3_PROFILE_VERSION


def _native_runner() -> Any:
    path = Path(__file__).resolve().parents[4] / "native" / "upmem" / "simplepim" / "upmem_sdk_execution_plan_runner.py"
    spec = importlib.util.spec_from_file_location("quantum_bench_upmem_execution_plan_v3_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("native_v3_runner_unavailable: cannot load the v3 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(build_dir: Path, *, tasklets: int, environment: Mapping[str, str] | None = None) -> Mapping[str, Any]:
    """Build tasklet-keyed v3 binaries through the native runner file."""

    _validate_resources(1, tasklets)
    result = _native_runner().build(
        Path(build_dir), tasklets_per_dpu=tasklets, environment=environment
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("native_v3_build_failed: runner returned a non-mapping")
    host = result.get("host_binary") or result.get("runner")
    dpu = result.get("dpu_binary")
    initialization = result.get("initialization_binary")
    if not host or not dpu or not initialization:
        raise RuntimeError(
            "native_v3_build_failed: build metadata lacks host_binary/dpu_binary/initialization_binary"
        )
    if Path(str(host)).resolve().parent != Path(str(initialization)).resolve().parent:
        raise RuntimeError("native_v3_build_failed: initialization binary is not beside host_binary")
    return {
        **dict(result),
        "host_binary": str(host),
        "dpu_binary": str(dpu),
        "initialization_binary": str(initialization),
        "host_binary_sha256": _sha256_file(host),
        "dpu_binary_sha256": _sha256_file(dpu),
        "initialization_binary_sha256": _sha256_file(initialization),
        "selected_rank_path": (environment or {}).get("UPMEM_HW_RANK_PATH"),
        "tasklets_per_dpu": tasklets,
        "max_dpus": MAX_DPUS,
        "max_elements": 65536,
        "mram_pool_bytes": 512 * 1024,
        "output_tile_elements": 2,
    }


def prepare_request(
    *,
    case: Mapping[str, Any],
    materialized: Mapping[str, Any],
    dpu_count: int,
    tasklets: int,
    quantization_mode: str,
    partition_strategy: str,
    build: Mapping[str, Any],
    root: Path,
) -> Mapping[str, Any]:
    """Lower one retained task or deterministic matrix into a v3 request."""

    _validate_resources(dpu_count, tasklets)
    if quantization_mode not in {"none", NUMERIC_INT8, NUMERIC_INT8_REQUANTIZE}:
        raise ValueError("hardware_profile_violation: unsupported numeric mode")
    if quantization_mode == NUMERIC_INT8_REQUANTIZE and dpu_count != 1:
        raise ValueError(
            "hardware_profile_violation: legacy DPU requantization is a one-DPU diagnostic"
        )
    if partition_strategy not in {PARTITION_OUTPUT, PARTITION_CONTRACTED}:
        raise ValueError("hardware_profile_violation: unsupported partition strategy")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if case.get("non_quantum") is True or case.get("quantum_case") == "non_quantum":
        graph, network, evidence = _synthetic_package(case, dpu_count)
    else:
        graph, network, evidence = _real_package(materialized)

    profile = _canonical_profile(
        tasklets,
        version=RESIDENT_V3_PROFILE_VERSION,
        requested_dpu_count=dpu_count,
    )
    policy_reference = build_resident_policy_reference(
        graph, network, quantization_mode=quantization_mode, profile=profile
    )
    full_precision_reference = build_resident_policy_reference(
        graph, network, quantization_mode="none", profile=profile
    )
    policy_reference = {
        **_real_float32_reference(policy_reference),
        "reference_kind": "cpu_active_numeric_policy_reference",
    }
    full_precision_reference = {
        **_real_float32_reference(full_precision_reference),
        "reference_kind": "cpu_full_precision_float32_reference",
    }
    package = build_resident_graph_package(
        graph,
        network,
        case_id=str(case.get("case_id", graph.network.circuit.name)),
        suite_id=str(case.get("suite_id", "upmem_hardware_distributed_m5")),
        quantization_mode=quantization_mode,
        profile=profile,
        full_precision_output=np.asarray(full_precision_reference["output"]),
        operation_abi_version=RESIDENT_OPERATION_ABI_V2,
    )
    staged_dpu = _stage_dpu_binary(build.get("dpu_binary"), root)
    initialization_binary = _required_binary_binding(
        build, "initialization_binary", "initialization_binary_sha256", "SimplePIM initialization binary"
    )
    host_binary = Path(str(build.get("host_binary") or build.get("runner"))).resolve()
    if host_binary.parent != initialization_binary.resolve().parent:
        raise ValueError("native_v3_build_invalid: initialization binary is not beside host_binary")
    request_id = _request_id(case, dpu_count, tasklets, quantization_mode, partition_strategy)
    package = package.write(root, dpu_binary=staged_dpu, request_id=request_id)
    _write_native_v3_manifest_identity(package.manifest_path)

    policy_reference_path = root / "policy_reference_f32.bin"
    np.asarray(policy_reference["output"], dtype="<f4").ravel().tofile(policy_reference_path)
    integer_reference_path: Path | None = None
    if quantization_mode == NUMERIC_INT8:
        integer_reference_path = root / "integer_reference_i32.bin"
        np.asarray(policy_reference["raw_output"], dtype="<i4").ravel().tofile(
            integer_reference_path
        )
    full_precision_reference_path = root / "full_precision_reference_f32.bin"
    np.asarray(full_precision_reference["output"], dtype="<f4").ravel().tofile(
        full_precision_reference_path
    )
    output_path = package.final_output_paths["real"]
    raw_output_path = package.raw_final_output_paths.get("real")
    package_bytes = package.package_path.read_bytes() if package.package_path else b""
    operation_bytes = _operation_bytes(package_bytes)
    operation = package.operations[0]
    numeric_mode = NUMERIC_FLOAT32 if quantization_mode == "none" else quantization_mode
    if partition_strategy == PARTITION_OUTPUT:
        plan = distributed_plan_v3.build_output_tile_plan_v3(
            logical_operation_id=operation.task_id,
            logical_task_id=operation.task_id,
            total_output_elements=operation.output_elements,
            total_contracted_elements=int(operation.args["contracted_combination_count"]),
            package_sha256=hashlib.sha256(package_bytes).hexdigest(),
            operation_sha256=hashlib.sha256(operation_bytes).hexdigest(),
            output_slot=operation.slot_out_real,
            dpu_count=dpu_count,
            tasklets_per_dpu=tasklets,
            numeric_mode=numeric_mode,
        )
    else:
        plan = distributed_plan_v3.build_contracted_partial_sum_plan_v3(
            logical_operation_id=operation.task_id,
            logical_task_id=operation.task_id,
            total_output_elements=operation.output_elements,
            total_contracted_elements=int(operation.args["contracted_combination_count"]),
            package_sha256=hashlib.sha256(package_bytes).hexdigest(),
            operation_sha256=hashlib.sha256(operation_bytes).hexdigest(),
            output_slot=operation.slot_out_real,
            dpu_count=dpu_count,
            tasklets_per_dpu=tasklets,
            numeric_mode=numeric_mode,
        )
    sidecar_path = root / "distributed_plan_v3.bin"
    sidecar_path.write_bytes(
        distributed_plan_v3.serialize_upxdpv3(
            plan, package_bytes=package_bytes, operation_bytes=operation_bytes
        )
    )
    sidecar_validation = distributed_plan_v3.validate_upxdpv3(
        sidecar_path.read_bytes(), plan, package_bytes=package_bytes, operation_bytes=operation_bytes
    )
    plan_json_path = root / "distributed_plan_v3.json"
    plan_json_path.write_text(json.dumps(plan.to_json_dict(), sort_keys=True, indent=2), encoding="utf-8")
    response_path = root / "response.json"
    evidence = {
        **evidence,
        "package_circuit_semantics_hash": graph.circuit_semantics_hash,
        "package_tensor_network_hash": graph.tensor_network_hash,
        "package_contraction_plan_hash": graph.contraction_plan_hash,
        "package_sha256": hashlib.sha256(package_bytes).hexdigest(),
        "operation_sha256": hashlib.sha256(operation_bytes).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        "task_id": operation.task_id,
        "task_hash": evidence.get("task_hash") or canonical_hash(to_jsonable(graph.tasks[0])),
        "host_binary_hash": _sha256_file(build.get("host_binary") or build.get("runner")),
        "dpu_binary_hash": _sha256_file(staged_dpu),
    }
    policy_reference_sha256 = _sha256_file(policy_reference_path)
    integer_reference_sha256 = (
        _sha256_file(integer_reference_path) if integer_reference_path is not None else None
    )
    full_precision_reference_sha256 = _sha256_file(full_precision_reference_path)
    policy_tolerance = _policy_reference_tolerance(quantization_mode)
    full_precision_tolerance = _full_precision_tolerance(quantization_mode)
    return {
        "schema_version": SCHEMA_VERSION,
        "native_response_schema": NATIVE_RESPONSE_SCHEMA,
        "host_binary": str(host_binary),
        "dpu_binary": str(build.get("dpu_binary")),
        "initialization_binary": str(initialization_binary),
        "initialization_binary_sha256": build["initialization_binary_sha256"],
        "resident_manifest": str(package.manifest_path),
        "resident_package": str(package.package_path),
        "distributed_plan": str(sidecar_path),
        "distributed_plan_json": str(plan_json_path),
        "sidecar_path": str(sidecar_path),
        "response_path": str(response_path),
        "output_path": str(output_path),
        "raw_output_path": str(raw_output_path) if raw_output_path is not None else None,
        "policy_reference": {
            "path": str(policy_reference_path),
            "sha256": policy_reference_sha256,
            "max_abs_tolerance": policy_tolerance,
            "reference_kind": "cpu_active_numeric_policy_reference",
        },
        "integer_reference": {
            "path": str(integer_reference_path) if integer_reference_path is not None else None,
            "sha256": integer_reference_sha256,
            "required": quantization_mode == NUMERIC_INT8,
            "reference_kind": "cpu_exact_packed_int8_int32_reference",
        },
        "full_precision_reference": {
            "path": str(full_precision_reference_path),
            "sha256": full_precision_reference_sha256,
            "max_abs_tolerance": full_precision_tolerance,
            "required": quantization_mode == "none",
            "reference_kind": "cpu_full_precision_float32_reference",
        },
        "selected_rank_path": build.get("selected_rank_path") or build.get("upmem_rank_path_requested"),
        "rank_path": build.get("selected_rank_path") or build.get("upmem_rank_path_requested"),
        "upmem_rank_path_requested": build.get("selected_rank_path") or build.get("upmem_rank_path_requested"),
        "dpu_count": dpu_count,
        "requested_dpus": dpu_count,
        "tasklets": tasklets,
        "tasklets_per_dpu": tasklets,
        "quantization_mode": quantization_mode,
        "numeric_mode": numeric_mode,
        "numeric_transport": (
            TRANSPORT_PACKED_INT8_MRAM
            if quantization_mode == NUMERIC_INT8
            else TRANSPORT_FLOAT32_MRAM
        ),
        "transport": (
            TRANSPORT_PACKED_INT8_MRAM
            if quantization_mode == NUMERIC_INT8
            else TRANSPORT_FLOAT32_MRAM
        ),
        "packed_int8_transfer": quantization_mode == NUMERIC_INT8,
        "host_quantization": quantization_mode == NUMERIC_INT8,
        "host_quantization_time_s": package.host_quantization_time_s,
        "dpu_intermediate_requantization": False,
        "partition_strategy": partition_strategy,
        "partition_kind": plan.partition_kind,
        "scaling_kind": _scaling_kind(case),
        "output_elements": int(operation.output_elements),
        "contracted_elements": int(operation.args["contracted_combination_count"]),
        "mac_count": int(
            operation.output_elements
            * int(operation.args["contracted_combination_count"])
        ),
        "timing_scope": TIMING_SCOPE,
        "simplepim_role": "initialization_binary_and_management_state_only",
        "collective_provider": "none" if partition_strategy == PARTITION_OUTPUT else "host_mediated_sum_v1",
        "timeout_s": DEFAULT_TIMEOUT_S,
        "execution_plan_hash": plan.execution_plan_hash,
        "execution_input_hash": evidence["package_sha256"],
        "sidecar_validation": sidecar_validation,
        "policy_reference_metadata": _reference_metadata(policy_reference),
        "full_precision_reference_metadata": _reference_metadata(
            full_precision_reference
        ),
        "non_quantum": bool(case.get("non_quantum") is True or case.get("quantum_case") == "non_quantum"),
        **evidence,
        "host_binary_sha256": evidence["host_binary_hash"],
        "dpu_binary_sha256": evidence["dpu_binary_hash"],
    }


def validate_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the v3 sidecar/package bindings without touching UPMEM."""

    manifest_path = Path(str(request["resident_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_path = manifest_path.parent / str(manifest["package_path"])
    package_bytes = package_path.read_bytes()
    operation_bytes = _operation_bytes(package_bytes)
    sidecar = Path(str(request["distributed_plan"])).read_bytes()
    plan = distributed_plan_v3.load_upxdpv3(sidecar, package_bytes=package_bytes, operation_bytes=operation_bytes)
    return distributed_plan_v3.validate_upxdpv3(
        sidecar, plan, package_bytes=package_bytes, operation_bytes=operation_bytes
    )


def _real_package(materialized: Mapping[str, Any]) -> tuple[TaskGraph, TensorNetworkValue, JsonDict]:
    selection = materialized.get("selection_object")
    source_graph = _first(materialized, "_source_graph", "source_graph", "graph")
    source_task = _first(materialized, "_selected_task", "selected_task", "task")
    if selection is not None:
        source_graph = source_graph or getattr(selection, "identified_graph", None) or getattr(selection, "graph", None)
        source_task = source_task or getattr(selection, "selected_task", None) or getattr(selection, "task", None)
    if source_task is None and source_graph is not None:
        task_id = materialized.get("task_id")
        source_task = next((item for item in source_graph.tasks if item.id == task_id), None)
    if not isinstance(source_graph, TaskGraph) or not isinstance(source_task, ContractionTask):
        raise ValueError("task_selection_invalid: retained real TaskGraph and ContractionTask are required")
    source_graph = with_execution_identity(source_graph)
    left = _first(materialized, "_left_operand", "left_operand")
    right = _first(materialized, "_right_operand", "right_operand")
    if left is None and selection is not None:
        left = getattr(selection, "left_operand", None)
        right = getattr(selection, "right_operand", None)
    if left is None or right is None:
        raise ValueError("task_selection_invalid: materialized real operands are required")
    graph, network = _one_task_package(
        source_graph, source_task, left, right, source_graph.network.circuit.name
    )
    return graph, network, _identity_evidence(materialized, source_graph, source_task)


def _one_task_package(
    source_graph: TaskGraph,
    source_task: ContractionTask,
    left: Any,
    right: Any,
    circuit_name: str,
) -> tuple[TaskGraph, TensorNetworkValue]:
    specs = {item.id: item for item in source_graph.network.tensors}
    left_spec = specs.get(source_task.input_tensor_ids[0])
    right_spec = specs.get(source_task.input_tensor_ids[1])
    if left_spec is None or right_spec is None:
        raise ValueError("task_selection_invalid: selected task inputs are absent from its network")
    package_left = TensorSpec(left_spec.id, left_spec.labels, left_spec.shape, left_spec.structure, dtype="float32")
    package_right = TensorSpec(right_spec.id, right_spec.labels, right_spec.shape, right_spec.structure, dtype="float32")
    network_spec = TensorNetworkSpec(
        circuit=source_graph.network.circuit,
        tensors=(package_left, package_right),
        output_labels=source_task.output_labels,
        einsum_expression=source_task.index_expression,
    )
    task = replace(source_task, dependencies=())
    graph = TaskGraph(
        network=network_spec,
        tasks=(task,),
        path=((0, 1),),
        path_summary=replace(source_graph.path_summary, path_length=1, task_count=1),
        planning_time_s=0.0,
    )
    network = TensorNetworkValue(
        network_spec,
        [TensorValue(package_left, _real_operand(left, source_task.input_shapes[0])), TensorValue(package_right, _real_operand(right, source_task.input_shapes[1]))],
    )
    return with_execution_identity(graph), network


def _synthetic_package(case: Mapping[str, Any], dpu_count: int) -> tuple[TaskGraph, TensorNetworkValue, JsonDict]:
    raw_shapes = case.get("matrix_shapes")
    if not isinstance(raw_shapes, (list, tuple)) or len(raw_shapes) != 2:
        raise ValueError("synthetic_case_invalid: matrix_shapes must contain two matrices")
    left_shape = _shape(raw_shapes[0], dpu_count)
    right_shape = _shape(raw_shapes[1], dpu_count)
    if len(left_shape) != 2 or len(right_shape) != 2 or left_shape[1] != right_shape[0]:
        raise ValueError("synthetic_case_invalid: matrix shapes are not contractible")
    output_shape = (left_shape[0], right_shape[1])
    circuit = CircuitSpec(str(case.get("case_id", "synthetic_m5")), 0, (), {"non_quantum": True, "case": dict(case)})
    left_spec = TensorSpec("synthetic_left", (0, 1), left_shape, "dense", dtype="float32")
    right_spec = TensorSpec("synthetic_right", (1, 2), right_shape, "dense", dtype="float32")
    network_spec = TensorNetworkSpec(circuit, (left_spec, right_spec), (0, 2), "ab,bc->ac")
    task = ContractionTask(
        "synthetic_task_0", (left_spec.id, right_spec.id), "synthetic_output", (),
        "ab,bc->ac", (left_shape, right_shape), output_shape, (0, 1), (1, 2), (1,), (0, 2),
        left_shape[0], left_shape[1], right_shape[1], "dense",
        int(2 * np.prod(output_shape) * left_shape[1]),
        int((np.prod(left_shape) + np.prod(right_shape) + np.prod(output_shape)) * 4),
    )
    graph = TaskGraph(network_spec, (task,), ((0, 1),), PathSummary("synthetic", "none", 1, None, None, None, "synthetic one matrix contraction"), 0.0)
    network = TensorNetworkValue(network_spec, [
        TensorValue(left_spec, _deterministic_matrix(left_shape, 11)),
        TensorValue(right_spec, _deterministic_matrix(right_shape, 23)),
    ])
    graph = with_execution_identity(graph)
    return graph, network, {
        "circuit_semantics_hash": f"non_quantum:{canonical_hash(dict(case))}",
        "tensor_network_hash": f"non_quantum:{canonical_hash(dict(case))}",
        "contraction_plan_hash": f"non_quantum:{canonical_hash(dict(case))}",
        "contraction_path_structure_hash": f"non_quantum:{canonical_hash(dict(case))}",
        "task_hash": canonical_hash(to_jsonable(task)),
    }


def _identity_evidence(materialized: Mapping[str, Any], graph: TaskGraph, task: ContractionTask) -> JsonDict:
    selection = materialized.get("selection_object")
    return {
        "circuit_semantics_hash": materialized.get("circuit_semantics_hash") or graph.circuit_semantics_hash,
        "tensor_network_hash": materialized.get("tensor_network_hash") or graph.tensor_network_hash,
        "contraction_plan_hash": materialized.get("contraction_plan_hash") or graph.contraction_plan_hash,
        "contraction_path_structure_hash": materialized.get("contraction_path_structure_hash") or materialized.get("path_hash"),
        "task_hash": materialized.get("task_hash") or materialized.get("selected_task_hash") or getattr(selection, "task_hash", None) or canonical_hash(to_jsonable(task)),
    }


def _real_operand(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if np.any(array.imag != 0):
            raise ValueError("hardware_profile_violation: v3 real route received nonzero imaginary input")
        array = array.real
    if tuple(array.shape) != tuple(shape) or not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError("hardware_profile_violation: invalid materialized real operand")
    return np.ascontiguousarray(array, dtype=np.float32)


def _shape(value: Any, dpu_count: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("synthetic_case_invalid: matrix shape must be a sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            result.append(item)
            continue
        text = str(item).replace(" ", "")
        if text == "4*dpu_count":
            result.append(4 * dpu_count)
        elif text.isdigit():
            result.append(int(text))
        else:
            raise ValueError(f"synthetic_case_invalid: unsupported dimension {item!r}")
    if any(item <= 0 for item in result):
        raise ValueError("synthetic_case_invalid: dimensions must be positive")
    return tuple(result)


def _deterministic_matrix(shape: tuple[int, ...], seed: int) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return np.ascontiguousarray(((values + seed) % 29 - 14) / 29, dtype=np.float32)


def _operation_bytes(package_bytes: bytes) -> bytes:
    fields = struct.unpack_from(RESIDENT_PACKAGE_HEADER_FORMAT, package_bytes)
    offset, length = int(fields[8]), int(fields[9])
    return package_bytes[offset:offset + length]


def _stage_dpu_binary(value: Any, root: Path) -> Path:
    if not value:
        raise ValueError("native_v3_build_invalid: dpu_binary is required")
    source = Path(str(value))
    # The v3 native loader validates the tasklet-keyed basename, for example
    # ``dpu_resident_v3_t3``.  Preserve the build artifact identity when
    # staging it beside the request manifest.
    destination = root / source.name
    if not source.is_file():
        raise ValueError(f"native_v3_build_invalid: dpu_binary does not exist: {source}")
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _write_native_v3_manifest_identity(manifest_path: Path | None) -> None:
    """Record the v3 identity without rewriting the request contract."""

    if manifest_path is None:
        raise ValueError("native_v3_manifest_invalid: manifest path is missing")
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["resident_v3_profile_version"] = RESIDENT_V3_PROFILE_VERSION
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


def _request_id(case: Mapping[str, Any], dpu_count: int, tasklets: int, mode: str, partition: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{case.get('case_id', 'm5')}_{dpu_count}_{tasklets}_{mode}_{partition}")


def _scaling_kind(case: Mapping[str, Any]) -> str:
    diagnostic = str(case.get("diagnostic", "strong_scaling"))
    if diagnostic in {"weak", "weak_scaling"}:
        return "weak_scaling"
    return "strong_scaling"


def _policy_reference_tolerance(quantization_mode: str) -> float:
    del quantization_mode
    return 1.0e-5


def _real_float32_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    output = np.asarray(reference["output"])
    if np.iscomplexobj(output):
        if np.any(output.imag != 0):
            raise ValueError("hardware_profile_violation: v3 reference is not real")
        output = output.real
    return {**reference, "output": np.asarray(output, dtype=np.float32)}


def _reference_metadata(value: Any) -> Any:
    """Remove numerical output arrays while preserving scalar reference evidence."""

    if isinstance(value, Mapping):
        return {
            str(key): _reference_metadata(item)
            for key, item in value.items()
            if key not in {"output", "raw_output"}
        }
    if isinstance(value, (list, tuple)):
        return [_reference_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        raise ValueError(
            "native_v3_request_invalid: unexpected array in reference metadata"
        )
    if isinstance(value, np.generic):
        return value.item()
    return value


def _full_precision_tolerance(quantization_mode: str) -> float:
    return 1.0e-5 if quantization_mode == "none" else 0.25


def _sha256_file(value: Any) -> str:
    if not value:
        raise ValueError("native_v3_build_invalid: binary path is required")
    path = Path(str(value))
    if not path.is_file():
        raise ValueError(f"native_v3_build_invalid: binary does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_binary_binding(
    build: Mapping[str, Any], path_key: str, hash_key: str, label: str
) -> Path:
    value = build.get(path_key)
    expected = build.get(hash_key)
    if not value:
        raise ValueError(f"native_v3_build_invalid: {label} path is required")
    path = Path(str(value))
    if not path.is_file():
        raise ValueError(f"native_v3_build_invalid: {label} does not exist: {path}")
    actual = _sha256_file(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"native_v3_build_invalid: {label} SHA-256 does not match build metadata")
    return path


def _validate_resources(dpu_count: int, tasklets: int) -> None:
    if isinstance(dpu_count, bool) or not 1 <= int(dpu_count) <= MAX_DPUS:
        raise ValueError("hardware_profile_violation: DPU count must be in 1..64")
    if isinstance(tasklets, bool) or not MIN_TASKLETS <= int(tasklets) <= MAX_TASKLETS:
        raise ValueError("hardware_profile_violation: tasklets must be in 1..24")


def _first(values: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values and values[key] is not None:
            return values[key]
    return None


__all__ = ["DEFAULT_TIMEOUT_S", "build", "prepare_request", "validate_request"]
