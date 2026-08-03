from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import warnings

import numpy as np
import pytest

from quantum_bench.circuits import gate_matrix
from quantum_bench.core.records import CircuitOperation, CircuitSpec
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    build_resident_policy_reference,
)
from quantum_bench.targets.upmem.hardware_taskgraph_sliced_resident import (
    SLICED_RESIDENT_M2_3_PROFILE_VERSION,
    build_two_slice_resident_graph_packages,
    build_two_slice_resident_plan,
    load_and_reconstruct_two_slice_native_outputs,
    reconstruct_host_slice_outputs,
    validate_two_slice_resident_plan,
    validate_written_two_slice_packages,
    write_two_slice_resident_graph_packages,
    _m2_reference_real_float32,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    allocate_resident_slots,
)
from quantum_bench.tn import with_execution_identity
from quantum_bench.tn.execution import execute_task_sequence_np_einsum
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.slicing import (
    SliceInputRestriction,
    build_slice_aware_taskgraph_model,
)
from quantum_bench.tn.task_graph import plan_task_graph_with_config

from .support import split_complex_graph


_M2_3_CUSTOM_PLANNER = {
    "engine": "custom_upmem",
    "algorithm": "greedy",
    "objective_version": "upmem_path_cost_v2",
    "selection_scope": "projected_prefix",
    "weight_profile": "balanced_literature_informed",
    "normalization": "fixed_log1p_generic_budgets_v2",
    "execution_policy": "generic_single_dpu_split_complex_v2",
}


def _m2_3_case(angles: tuple[float, float], planner: dict) -> tuple[object, object]:
    circuit = CircuitSpec(
        "m2_3_ry_h_ry",
        1,
        (
            CircuitOperation("ry", (0,), (angles[0],)),
            CircuitOperation("h", (0,)),
            CircuitOperation("ry", (0,), (angles[1],)),
        ),
        {"kind": "fixture", "m2_3": True},
    )
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, planner)
    return graph, network


def _m2_3_dependent_prefix_model(graph):
    model = build_slice_aware_taskgraph_model(
        graph, max_slice_count=2, sliced_task_id=graph.tasks[-1].id
    )
    label = model.sliced_indices[0]
    intermediate_ids = {task.output_tensor_id for task in graph.tasks}
    restrictions = tuple(
        tuple(
            SliceInputRestriction(
                tensor.id,
                label,
                tensor.labels.index(label),
                slice_id,
            )
            for tensor in graph.network.tensors
            if tensor.id not in intermediate_ids and label in tensor.labels
        )
        for slice_id in (0, 1)
    )
    return replace(
        model,
        slice_model_kind="dependent_prefix_replicated",
        slice_tasks=tuple(
            replace(task, input_restrictions=restriction)
            for task, restriction in zip(model.slice_tasks, restrictions, strict=True)
        ),
    )


def _m2_3_plan():
    graph, network = _m2_3_case(
        (np.pi / 5.0, np.pi / 9.0),
        {"engine": "opt_einsum", "optimize": "greedy"},
    )
    model = _m2_3_dependent_prefix_model(graph)
    return build_two_slice_resident_plan(
        graph,
        network,
        model=model,
        profile_version=SLICED_RESIDENT_M2_3_PROFILE_VERSION,
    )


def _replace_slice_task(plan, slice_index: int, slice_task):
    model_tasks = list(plan.model.slice_tasks)
    model_tasks[slice_index] = slice_task
    slice_plans = list(plan.slice_plans)
    slice_plans[slice_index] = replace(
        slice_plans[slice_index], slice_task=slice_task
    )
    return replace(
        plan,
        model=replace(plan.model, slice_tasks=tuple(model_tasks)),
        slice_plans=tuple(slice_plans),
    )


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        (
            (np.pi / 5.0, np.pi / 9.0),
            np.array([0.7986355100472927, 0.6018150231520483]),
        ),
        (
            (np.pi / 9.0, np.pi / 6.0),
            np.array([0.6427876096865393, 0.766044443118978]),
        ),
    ],
)
def test_m2_3_ry_h_ry_cpu_reference(
    angles: tuple[float, float], expected: np.ndarray
) -> None:
    graph, network = _m2_3_case(angles, {"engine": "opt_einsum", "optimize": "greedy"})
    actual, _ = execute_task_sequence_np_einsum(graph, network)

    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    assert graph.path == ((0, 1), (0, 1), (0, 1))
    assert all(np.allclose(gate_matrix("ry", (theta,)).imag, 0.0) for theta in angles)


@pytest.mark.parametrize(
    ("planner", "expected_path"),
    [
        ({"engine": "opt_einsum", "optimize": "greedy"}, ((0, 1), (0, 1), (0, 1))),
        (_M2_3_CUSTOM_PLANNER, ((0, 1), (0, 2), (0, 1))),
    ],
)
@pytest.mark.parametrize("angles", ((np.pi / 5.0, np.pi / 9.0), (np.pi / 9.0, np.pi / 6.0)))
def test_m2_3_paths_are_three_task_same_output_and_distinct_plan(
    planner: dict, expected_path: tuple[tuple[int, int], ...], angles: tuple[float, float]
) -> None:
    graph, network = _m2_3_case(angles, planner)
    actual, _ = execute_task_sequence_np_einsum(graph, network)
    reference, _ = execute_task_sequence_np_einsum(
        *_m2_3_case(angles, {"engine": "opt_einsum", "optimize": "greedy"})
    )

    assert len(graph.tasks) == 3
    assert graph.path == expected_path
    np.testing.assert_allclose(actual, reference, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize("angles", ((np.pi / 5.0, np.pi / 9.0), (np.pi / 9.0, np.pi / 6.0)))
def test_m2_3_path_variants_have_distinct_plan_hashes(
    angles: tuple[float, float],
) -> None:
    greedy, _ = _m2_3_case(
        angles, {"engine": "opt_einsum", "optimize": "greedy"}
    )
    custom, _ = _m2_3_case(angles, _M2_3_CUSTOM_PLANNER)

    assert greedy.path != custom.path
    assert greedy.contraction_plan_hash != custom.contraction_plan_hash


def test_m2_3_profile_accepts_exactly_three_tasks_and_legacy_profile_rejects_them(
    tmp_path,
) -> None:
    graph, network = _m2_3_case(
        (np.pi / 5.0, np.pi / 9.0),
        {"engine": "opt_einsum", "optimize": "greedy"},
    )
    model = _m2_3_dependent_prefix_model(graph)

    plan = build_two_slice_resident_plan(
        graph,
        network,
        model=model,
        profile_version=SLICED_RESIDENT_M2_3_PROFILE_VERSION,
    )
    assert plan.hardware_profile_version == SLICED_RESIDENT_M2_3_PROFILE_VERSION
    assert len(plan.graph.tasks) == 3
    assert validate_two_slice_resident_plan(plan) == (True, None)
    packages = build_two_slice_resident_graph_packages(
        plan,
        case_id="m2_3_fixture",
        suite_id="m2_3_fixture",
        quantization_mode="none",
    )
    dpu_binary = tmp_path / "dpu_resident"
    dpu_binary.write_bytes(b"fixture")
    written = write_two_slice_resident_graph_packages(
        packages,
        tmp_path,
        dpu_binary=dpu_binary,
        request_id_prefix="m2-3-profile",
    )
    for item in written:
        manifest = json.loads(item.package.manifest_path.read_text(encoding="utf-8"))
        assert (
            manifest["slice_execution"]["outer_execution_identity"][
                "hardware_profile_version"
            ]
            == SLICED_RESIDENT_M2_3_PROFILE_VERSION
        )
    assert validate_written_two_slice_packages(plan, written)["validated"] is True

    with pytest.raises(ValueError, match="prefix_task_cap_exceeded"):
        build_two_slice_resident_plan(graph, network, model=model)

    two_task_circuit = CircuitSpec(
        "m2_legacy_h_x",
        1,
        (CircuitOperation("h", (0,)), CircuitOperation("x", (0,))),
        {"kind": "fixture"},
    )
    two_task_network = build_tensor_network(two_task_circuit)
    two_task_graph = plan_task_graph_with_config(
        two_task_network, {"engine": "opt_einsum", "optimize": "greedy"}
    )
    two_task_model = _m2_3_dependent_prefix_model(two_task_graph)
    two_task_plan = build_two_slice_resident_plan(
        two_task_graph, two_task_network, model=two_task_model
    )
    assert len(two_task_plan.graph.tasks) == 2
    assert two_task_plan.hardware_profile_version != SLICED_RESIDENT_M2_3_PROFILE_VERSION


@pytest.mark.parametrize(
    "planner",
    [
        {"engine": "opt_einsum", "optimize": "greedy"},
        _M2_3_CUSTOM_PLANNER,
    ],
)
@pytest.mark.parametrize("angles", ((np.pi / 5.0, np.pi / 9.0), (np.pi / 9.0, np.pi / 6.0)))
def test_m2_3_resident_policy_quantization_error_is_discriminating(
    planner: dict, angles: tuple[float, float]
) -> None:
    graph, network = _m2_3_case(angles, planner)
    model = _m2_3_dependent_prefix_model(graph)
    plan = build_two_slice_resident_plan(
        graph,
        network,
        model=model,
        profile_version=SLICED_RESIDENT_M2_3_PROFILE_VERSION,
    )
    packages = build_two_slice_resident_graph_packages(
        plan,
        case_id="m2_3_fixture",
        suite_id="m2_3_fixture",
        quantization_mode="none",
    )
    full_precision, _ = execute_task_sequence_np_einsum(graph, network)

    outputs = {}
    for mode in ("none", "per_task_resident_requantize"):
        partials = [
            build_resident_policy_reference(
                item.package.graph,
                item.network,
                quantization_mode=mode,
            )["output"]
            for item in packages
        ]
        outputs[mode] = np.sum(np.stack(partials), axis=0)

    assert np.max(np.abs(outputs["none"] - full_precision)) <= 1.0e-6
    quantized_error = float(
        np.max(np.abs(outputs["per_task_resident_requantize"] - full_precision))
    )
    assert 1.0e-4 < quantized_error < 1.0e-2


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda restrictions: (restrictions[0], restrictions[0]),
            "sliced_resident_model_restrictions_mismatch_duplicate_tensor_id",
        ),
        (
            lambda restrictions: (
                replace(restrictions[0], axis=-1),
                restrictions[1],
            ),
            "sliced_resident_model_restrictions_mismatch_axis_invalid",
        ),
        (
            lambda restrictions: (
                replace(restrictions[0], axis=99),
                restrictions[1],
            ),
            "sliced_resident_model_restrictions_mismatch_axis_invalid",
        ),
        (
            lambda restrictions: (None, restrictions[1]),
            "sliced_resident_model_restrictions_mismatch_malformed",
        ),
        (
            lambda restrictions: None,
            "sliced_resident_model_restrictions_mismatch_malformed",
        ),
    ],
)
def test_m2_3_plan_rejects_malformed_restrictions_at_plan_boundary(
    mutation, reason: str
) -> None:
    plan = _m2_3_plan()
    source_task = plan.model.slice_tasks[0]
    malformed = replace(
        source_task,
        input_restrictions=mutation(source_task.input_restrictions),
    )
    malformed_plan = _replace_slice_task(plan, 0, malformed)

    assert validate_two_slice_resident_plan(malformed_plan) == (False, reason)


@pytest.mark.parametrize(
    "dependencies",
    [("task_0",), ("task_1", "task_0")],
)
def test_m2_3_source_dependencies_must_be_complete_and_input_ordered(
    dependencies: tuple[str, ...],
) -> None:
    plan = _m2_3_plan()
    malformed_source = replace(plan.graph.tasks[-1], dependencies=dependencies)
    malformed_graph = with_execution_identity(
        replace(
            plan.graph,
            tasks=(*plan.graph.tasks[:-1], malformed_source),
            circuit_semantics_hash="",
            tensor_network_hash="",
            contraction_plan_hash="",
        )
    )
    model = _m2_3_dependent_prefix_model(malformed_graph)
    malformed_plan = replace(
        plan,
        graph=malformed_graph,
        model=model,
        execution_plan=replace(
            plan.execution_plan,
            circuit_semantics_hash=malformed_graph.circuit_semantics_hash,
            tensor_network_hash=malformed_graph.tensor_network_hash,
            contraction_plan_hash=malformed_graph.contraction_plan_hash,
        ),
        slice_plans=tuple(
            replace(slice_plan, slice_task=slice_task)
            for slice_plan, slice_task in zip(
                plan.slice_plans, model.slice_tasks, strict=True
            )
        ),
        reconstruction_step=model.reconstruction_step,
    )

    assert validate_two_slice_resident_plan(malformed_plan) == (
        False,
        "sliced_resident_source_task_dependencies_mismatch",
    )


@pytest.mark.parametrize(
    ("profile_version", "reason"),
    [
        (None, "sliced_resident_profile_missing"),
        ("unknown_m2_profile", "sliced_resident_profile_unsupported:unknown_m2_profile"),
    ],
)
def test_m2_3_plan_rejects_missing_or_unknown_profile(
    profile_version, reason: str
) -> None:
    plan = replace(_m2_3_plan(), hardware_profile_version=profile_version)

    assert validate_two_slice_resident_plan(plan) == (False, reason)


@pytest.mark.parametrize(
    ("profile_version", "error"),
    [
        (None, "sliced_resident_profile_missing"),
        ("unknown_m2_profile", "sliced_resident_profile_unsupported"),
    ],
)
def test_m2_3_manifest_reader_rejects_missing_or_unknown_profile(
    tmp_path, profile_version, error: str
) -> None:
    plan = _m2_3_plan()
    packages = build_two_slice_resident_graph_packages(
        plan,
        case_id="m2_3_fixture",
        suite_id="m2_3_fixture",
        quantization_mode="none",
    )
    dpu_binary = tmp_path / "dpu_resident"
    dpu_binary.write_bytes(b"fixture")
    written = write_two_slice_resident_graph_packages(
        packages,
        tmp_path,
        dpu_binary=dpu_binary,
        request_id_prefix="m2-3-profile-validation",
    )
    manifest_path = written[0].package.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outer = manifest["slice_execution"]["outer_execution_identity"]
    if profile_version is None:
        outer.pop("hardware_profile_version")
    else:
        outer["hardware_profile_version"] = profile_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        validate_written_two_slice_packages(plan, written)


def _zero_imag_case():
    case = split_complex_graph()
    tensors = tuple(
        replace(tensor, array=np.asarray(tensor.array.real, dtype=np.complex128))
        for tensor in case.network.tensors
    )
    return replace(case, network=TensorNetworkValue(case.network.spec, tensors))


def _written_real_packages(tmp_path):
    case = _zero_imag_case()
    plan = build_two_slice_resident_plan(case.graph, case.network)
    packages = build_two_slice_resident_graph_packages(
        plan,
        case_id="slice-fixture",
        suite_id="slice-suite",
        quantization_mode="none",
    )
    dpu_binary = tmp_path / "dpu_resident"
    dpu_binary.write_bytes(b"fixture")
    return (
        case,
        plan,
        write_two_slice_resident_graph_packages(
            packages,
            tmp_path,
            dpu_binary=dpu_binary,
            request_id_prefix="slice-fixture",
        ),
    )


def _completed_native_response(written, values: tuple[float, float] = (1.0, 2.0)):
    slices = []
    for item, value in zip(written, values, strict=True):
        output = next(iter(item.package.final_output_paths.values()))
        elements = int(np.prod(item.slice_plan.slice_task.output_shape))
        np.full(elements, value, dtype="<f4").tofile(output)
        slices.append(
            {
                "slice_id": item.slice_id,
                "dpu_index": item.dpu_id,
                "allocated": True,
                "release_confirmed": True,
                "manifest_path": str(item.package.manifest_path),
                "manifest_fnv1a64": _c_fnv1a64(item.package.manifest_path.read_bytes()),
                "package_transferred": True,
                "input_count": 2,
                "inputs_transferred": True,
                "partial_output_path": str(output),
                "partial_output_elements": elements,
                "partial_output_bytes": elements * 4,
                "partial_output_read": True,
                "partial_output_written": True,
                "completion_confirmed": True,
            }
        )
    return {
        "schema_version": "generic_loop_resident_two_dpu_contraction_slice_v1",
        "manifest_kind": "resident_two_slice_response",
        "status": "completed",
        "failure_stage": None,
        "cpu_fallback_used": False,
        "topology": "two_dpu_allocation",
        "hardware_execution": True,
        "native_reconstruction_performed": False,
        "reconstruction_contract": "python_sum_partials",
        "allocation": {
            "requested_dpus": 2,
            "allocated_dpus": 2,
            "profile": "backend=hw",
            "verified": True,
        },
        "launch": {
            "mode": "asynchronous",
            "async_launch_count": 1,
            "synchronize_count": 1,
            "completed": True,
        },
        "release": {"attempted": True, "confirmed": True},
        "slices": slices,
    }


def _c_fnv1a64(raw: bytes) -> str:
    value = 14695981039346656037
    for byte in raw:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def test_two_slice_plan_is_deterministic_and_preserves_identity() -> None:
    case = split_complex_graph()
    first = build_two_slice_resident_plan(case.graph, case.network)
    second = build_two_slice_resident_plan(case.graph, case.network)

    assert first.graph is case.graph
    assert first.network is case.network
    assert (
        first.circuit_semantics_hash,
        first.tensor_network_hash,
        first.contraction_plan_hash,
    ) == (
        case.graph.circuit_semantics_hash,
        case.graph.tensor_network_hash,
        case.graph.contraction_plan_hash,
    )
    assert first.to_json_dict() == second.to_json_dict()
    assert [(item.slice_id, item.dpu_id) for item in first.slice_plans] == [
        (0, 0),
        (1, 1),
    ]
    assert first.execution_plan.placement.resources.requested_dpu_count == 2
    assert first.execution_plan.placement.topology == "two_dpu_allocation"
    assert validate_two_slice_resident_plan(first) == (True, None)


def test_slice_packages_materialize_restricted_inputs_and_bind_manifests(
    tmp_path,
) -> None:
    case = _zero_imag_case()
    source_hashes = (
        case.graph.circuit_semantics_hash,
        case.graph.tensor_network_hash,
        case.graph.contraction_plan_hash,
    )
    plan = build_two_slice_resident_plan(case.graph, case.network)
    packages = build_two_slice_resident_graph_packages(
        plan,
        case_id="slice-fixture",
        suite_id="slice-suite",
        quantization_mode="none",
    )

    assert [item.slice_id for item in packages] == [0, 1]
    assert (
        case.graph.circuit_semantics_hash,
        case.graph.tensor_network_hash,
        case.graph.contraction_plan_hash,
    ) == source_hashes
    left_inputs = [item.network.tensors[0].array for item in packages]
    right_inputs = [item.network.tensors[1].array for item in packages]
    assert all(array.shape == (2, 1) for array in left_inputs)
    assert all(array.shape == (1, 2) for array in right_inputs)
    assert not np.array_equal(left_inputs[0], left_inputs[1])
    assert not np.array_equal(right_inputs[0], right_inputs[1])
    for item in packages:
        task = item.package.graph.tasks[0]
        assert task.input_shapes == ((2, 1), (1, 2))
        assert task.output_shape == case.graph.tasks[0].output_shape
        assert task.gemm_k == 1
        assert all(
            operation.args["contracted_dims"] == (1,)
            for operation in item.package.operations
            if operation.kind == "contract"
        )
        assert (
            item.package.graph.contraction_plan_hash != case.graph.contraction_plan_hash
        )
        assert all(tensor.spec.dtype == "float32" for tensor in item.network.tensors)
        assert all(tensor.array.dtype == np.float32 for tensor in item.network.tensors)

    dpu_binary = tmp_path / "dpu_resident"
    dpu_binary.write_bytes(b"fixture")
    written = write_two_slice_resident_graph_packages(
        packages, tmp_path, dpu_binary=dpu_binary, request_id_prefix="slice-fixture"
    )
    manifests = [
        json.loads(item.package.manifest_path.read_text(encoding="utf-8"))
        for item in written
    ]
    assert len({item.package.package_path for item in written}) == 2
    assert len({item.package.manifest_path for item in written}) == 2
    assert len(
        {path for item in written for path in item.package.final_output_paths.values()}
    ) == sum(len(item.package.final_output_paths) for item in written)
    for item, manifest in zip(written, manifests, strict=True):
        binding = manifest["slice_execution"]
        assert binding["slice_id"] == item.slice_id
        assert binding["dpu_id"] == item.dpu_id
        assert binding["source_hashes"] == {
            "circuit_semantics_hash": case.graph.circuit_semantics_hash,
            "tensor_network_hash": case.graph.tensor_network_hash,
            "contraction_plan_hash": case.graph.contraction_plan_hash,
        }
        assert binding["restricted_input_sha256"] == item.restricted_input_sha256
        assert binding["restricted_input_sha256"] == {
            entry["input_path"]: hashlib.sha256(
                (item.package.manifest_path.parent / entry["input_path"]).read_bytes()
            ).hexdigest()
            for entry in manifest["initial_slots"]
        }
        assert binding["restricted_input_fnv1a64"] == item.restricted_input_fnv1a64
        assert (
            binding["resident_descriptor_sha256"]
            == hashlib.sha256(item.package.package_path.read_bytes()).hexdigest()
        )
        assert (
            binding["resident_descriptor_fnv1a64"] == item.resident_descriptor_fnv1a64
        )
        assert binding["resident_descriptor_fnv1a64"] == _c_fnv1a64(
            item.package.package_path.read_bytes()
        )
        assert binding["reconstruction_contract"] == "python_sum_partials"

    partials = {
        item.slice_id: execute_task_sequence_np_einsum(
            item.package.graph, item.network
        )[0]
        for item in packages
    }
    expected = execute_task_sequence_np_einsum(case.graph, case.network)[0]
    np.testing.assert_array_equal(
        reconstruct_host_slice_outputs(plan, partials), expected
    )


def test_m2_rejects_nonzero_complex_source_inputs() -> None:
    case = split_complex_graph()
    plan = build_two_slice_resident_plan(case.graph, case.network)

    with pytest.raises(ValueError, match="m2_nonzero_imaginary_source_input"):
        build_two_slice_resident_graph_packages(
            plan,
            case_id="slice-fixture",
            suite_id="slice-suite",
            quantization_mode="none",
        )


def test_mixed_complex_inputs_synthesize_zero_imaginary_operand_slots() -> None:
    case = split_complex_graph()
    tensors = list(case.network.tensors)
    source_indices = [
        index
        for index, tensor in enumerate(tensors)
        if tensor.spec.produced_by is None
    ]
    assert len(source_indices) >= 2
    tensors[source_indices[0]] = replace(
        tensors[source_indices[0]],
        array=np.asarray(tensors[source_indices[0]].array.real, dtype=np.complex128),
    )
    mixed = replace(
        case,
        network=TensorNetworkValue(case.network.spec, tuple(tensors)),
    )
    allocation = allocate_resident_slots(mixed.graph, mixed.network)

    zero_imag_id = tensors[source_indices[0]].spec.id
    genuine_complex_ids = {
        tensors[index].spec.id
        for index in source_indices[1:]
        if np.any(np.asarray(tensors[index].array).imag != 0.0)
    }
    assert genuine_complex_ids
    assert "imag" in allocation.tensor_components[zero_imag_id]
    assert all("imag" in allocation.tensor_components[item] for item in genuine_complex_ids)


def test_written_packages_validate_actual_inputs_and_descriptor_bytes(tmp_path) -> None:
    _case, plan, written = _written_real_packages(tmp_path)

    metadata = validate_written_two_slice_packages(plan, written)

    assert metadata["validated"] is True
    assert set(metadata["slices"]) == {"0", "1"}
    assert all(
        len(item["resident_descriptor_fnv1a64"]) == 16
        for item in metadata["slices"].values()
    )


@pytest.mark.parametrize("target", ["input", "package"])
def test_written_packages_reject_stale_input_or_descriptor(
    tmp_path, target: str
) -> None:
    _case, plan, written = _written_real_packages(tmp_path)
    package = written[0].package
    if target == "input":
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
        path = package.manifest_path.parent / manifest["initial_slots"][0]["input_path"]
    else:
        path = package.package_path
    path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(
        ValueError,
        match="(slice_execution_mismatch|descriptor_sha256_mismatch)",
    ):
        validate_written_two_slice_packages(plan, written)


def test_load_and_reconstruct_native_two_slice_outputs(tmp_path) -> None:
    _case, plan, written = _written_real_packages(tmp_path)
    response = _completed_native_response(written)
    response_path = tmp_path / "native-response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")

    actual, metadata = load_and_reconstruct_two_slice_native_outputs(
        plan, written, response_path
    )

    np.testing.assert_array_equal(actual, np.full((2, 2), 3.0, dtype=np.float32))
    assert set(metadata["partial_outputs"]) == {"0", "1"}
    assert len(metadata["native_response_fnv1a64"]) == 16


def test_policy_reference_zero_imaginary_complex_conversion_is_warning_free() -> None:
    value = np.asarray([1.0, 2.0], dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        converted = _m2_reference_real_float32(value)

    np.testing.assert_array_equal(converted, np.asarray([1.0, 2.0], dtype=np.float32))


def test_policy_reference_nonzero_imaginary_conversion_is_rejected() -> None:
    value = np.asarray([1.0 + 1.0e-5j], dtype=np.complex128)

    with pytest.raises(
        ValueError, match="sliced_resident_cpu_partial_reference_nonzero_imaginary"
    ):
        _m2_reference_real_float32(value)


def test_policy_reference_near_zero_imaginary_conversion_is_accepted() -> None:
    value = np.asarray([1.0 + 1.0e-7j], dtype=np.complex128)

    converted = _m2_reference_real_float32(value)

    np.testing.assert_array_equal(converted, np.asarray([1.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("status", "failed", "response_status_invalid"),
        ("cpu_fallback_used", True, "response_status_invalid"),
        ("slices.0.partial_output_path", "wrong.bin", "partial_output_path_mismatch"),
        ("slices.1.completion_confirmed", False, "slice_evidence_invalid"),
    ],
)
def test_native_response_rejects_wrong_status_or_path(
    tmp_path, field, value, error
) -> None:
    _case, plan, written = _written_real_packages(tmp_path)
    response = _completed_native_response(written)
    parts = field.split(".")
    if len(parts) == 1:
        response[field] = value
    else:
        response[parts[0]][int(parts[1])][parts[2]] = value

    with pytest.raises(ValueError, match=error):
        load_and_reconstruct_two_slice_native_outputs(plan, written, response)


def test_host_reconstruction_sums_partials_in_slice_order() -> None:
    case = split_complex_graph()
    plan = build_two_slice_resident_plan(case.graph, case.network)

    actual = reconstruct_host_slice_outputs(
        plan,
        {
            0: np.ones((2, 2), dtype=np.complex64),
            1: 2 * np.ones((2, 2), dtype=np.complex64),
        },
    )

    np.testing.assert_array_equal(actual, 3 * np.ones((2, 2), dtype=np.complex64))
    with pytest.raises(ValueError, match="slice_ids_0_and_1"):
        reconstruct_host_slice_outputs(plan, {0: np.ones((2, 2))})
    with pytest.raises(ValueError, match="partial_shape_mismatch"):
        reconstruct_host_slice_outputs(
            plan,
            {
                0: np.ones((1, 4)),
                1: 2 * np.ones((1, 4)),
            },
        )


def test_dependent_selected_task_is_rejected() -> None:
    case = split_complex_graph()
    model = build_slice_aware_taskgraph_model(case.graph, max_slice_count=2)
    dependent = model.slice_tasks[0].__class__(
        **{**model.slice_tasks[0].__dict__, "dependencies": ("upstream",)}
    )
    dependent_model = model.__class__(
        **{**model.__dict__, "slice_tasks": (dependent, model.slice_tasks[1])}
    )

    with pytest.raises(ValueError, match="dependent_selected_task"):
        build_two_slice_resident_plan(case.graph, case.network, model=dependent_model)


def test_foreign_model_is_rejected() -> None:
    case = split_complex_graph()
    foreign_task = replace(
        case.graph.tasks[0], id="foreign_task", output_tensor_id="foreign_out"
    )
    foreign_graph = with_execution_identity(
        replace(
            case.graph,
            tasks=(foreign_task,),
            circuit_semantics_hash="",
            tensor_network_hash="",
            contraction_plan_hash="",
        )
    )
    foreign_model = build_slice_aware_taskgraph_model(foreign_graph, max_slice_count=2)

    with pytest.raises(ValueError, match="model_source_task_mismatch"):
        build_two_slice_resident_plan(case.graph, case.network, model=foreign_model)


def test_downstream_task_is_rejected() -> None:
    case = split_complex_graph()
    downstream = replace(
        case.graph.tasks[0],
        id="downstream_task",
        input_tensor_ids=(case.graph.tasks[0].output_tensor_id, "right"),
        output_tensor_id="downstream_out",
        dependencies=(case.graph.tasks[0].id,),
    )
    graph = with_execution_identity(
        replace(
            case.graph,
            tasks=(case.graph.tasks[0], downstream),
            circuit_semantics_hash="",
            tensor_network_hash="",
            contraction_plan_hash="",
        )
    )

    with pytest.raises(ValueError, match="terminal_single_task_graph"):
        build_two_slice_resident_plan(graph, case.network)


def test_slice_assignment_value_must_match_slice_id() -> None:
    case = split_complex_graph()
    model = build_slice_aware_taskgraph_model(case.graph, max_slice_count=2)
    bad_second = replace(
        model.slice_tasks[1],
        assignment=replace(model.slice_tasks[1].assignment, value=0),
    )
    bad_model = replace(model, slice_tasks=(model.slice_tasks[0], bad_second))

    with pytest.raises(ValueError, match="assignment_values_must_match_slice_ids"):
        build_two_slice_resident_plan(case.graph, case.network, model=bad_model)


def test_overlapping_slice_restrictions_are_rejected() -> None:
    case = split_complex_graph()
    model = build_slice_aware_taskgraph_model(case.graph, max_slice_count=2)
    first = model.slice_tasks[0]
    overlapping = replace(
        first,
        input_restrictions=(first.input_restrictions[0], first.input_restrictions[0]),
    )
    bad_model = replace(
        model,
        slice_tasks=(overlapping, model.slice_tasks[1]),
    )

    with pytest.raises(ValueError, match="restrictions_mismatch"):
        build_two_slice_resident_plan(case.graph, case.network, model=bad_model)


def test_slice_plan_task_must_match_model_task() -> None:
    case = split_complex_graph()
    plan = build_two_slice_resident_plan(case.graph, case.network)
    mutated_task = replace(
        plan.slice_plans[0].slice_task,
        input_restrictions=(),
    )
    mutated_slice_plan = replace(plan.slice_plans[0], slice_task=mutated_task)
    mutated_plan = replace(
        plan,
        slice_plans=(mutated_slice_plan, plan.slice_plans[1]),
    )

    assert validate_two_slice_resident_plan(mutated_plan) == (
        False,
        "slice_task_model_mismatch",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("operation", "product_partials"), ("output_tensor_id", "wrong_output")],
)
def test_reconstruction_step_must_match_model_step(field: str, value: str) -> None:
    case = split_complex_graph()
    plan = build_two_slice_resident_plan(case.graph, case.network)
    mutated_step = replace(plan.reconstruction_step, **{field: value})
    mutated_plan = replace(plan, reconstruction_step=mutated_step)

    assert validate_two_slice_resident_plan(mutated_plan) == (
        False,
        "reconstruction_step_model_mismatch",
    )
