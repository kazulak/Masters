from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from quantum_bench.targets.upmem.hardware_taskgraph_sliced_resident import (
    build_two_slice_resident_graph_packages,
    build_two_slice_resident_plan,
    load_and_reconstruct_two_slice_native_outputs,
    reconstruct_host_slice_outputs,
    validate_two_slice_resident_plan,
    validate_written_two_slice_packages,
    write_two_slice_resident_graph_packages,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    allocate_resident_slots,
)
from quantum_bench.tn import with_execution_identity
from quantum_bench.tn.execution import execute_task_sequence_np_einsum
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.tn.slicing import build_slice_aware_taskgraph_model

from .support import split_complex_graph


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
